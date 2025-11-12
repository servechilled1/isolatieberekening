# app.py
import os
import io
import random
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
from collections import OrderedDict, defaultdict

# ===== externe libs =====
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pandas as pd
from fpdf import FPDF
import streamlit as st

# ---- DXF lib (optioneel) ----
try:
    import ezdxf  # pip install ezdxf
    EZDXF_AVAILABLE = True
except Exception:
    EZDXF_AVAILABLE = False

# ========= Streamlit config – MOET eerste Streamlit-call zijn =========
st.set_page_config(layout="wide", page_title="Plaatoptimalisatie Tool")

"""
Plaatoptimalisatie Tool
-----------------------
• Max-rectangles + Shelf (rij) algoritme; beide worden geprobeerd, beste resultaat wordt gekozen.  
• Elk onderdeel mag 90° draaien; per plaatsing kiest de code de beste oriëntatie.  
• Multi-run + meerdere heuristieken (area/short/long/combined) voor Max-Rects (of 'all' om alles te proberen).  
• Instelbare kerf (zaagspleet) – toegepast aan de rechter- en onderzijde, zodat buitenranden strak blijven.  
• DXF import (gesloten (LW)POLYLINE / 2D POLYLINE → bounding box → gegroepeerd).  
• Onderdelen hebben **dikte** en **type isolatie**; nesting gebeurt per combinatie daarvan.  
• Overzicht per materiaal + dikte in UI én PDF, plus per-plaat visualisatie en CSV-export.  
"""

# ========= Data modellen =========
@dataclass
class Part:
    label: str
    w: int
    h: int
    qty: int
    color: str
    thickness: float = 0.0
    material: str = ""

    def area(self) -> int:
        return int(self.w) * int(self.h)

    def copy(self) -> "Part":
        return Part(
            self.label,
            int(self.w),
            int(self.h),
            int(self.qty),
            self.color,
            float(self.thickness),
            str(self.material),
        )


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h


# ========= Packing helpers =========
def _apply_kerf(w: int, h: int, kerf: int) -> Tuple[int, int]:
    return w + kerf, h + kerf  # kerf rechts + onder


def _fits(fr: Rect, w: int, h: int) -> bool:
    return w <= fr.w and h <= fr.h


def _split_free_rect(fr: Rect, used: Dict[str, int]) -> Tuple[List[Rect], bool]:
    new_rects: List[Rect] = []
    if (
        used["x"] >= fr.x + fr.w
        or used["x"] + used["w"] <= fr.x
        or used["y"] >= fr.y + fr.h
        or used["y"] + used["h"] <= fr.y
    ):
        return [fr], False

    # boven
    if used["y"] > fr.y:
        new_rects.append(Rect(fr.x, fr.y, fr.w, used["y"] - fr.y))

    # onder
    bottom = used["y"] + used["h"]
    if bottom < fr.y + fr.h:
        new_rects.append(Rect(fr.x, bottom, fr.w, (fr.y + fr.h) - bottom))

    # links
    if used["x"] > fr.x:
        new_rects.append(
            Rect(
                fr.x,
                max(fr.y, used["y"]),
                used["x"] - fr.x,
                min(fr.y + fr.h, used["y"] + used["h"]) - max(fr.y, used["y"]),
            )
        )

    # rechts
    right_x = used["x"] + used["w"]
    if right_x < fr.x + fr.w:
        new_rects.append(
            Rect(
                right_x,
                max(fr.y, used["y"]),
                (fr.x + fr.w) - right_x,
                min(fr.y + fr.h, used["y"] + used["h"]) - max(fr.y, used["y"]),
            )
        )

    return new_rects, True


def _prune(rects: List[Rect]) -> List[Rect]:
    pruned: List[Rect] = []
    for i, r in enumerate(rects):
        contained = False
        for j, r2 in enumerate(rects):
            if (
                i != j
                and r.x >= r2.x
                and r.y >= r2.y
                and r.x + r.w <= r2.x + r2.w
                and r.y + r.h <= r2.y + r2.h
            ):
                contained = True
                break
        if not contained and r.w > 0 and r.h > 0:
            pruned.append(r)
    return pruned


def _choose_best_fit(
    free_rects: List[Rect], w: int, h: int, kerf: int, heuristic: str
):
    best = None
    for idx, fr in enumerate(free_rects):
        rotations = ((w, h), (h, w)) if w != h else ((w, h),)
        for (cw, ch) in rotations:
            w_eff, h_eff = _apply_kerf(cw, ch, kerf)
            if _fits(fr, w_eff, h_eff):
                horiz_waste = fr.w - cw
                vert_waste = fr.h - ch
                area_waste = fr.area() - (cw * ch)
                pieces_in_row = (fr.w // cw) if cw > 0 else 0

                if heuristic == "area":
                    score = (area_waste, horiz_waste, vert_waste)
                elif heuristic == "short":
                    score = (
                        min(horiz_waste, vert_waste),
                        area_waste,
                        max(horiz_waste, vert_waste),
                    )
                elif heuristic == "long":
                    score = (
                        max(horiz_waste, vert_waste),
                        area_waste,
                        min(horiz_waste, vert_waste),
                    )
                else:  # combined
                    score = (-pieces_in_row, horiz_waste, vert_waste, area_waste)

                if best is None or score < best["score"]:
                    best = dict(
                        idx=idx,
                        w=cw,
                        h=ch,
                        w_eff=w_eff,
                        h_eff=h_eff,
                        score=score,
                    )
    return best


def pack_plate_maxrects(
    W: int, H: int, parts_left: List[Part], kerf: int, heuristic: str = "area"
):
    free_rects: List[Rect] = [Rect(0, 0, W, H)]
    placed = []
    parts_left.sort(key=lambda p: p.area(), reverse=True)

    for p in parts_left:
        while p.qty > 0:
            best = None
            rotations = ((p.w, p.h), (p.h, p.w)) if p.w != p.h else ((p.w, p.h),)
            for (w, h) in rotations:
                cand = _choose_best_fit(free_rects, w, h, kerf, heuristic)
                if cand and (best is None or cand["score"] < best["score"]):
                    best = cand
            if best is None:
                break

            fr = free_rects[best["idx"]]
            used = dict(x=fr.x, y=fr.y, w=best["w"], h=best["h"])
            placed.append(
                dict(
                    **used,
                    label=p.label,
                    color=p.color,
                    thickness=p.thickness,
                    material=p.material,
                )
            )

            used_expanded = dict(
                x=used["x"], y=used["y"], w=best["w_eff"], h=best["h_eff"]
            )
            new_free: List[Rect] = []
            for fr2 in free_rects:
                split_rects, did = _split_free_rect(fr2, used_expanded)
                new_free.extend(split_rects if did else [fr2])
            free_rects = _prune(new_free)
            p.qty -= 1

    return placed, parts_left


def pack_plate_shelf(W: int, H: int, parts_left: List[Part], kerf: int):
    placed = []
    x = y = 0
    row_h = 0
    parts_left.sort(key=lambda p: max(p.w, p.h), reverse=True)

    for p in parts_left:
        while p.qty > 0:
            candidates = []
            rotations = ((p.w, p.h), (p.h, p.w)) if p.w != p.h else ((p.w, p.h),)
            for (cw, ch) in rotations:
                w_eff, h_eff = _apply_kerf(cw, ch, kerf)
                if (x + w_eff) <= W and (y + h_eff) <= H:
                    pieces_in_row = (W // cw) if cw > 0 else 0
                    horiz_waste = W - (x + cw)
                    candidates.append(
                        (-pieces_in_row, horiz_waste, cw, ch, w_eff, h_eff)
                    )

            if candidates:
                candidates.sort()
                _, _, cw, ch, w_eff, h_eff = candidates[0]
                placed.append(
                    dict(
                        x=x,
                        y=y,
                        w=cw,
                        h=ch,
                        label=p.label,
                        color=p.color,
                        thickness=p.thickness,
                        material=p.material,
                    )
                )
                x += w_eff
                row_h = max(row_h, h_eff)
                p.qty -= 1
            else:
                x = 0
                y += row_h
                row_h = 0
                if y + min(p.w, p.h) > H:
                    return placed, parts_left

    return placed, parts_left


def pack_all(
    W: int,
    H: int,
    parts: List[Part],
    kerf: int,
    heuristic: str = "combined",
    runs: int = 20,
):
    # 1) Shelf baseline
    shelf_parts = [p.copy() for p in parts]
    plates_shelf = []
    while any(p.qty > 0 for p in shelf_parts):
        pl, shelf_parts = pack_plate_shelf(W, H, shelf_parts, kerf)
        if not pl:
            break
        plates_shelf.append(pl)

    rest_shelf = (
        sum((W * H - sum(r["w"] * r["h"] for r in pl)) for pl in plates_shelf)
        if plates_shelf
        else W * H
    )
    score_shelf = (
        len(plates_shelf) if plates_shelf else float("inf"),
        rest_shelf,
    )

    # 2) Max-rects multi-run
    best_mr = None
    random.seed(42)
    heur_to_try = (
        [heuristic]
        if heuristic != "all"
        else ["combined", "area", "short", "long"]
    )

    for _ in range(runs):
        base = [p.copy() for p in parts]
        random.shuffle(base)
        for h in heur_to_try:
            parts_h = [p.copy() for p in base]
            plates_h = []
            safety_guard = 0
            while any(p.qty > 0 for p in parts_h):
                pl, parts_h = pack_plate_maxrects(W, H, parts_h, kerf, heuristic=h)
                if not pl:
                    break
                plates_h.append(pl)
                safety_guard += 1
                if safety_guard > 1000:
                    break
            if not plates_h:
                continue

            total_rest = sum(
                (W * H - sum(r["w"] * r["h"] for r in pl)) for pl in plates_h
            )
            score = (len(plates_h), total_rest)
            if best_mr is None or score < best_mr["score"]:
                best_mr = dict(plates=plates_h, score=score)

    if best_mr and best_mr["score"] < score_shelf:
        return best_mr["plates"]
    return plates_shelf


# ========= Groeperen & statistiek =========
def plate_signature(placed: Iterable[Dict], group_key=None) -> Tuple:
    base = tuple(
        sorted((r["x"], r["y"], r["w"], r["h"], r["label"]) for r in placed)
    )
    return (group_key, base)


def compute_global_stats(
    plates: List[List[Dict]], W: int, H: int, parts: List[Part]
) -> Dict[str, float]:
    total_plates = len(plates)
    total_plate_area = W * H * total_plates
    used_area = sum(r["w"] * r["h"] for pl in plates for r in pl)
    utilisation = (
        (used_area / total_plate_area * 100) if total_plate_area > 0 else 0.0
    )
    waste_pct = 100 - utilisation
    parts_total_area_mm2 = sum(p.area() * p.qty for p in parts)
    parts_total_area_m2 = parts_total_area_mm2 / 1_000_000.0

    return dict(
        total_plates=total_plates,
        total_plate_area=total_plate_area,
        used_area=used_area,
        utilisation=utilisation,
        waste_pct=waste_pct,
        parts_total_area_mm2=parts_total_area_mm2,
        parts_total_area_m2=parts_total_area_m2,
    )


def find_oversized_parts(
    parts: List[Part], W: int, H: int, kerf: int
) -> List[Part]:
    too_big: List[Part] = []
    for p in parts:
        w1, h1 = _apply_kerf(p.w, p.h, kerf)
        w2, h2 = _apply_kerf(p.h, p.w, kerf)
        fits_normal = w1 <= W and h1 <= H
        fits_rot = w2 <= W and h2 <= H
        if not (fits_normal or fits_rot):
            too_big.append(p)
    return too_big


# ========= tekenen / PDF / CSV =========
def draw_plate_png(
    placed,
    plate_no: int,
    W: int,
    H: int,
    grid_step: int,
    rest_pct: float,
    count: int = 1,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    if grid_step > 0:
        for gx in range(0, W + 1, grid_step):
            ax.axvline(gx, color="lightgray", lw=0.4)
        for gy in range(0, H + 1, grid_step):
            ax.axhline(gy, color="lightgray", lw=0.4)

    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="black", lw=1.5))

    legend: Dict[str, str] = {}
    for r in placed:
        rect = Rectangle(
            (r["x"], r["y"]),
            r["w"],
            r["h"],
            fc=r["color"],
            ec="black",
            lw=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            r["x"] + r["w"] / 2,
            r["y"] + r["h"] / 2,
            f"{r['w']}x{r['h']}",
            ha="center",
            va="center",
            fontsize=8,
        )
        legend[r["label"]] = r["color"]

    title_suffix = f" x{count}" if count > 1 else ""
    ax.set_title(
        f"Plaat {plate_no}{title_suffix} - {W}x{H} mm | Rest: {rest_pct:.1f}%",
        fontsize=13,
        weight="bold",
        pad=8,
    )

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf, legend


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def pdf_safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_from_pages(plate_pages: List[Dict], parts: List[Part]) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)

    margin = 10
    page_w = 210
    usable_w = page_w - 2 * margin

    # totaal platen per groep
    total_plates_by_group: Dict[Tuple[float, str], int] = defaultdict(int)
    for page in plate_pages:
        th = page.get("thickness", 0.0)
        mat = page.get("material", "")
        total_plates_by_group[(th, mat)] += int(page.get("count", 0))

    # parts per groep
    parts_by_group: Dict[Tuple[float, str], List[Part]] = defaultdict(list)
    for p in parts:
        key = (float(p.thickness), str(p.material))
        parts_by_group[key].append(p)

    # === overzicht per materiaal/dikte ===
    pdf.add_page()
    pdf.set_xy(margin, margin)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        usable_w,
        8,
        pdf_safe("Plaatoptimalisatie - Overzicht per materiaal"),
        ln=1,
        align="C",
    )

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        usable_w,
        5,
        pdf_safe(
            "Per materiaal en dikte: totaal verbruikte platen en alle onderdelen (maatvoeringen)."
        ),
    )
    pdf.ln(3)

    col_label_w = usable_w * 0.30
    col_w_w     = usable_w * 0.15
    col_h_w     = usable_w * 0.15
    col_qty_w   = usable_w * 0.15
    col_area_w  = usable_w * 0.25

    groups_sorted = sorted(
        parts_by_group.keys(),
        key=lambda k: (str(k[1]), float(k[0])),
    )

    for (th, mat) in groups_sorted:
        group_parts = parts_by_group[(th, mat)]
        total_pl = total_plates_by_group.get((th, mat), 0)

        pdf.set_font("Helvetica", "B", 11)
        pdf.ln(3)
        pdf.cell(usable_w, 6, pdf_safe(f"Materiaal: {mat or '-'}"), ln=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(
            usable_w,
            5,
            pdf_safe(f"Totaal verbruikte platen: {total_pl}"),
            ln=1,
        )
        pdf.cell(
            usable_w,
            5,
            pdf_safe(f"Dikte: {th} mm"),
            ln=1,
        )
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(margin)
        pdf.cell(col_label_w, 6, pdf_safe("Label"),          border=1)
        pdf.cell(col_w_w,     6, pdf_safe("Breedte (mm)"),   border=1)
        pdf.cell(col_h_w,     6, pdf_safe("Hoogte (mm)"),    border=1)
        pdf.cell(col_qty_w,   6, pdf_safe("Aantal"),         border=1)
        pdf.cell(col_area_w,  6, pdf_safe("Totaal opp. (m2)"), border=1, ln=1)

        pdf.set_font("Helvetica", "", 9)
        for p in sorted(group_parts, key=lambda x: x.label):
            total_area_mm2 = p.area() * p.qty
            total_area_m2 = total_area_mm2 / 1_000_000.0
            pdf.set_x(margin)
            pdf.cell(col_label_w, 5, pdf_safe(str(p.label)),  border=1)
            pdf.cell(col_w_w,     5, pdf_safe(str(p.w)),      border=1)
            pdf.cell(col_h_w,     5, pdf_safe(str(p.h)),      border=1)
            pdf.cell(col_qty_w,   5, pdf_safe(str(p.qty)),    border=1)
            pdf.cell(col_area_w,  5, pdf_safe(f"{total_area_m2:.2f}"), border=1, ln=1)

    # === per plaat-type pagina ===
    img_h = 110

    for page in plate_pages:
        pdf.add_page()

        title = f"Plaat-type {page['plate_no']} x{page['count']}"
        extra = []
        th = page.get("thickness", 0.0)
        mat = page.get("material", "")
        if th and th != 0.0:
            extra.append(f"{th}mm")
        if mat:
            extra.append(mat)
        if extra:
            title += " - " + " ".join(extra)

        pdf.set_xy(margin, margin)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(usable_w, 8, pdf_safe(title), ln=1, align="C")

        img_y = pdf.get_y() + 2
        pdf.image(page["png_path"], x=margin, y=img_y, w=usable_w, h=img_h)

        y = img_y + img_h + 5
        pdf.set_xy(margin, y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(usable_w, 6, pdf_safe("Onderdelen overzicht (per plaat)"), ln=1)

        rows = page["rows"]
        sorted_rows = sorted(rows.items(), key=lambda kv: kv[0])
        col1_w = usable_w * 0.7
        col2_w = usable_w * 0.3

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(margin, pdf.get_y() + 1)
        pdf.cell(col1_w, 5, pdf_safe("Onderdeel (afm)"), border=1)
        pdf.cell(col2_w, 5, pdf_safe("Aantal"),          border=1, ln=1)

        pdf.set_font("Helvetica", "", 9)
        for label, qty in sorted_rows:
            pdf.set_x(margin)
            pdf.cell(col1_w, 5, pdf_safe(label),    border=1)
            pdf.cell(col2_w, 5, pdf_safe(str(qty)), border=1, ln=1)

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(
            usable_w,
            5,
            pdf_safe(f"Restmateriaal (per plaat): {page['rest_pct']:.1f}%"),
            ln=1,
        )
        pdf.cell(
            usable_w,
            5,
            pdf_safe(f"Aantal uit te voeren platen van dit type: {page['count']}"),
            ln=1,
        )

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(usable_w, 5, pdf_safe("Legenda"), ln=1)

        pdf.set_font("Helvetica", "", 9)
        for lbl, clr in page["legend"].items():
            r, g, b = hex_to_rgb(clr)
            y = pdf.get_y()
            pdf.set_xy(margin, y)
            pdf.set_fill_color(r, g, b)
            pdf.rect(margin, y, 4, 4, style="F")
            pdf.set_xy(margin + 6, y)
            pdf.cell(usable_w - 6, 4, pdf_safe(lbl), ln=1)

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out.read()


def build_csv_from_plates(grouped, W: int, H: int) -> bytes:
    rows = []
    for plate_no, item in enumerate(grouped.values(), start=1):
        placed = item["placed"]
        copies = item["count"]
        thickness = item.get("thickness", 0.0)
        material = item.get("material", "")
        for r in placed:
            rows.append(
                dict(
                    plaat_type=plate_no,
                    copies=copies,
                    label=r["label"],
                    x=r["x"],
                    y=r["y"],
                    w=r["w"],
                    h=r["h"],
                    dikte_mm=thickness,
                    type_isolatie=material,
                    plaat_breedte=W,
                    plaat_hoogte=H,
                )
            )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# ========= DXF import =========
def _bbox_from_vertices(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return maxx - minx, maxy - miny


def _rounded_int_mm(val: float) -> int:
    return int(round(val))


def parse_dxf_files_to_parts(
    uploaded_files, scale_to_mm: float = 1.0, min_area_mm2: float = 1.0
) -> List[Tuple[int, int, int]]:
    size_counter: Dict[Tuple[int, int], int] = defaultdict(int)

    for f in uploaded_files:
        try:
            data = f.read()
            f.seek(0)
            doc = ezdxf.read(io.BytesIO(data))
            msp = doc.modelspace()
        except Exception as e:
            st.warning(
                f"Kon DXF '{getattr(f, 'name', 'unknown')}' niet lezen: {e}"
            )
            continue

        for e in msp:
            try:
                dxf_type = e.dxftype()
                if dxf_type == "LWPOLYLINE":
                    closed = bool(getattr(e, "closed", False))
                    if not closed:
                        continue
                    verts = [(p[0], p[1]) for p in e.get_points()]
                elif dxf_type == "POLYLINE":
                    closed = bool(
                        getattr(e, "is_closed", False)
                        or getattr(e, "closed", False)
                    )
                    if not closed:
                        continue
                    verts = [
                        (v.dxf.location.x, v.dxf.location.y)
                        for v in e.vertices  # type: ignore
                    ]
                else:
                    continue

                if len(verts) < 3:
                    continue

                w_u, h_u = _bbox_from_vertices(verts)
                w_mm = w_u * scale_to_mm
                h_mm = h_u * scale_to_mm
                area = w_mm * h_mm
                if area < min_area_mm2:
                    continue

                w_i = _rounded_int_mm(w_mm)
                h_i = _rounded_int_mm(h_mm)
                key = tuple(sorted((w_i, h_i)))
                size_counter[key] += 1
            except Exception:
                continue

    parts_list: List[Tuple[int, int, int]] = []
    for (a, b), qty in sorted(
        size_counter.items(),
        key=lambda x: (x[0][0] * x[0][1], x[0][0], x[0][1]),
        reverse=True,
    ):
        parts_list.append((a, b, qty))
    return parts_list


# ========= UI helpers =========
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Instellingen")
        W = st.number_input(
            "Plaatbreedte (mm)",
            min_value=100,
            max_value=20000,
            value=3000,
            step=10,
            key="sb_plaatbreedte",
        )
        H = st.number_input(
            "Plaathoogte (mm)",
            min_value=100,
            max_value=20000,
            value=1500,
            step=10,
            key="sb_plaathoogte",
        )

        st.markdown("**Raster & kerf**")
        grid = st.number_input(
            "Grid (mm)",
            min_value=0,
            max_value=1000,
            value=100,
            step=10,
            key="sb_grid",
        )
        kerf = st.number_input(
            "Kerf / zaagspleet (mm)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            key="sb_kerf",
        )

        with st.expander("🔧 Geavanceerde optimalisatie", expanded=False):
            runs = st.number_input(
                "Optimalisatie-runs (Max-Rects)",
                min_value=1,
                max_value=200,
                value=30,
                step=1,
                key="sb_runs",
            )
            heuristic_choice = st.selectbox(
                "Heuristiek Max-Rects",
                ["combined", "area", "short", "long", "all"],
                index=0,
                help='"all" test alles. "combined" stuurt op horizontaal vullen eerst.',
                key="sb_heuristic",
            )

        st.markdown("---")
        st.caption("DXF status:")
        if EZDXF_AVAILABLE:
            st.success("`ezdxf` gevonden ✓ — DXF import klaar voor gebruik.")
        else:
            st.error("`ezdxf` ontbreekt. Installeer met: `pip install ezdxf`.")

    return W, H, grid, kerf, int(runs), heuristic_choice


def render_parts_input(default_colors) -> List[Part]:
    st.subheader("🧩 Onderdelen invoer")

    src = st.radio("Kies invoerbron", ["DXF import", "Handmatig"], horizontal=True)

    parts: List[Part] = []

    if src == "DXF import":
        st.markdown(
            "Sleep hier je **DXF-bestanden** in. Gesloten polylines in *model space* "
            "worden als onderdelen ingelezen (bounding box, gegroepeerd per afmeting)."
        )
        uploaded = st.file_uploader(
            "DXF-bestanden", type=["dxf"], accept_multiple_files=True
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            scale_to_mm = st.number_input(
                "Schaalfactor → mm",
                min_value=0.000001,
                max_value=1_000_000.0,
                value=1.0,
                step=0.1,
                help="Voor tekeningen in meters kies 1000; in inches 25.4; etc.",
                key="dx_scale",
            )
        with c2:
            min_area_mm2 = st.number_input(
                "Min. oppervlak (mm²) filter",
                min_value=0.0,
                value=1.0,
                step=1.0,
                help="Filtert ruis/heel kleine contouren weg.",
                key="dx_min_area",
            )
        with c3:
            dxf_thickness = st.number_input(
                "Dikte (mm) voor deze DXF's",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="0 = geen specifieke dikte",
                key="dx_thickness",
            )
        with c4:
            dxf_material = st.text_input(
                "Type isolatie voor deze DXF's", value="", key="dx_material"
            )

        label_prefix = st.text_input("Label prefix", value="DXF", key="dx_prefix")

        if uploaded:
            if not EZDXF_AVAILABLE:
                st.error(
                    "Kan DXF niet lezen: `ezdxf` ontbreekt. Installeer: `pip install ezdxf`."
                )
            else:
                parsed = parse_dxf_files_to_parts(
                    uploaded,
                    scale_to_mm=scale_to_mm,
                    min_area_mm2=min_area_mm2,
                )
                if not parsed:
                    st.warning(
                        "Geen gesloten polylines gevonden (of alles uitgefilterd)."
                    )
                else:
                    rows_for_df = []
                    for i, (w, h, qty) in enumerate(parsed, start=1):
                        color = default_colors[(i - 1) % len(default_colors)]
                        label = f"{label_prefix} {i}"
                        parts.append(
                            Part(
                                label=label,
                                w=int(w),
                                h=int(h),
                                qty=int(qty),
                                color=color,
                                thickness=float(dxf_thickness),
                                material=dxf_material,
                            )
                        )
                        rows_for_df.append(
                            [
                                label,
                                int(w),
                                int(h),
                                int(qty),
                                dxf_thickness,
                                dxf_material,
                                color,
                            ]
                        )

                    st.markdown(
                        "**Geïmporteerde onderdelen (gegroepeerd op afmeting):**"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            rows_for_df,
                            columns=[
                                "Label",
                                "Breedte (mm)",
                                "Hoogte (mm)",
                                "Aantal",
                                "Dikte (mm)",
                                "Type isolatie",
                                "Kleur",
                            ],
                        ),
                        use_container_width=True,
                    )

    else:
        n = st.number_input(
            "Aantal verschillende onderdelen", 1, 50, 2, 1, key="n_parts"
        )
        st.markdown("Geef per onderdeel de afmetingen, dikte, type, aantallen en kleur op.")
        for i in range(n):
            st.markdown(f"#### Onderdeel {i + 1}")
            cc = st.columns(7)
            label = cc[0].text_input(
                "Naam", f"Onderdeel {i+1}", key=f"l{i}"
            )
            w = cc[1].number_input(
                "Breedte (mm)", 1, 20000, 500, 10, key=f"w{i}"
            )
            h = cc[2].number_input(
                "Hoogte (mm)", 1, 20000, 300, 10, key=f"h{i}"
            )
            thickness = cc[3].number_input(
                "Dikte (mm)", 0.0, 1000.0, 0.0, 1.0, key=f"t{i}"
            )
            material = cc[4].text_input(
                "Type isolatie", "", key=f"m{i}"
            )
            qty = cc[5].number_input(
                "Aantal", 1, 9999, 5, 1, key=f"q{i}"
            )
            color = cc[6].color_picker(
                "Kleur", default_colors[i % len(default_colors)], key=f"c{i}"
            )
            parts.append(
                Part(
                    label,
                    int(w),
                    int(h),
                    int(qty),
                    color,
                    float(thickness),
                    material,
                )
            )

        if parts:
            overview_df = pd.DataFrame(
                [
                    [
                        p.label,
                        p.w,
                        p.h,
                        p.thickness,
                        p.material,
                        p.qty,
                        p.color,
                        (p.area() * p.qty) / 1_000_000.0,
                    ]
                    for p in parts
                ],
                columns=[
                    "Label",
                    "Breedte (mm)",
                    "Hoogte (mm)",
                    "Dikte (mm)",
                    "Type isolatie",
                    "Aantal",
                    "Kleur",
                    "Totaal oppervlak (m²)",
                ],
            )
            st.markdown("**Overzicht ingevoerde onderdelen:**")
            st.dataframe(overview_df, use_container_width=True)

    return parts


# ========= Hoofd UI =========
st.title("🔪 Plaatoptimalisatie Tool – met DXF import & statistieken")

default_colors = [
    "#A3CEF1",
    "#90D26D",
    "#F29E4C",
    "#E59560",
    "#B56576",
    "#6D597A",
    "#355070",
    "#43AA8B",
    "#FFB5A7",
    "#BDE0FE",
    "#84A98C",
    "#F6BD60",
    "#6C757D",
    "#B08968",
    "#A2D2FF",
]

W, H, grid, kerf, runs, heuristic_choice = render_sidebar()

tab_input, tab_result = st.tabs(["📥 Invoer", "📐 Resultaat"])

with tab_input:
    parts = render_parts_input(default_colors)

with tab_result:
    st.subheader("📐 Optimalisatie uitvoeren")

    if not parts:
        st.info(
            "Er zijn nog geen onderdelen ingevoerd. Ga naar het tabblad **Invoer** om te beginnen."
        )
    else:
        oversized = find_oversized_parts(parts, W, H, kerf)
        if oversized:
            st.error(
                "De volgende onderdelen passen nooit op de gekozen plaat "
                "(ook niet met rotatie en huidige kerf):"
            )
            df_oversized = pd.DataFrame(
                [
                    [p.label, p.w, p.h, p.thickness, p.material, p.qty]
                    for p in oversized
                ],
                columns=[
                    "Label",
                    "Breedte (mm)",
                    "Hoogte (mm)",
                    "Dikte (mm)",
                    "Type isolatie",
                    "Aantal",
                ],
            )
            st.dataframe(df_oversized, use_container_width=True)

        run = st.button("🚀 Optimaliseer", type="primary")

        if run:
            with st.spinner("Bezig met optimaliseren..."):
                parts_by_group: Dict[Tuple[float, str], List[Part]] = defaultdict(list)
                for p in parts:
                    key = (float(p.thickness), str(p.material))
                    parts_by_group[key].append(p.copy())

                plates_by_group: Dict[Tuple[float, str], List[List[Dict]]] = {}
                all_plates_for_stats: List[List[Dict]] = []

                for key, plist in parts_by_group.items():
                    res = pack_all(
                        W,
                        H,
                        [pp.copy() for pp in plist],
                        kerf,
                        heuristic=heuristic_choice,
                        runs=int(runs),
                    )
                    if res:
                        plates_by_group[key] = res
                        all_plates_for_stats.extend(res)

            if not all_plates_for_stats:
                st.warning("Er is geen plaatsing gevonden met de huidige instellingen.")
            else:
                stats = compute_global_stats(all_plates_for_stats, W, H, parts)
                st.markdown("### 📊 Globale statistieken")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Totaal aantal platen", f"{stats['total_plates']}")
                col_s2.metric(
                    "Benutting plaatoppervlak",
                    f"{stats['utilisation']:.1f} %",
                )
                col_s3.metric(
                    "Afval (gemiddeld)",
                    f"{stats['waste_pct']:.1f} %",
                )
                col_s4.metric(
                    "Totaal onderdelen-oppervlak",
                    f"{stats['parts_total_area_m2']:.2f} m²",
                )

                grouped = OrderedDict()
                for key, plates_group in plates_by_group.items():
                    for placed in plates_group:
                        used_area = sum(r["w"] * r["h"] for r in placed)
                        rest_pct = (
                            100 - (used_area / (W * H) * 100)
                            if (W > 0 and H > 0)
                            else 0.0
                        )
                        sig = plate_signature(placed, group_key=key)
                        thickness, material = key
                        if sig in grouped:
                            grouped[sig]["count"] += 1
                        else:
                            grouped[sig] = dict(
                                placed=placed,
                                count=1,
                                rest_pct=rest_pct,
                                group_key=key,
                                thickness=thickness,
                                material=material,
                            )

                st.markdown("### 🧱 Overzicht per materiaal en dikte")

                total_plates_by_group: Dict[Tuple[float, str], int] = defaultdict(int)
                for item in grouped.values():
                    key = item["group_key"]
                    total_plates_by_group[key] += item["count"]

                ui_parts_by_group: Dict[Tuple[float, str], List[Part]] = defaultdict(list)
                for p in parts:
                    key = (float(p.thickness), str(p.material))
                    ui_parts_by_group[key].append(p)

                for key in sorted(
                    ui_parts_by_group.keys(), key=lambda k: (str(k[1]), float(k[0]))
                ):
                    th, mat = key
                    group_parts = ui_parts_by_group[key]
                    total_pl = total_plates_by_group.get(key, 0)

                    st.markdown(f"#### Materiaal: {mat or '-'}")
                    st.markdown(f"- **Totaal verbruikte platen:** {total_pl}")
                    st.markdown(f"- **Dikte:** {th} mm")

                    df_group = pd.DataFrame(
                        [
                            [
                                p.label,
                                p.w,
                                p.h,
                                p.qty,
                                (p.area() * p.qty) / 1_000_000.0,
                            ]
                            for p in group_parts
                        ],
                        columns=[
                            "Label",
                            "Breedte (mm)",
                            "Hoogte (mm)",
                            "Aantal",
                            "Totaal opp. (m²)",
                        ],
                    )
                    st.dataframe(df_group, use_container_width=True)
                    st.markdown("---")

                total_plates = sum(item["count"] for item in grouped.values())
                st.info(f"**Totaal aantal platen (incl. herhalingen):** {total_plates}")

                plate_pages: List[Dict] = []

                for idx, item in enumerate(grouped.values(), start=1):
                    placed = item["placed"]
                    count = item["count"]
                    rest_pct = item["rest_pct"]
                    thickness = item.get("thickness", 0.0)
                    material = item.get("material", "")

                    rows: Dict[str, int] = {}
                    for r in placed:
                        key_lbl = f"{r['label']} ({r['w']}x{r['h']})"
                        rows[key_lbl] = rows.get(key_lbl, 0) + 1
                    summary_df = (
                        pd.DataFrame(
                            [[k, v] for k, v in rows.items()],
                            columns=["Onderdeel (afm)", "Aantal per plaat"],
                        )
                        .sort_values("Onderdeel (afm)")
                        .reset_index(drop=True)
                    )

                    buf, legend = draw_plate_png(
                        placed, idx, W, H, grid, rest_pct, count=count
                    )
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".png"
                    ) as tmp:
                        tmp.write(buf.getvalue())
                        png_path = tmp.name

                    plate_pages.append(
                        dict(
                            plate_no=idx,
                            count=count,
                            rows=rows,
                            rest_pct=rest_pct,
                            legend=legend,
                            png_path=png_path,
                            thickness=thickness,
                            material=material,
                        )
                    )

                    colA, colB = st.columns([3, 2])
                    with colA:
                        extra = []
                        if thickness and thickness != 0.0:
                            extra.append(f"{thickness} mm")
                        if material:
                            extra.append(material)
                        extra_txt = f" ({', '.join(extra)})" if extra else ""
                        st.image(
                            buf,
                            caption=f"Plaat-type {idx} x{count}{extra_txt}",
                            use_container_width=True,
                        )
                    with colB:
                        st.markdown("**Onderdelen overzicht (per plaat)**")
                        st.dataframe(summary_df, use_container_width=True)
                        st.markdown(
                            f"**Restmateriaal (per plaat):** {rest_pct:.1f}%"
                        )
                        extra_lines = []
                        if thickness and thickness != 0.0:
                            extra_lines.append(f"Dikte: {thickness} mm")
                        if material:
                            extra_lines.append(f"Type: {material}")
                        if extra_lines:
                            st.markdown("**Materiaal**")
                            for line in extra_lines:
                                st.markdown(f"- {line}")
                        st.markdown(
                            f"**Aantal uit te voeren platen van dit type:** {count}"
                        )
                        st.markdown("**Legenda**")
                        for lbl, clr in legend.items():
                            st.markdown(
                                f"<div style='display:flex;align-items:center;gap:6px;'>"
                                f"<div style='width:14px;height:14px;background:{clr};border:1px solid #000;'></div>"
                                f"<span>{lbl}</span></div>",
                                unsafe_allow_html=True,
                            )
                    st.divider()

                try:
                    pdf_bytes = build_pdf_from_pages(plate_pages, parts)
                    csv_bytes = build_csv_from_plates(grouped, W, H)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "📄 Download PDF (overzicht per materiaal + per plaat-type)",
                            data=pdf_bytes,
                            file_name="plaatindeling.pdf",
                            mime="application/pdf",
                        )
                    with col_dl2:
                        st.download_button(
                            "📥 Download CSV met alle posities",
                            data=csv_bytes,
                            file_name="plaatindeling_posities.csv",
                            mime="text/csv",
                        )
                finally:
                    for page in plate_pages:
                        try:
                            os.remove(page["png_path"])
                        except Exception:
                            pass
