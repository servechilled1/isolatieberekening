# app.py
import os
import io
import random
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
from collections import OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pandas as pd
from fpdf import FPDF
import streamlit as st

# --- Streamlit config (eerste call) ---
st.set_page_config(layout="wide", page_title="Plaatoptimalisatie Tool")

# ---------- Datamodellen ----------
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
            self.label, int(self.w), int(self.h), int(self.qty),
            self.color, float(self.thickness), str(self.material)
        )

@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h

# ---------- Packing helpers ----------
def _apply_kerf(w: int, h: int, kerf: int) -> Tuple[int, int]:
    return w + kerf, h + kerf

def _fits(fr: Rect, w: int, h: int) -> bool:
    return w <= fr.w and h <= fr.h

def _split_free_rect(fr: Rect, used: Dict[str, int]) -> Tuple[List[Rect], bool]:
    new_rects: List[Rect] = []
    if (
        used["x"] >= fr.x + fr.w or used["x"] + used["w"] <= fr.x or
        used["y"] >= fr.y + fr.h or used["y"] + used["h"] <= fr.y
    ):
        return [fr], False
    if used["y"] > fr.y:
        new_rects.append(Rect(fr.x, fr.y, fr.w, used["y"] - fr.y))
    bottom = used["y"] + used["h"]
    if bottom < fr.y + fr.h:
        new_rects.append(Rect(fr.x, bottom, fr.w, (fr.y + fr.h) - bottom))
    if used["x"] > fr.x:
        new_rects.append(Rect(
            fr.x, max(fr.y, used["y"]),
            used["x"] - fr.x,
            min(fr.y + fr.h, used["y"] + used["h"]) - max(fr.y, used["y"]),
        ))
    right_x = used["x"] + used["w"]
    if right_x < fr.x + fr.w:
        new_rects.append(Rect(
            right_x, max(fr.y, used["y"]),
            (fr.x + fr.w) - right_x,
            min(fr.y + fr.h, used["y"] + used["h"]) - max(fr.y, used["y"]),
        ))
    return new_rects, True

def _prune(rects: List[Rect]) -> List[Rect]:
    pruned: List[Rect] = []
    for i, r in enumerate(rects):
        contained = False
        for j, r2 in enumerate(rects):
            if (
                i != j and r.x >= r2.x and r.y >= r2.y and
                r.x + r.w <= r2.x + r2.w and r.y + r.h <= r2.y + r2.h
            ):
                contained = True
                break
        if not contained and r.w > 0 and r.h > 0:
            pruned.append(r)
    return pruned

def _choose_best_fit(free_rects: List[Rect], w: int, h: int, kerf: int, heuristic: str):
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
                    score = (min(horiz_waste, vert_waste), area_waste, max(horiz_waste, vert_waste))
                elif heuristic == "long":
                    score = (max(horiz_waste, vert_waste), area_waste, min(horiz_waste, vert_waste))
                else:
                    score = (-pieces_in_row, horiz_waste, vert_waste, area_waste)
                if best is None or score < best["score"]:
                    best = dict(idx=idx, w=cw, h=ch, w_eff=w_eff, h_eff=h_eff, score=score)
    return best

def pack_plate_maxrects(W: int, H: int, parts_left: List[Part], kerf: int, heuristic: str = "area"):
    free_rects: List[Rect] = [Rect(0, 0, W, H)]
    placed = []
    parts_left.sort(key=lambda p: p.area(), reverse=True)
    for p in parts_left:
        while p.qty > 0:
            best = None
            for (w, h) in (((p.w, p.h), (p.h, p.w)) if p.w != p.h else ((p.w, p.h),)):
                cand = _choose_best_fit(free_rects, w, h, kerf, heuristic)
                if cand and (best is None or cand["score"] < best["score"]):
                    best = cand
            if best is None:
                break
            fr = free_rects[best["idx"]]
            used = dict(x=fr.x, y=fr.y, w=best["w"], h=best["h"])
            placed.append(dict(**used, label=p.label, color=p.color, thickness=p.thickness, material=p.material))
            used_exp = dict(x=used["x"], y=used["y"], w=best["w_eff"], h=best["h_eff"])
            new_free: List[Rect] = []
            for fr2 in free_rects:
                split_rects, did = _split_free_rect(fr2, used_exp)
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
                    candidates.append((-pieces_in_row, horiz_waste, cw, ch, w_eff, h_eff))
            if candidates:
                candidates.sort()
                _, _, cw, ch, w_eff, h_eff = candidates[0]
                placed.append(dict(x=x, y=y, w=cw, h=ch, label=p.label, color=p.color, thickness=p.thickness, material=p.material))
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

def pack_all(W: int, H: int, parts: List[Part], kerf: int, heuristic: str = "combined", runs: int = 20):
    shelf_parts = [p.copy() for p in parts]
    plates_shelf = []
    while any(p.qty > 0 for p in shelf_parts):
        pl, shelf_parts = pack_plate_shelf(W, H, shelf_parts, kerf)
        if not pl:
            break
        plates_shelf.append(pl)
    rest_shelf = sum((W * H - sum(r["w"] * r["h"] for r in pl)) for pl in plates_shelf) if plates_shelf else W * H
    score_shelf = (len(plates_shelf) if plates_shelf else float("inf"), rest_shelf)

    best_mr = None
    heur_to_try = [heuristic] if heuristic != "all" else ["combined", "area", "short", "long"]
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
            total_rest = sum((W * H - sum(r["w"] * r["h"] for r in pl)) for pl in plates_h)
            score = (len(plates_h), total_rest)
            if best_mr is None or score < best_mr["score"]:
                best_mr = dict(plates=plates_h, score=score)
    if best_mr and best_mr["score"] < score_shelf:
        return best_mr["plates"]
    return plates_shelf

# ---------- Scoring & utilities ----------
def plate_signature(placed: Iterable[Dict], group_key=None) -> Tuple:
    base = tuple(sorted((r["x"], r["y"], r["w"], r["h"], r["label"]) for r in placed))
    return (group_key, base)

def compute_global_stats(plates: List[List[Dict]], W: int, H: int, parts: List[Part]) -> Dict[str, float]:
    total_plates = len(plates)
    total_plate_area = W * H * total_plates
    used_area = sum(r["w"] * r["h"] for pl in plates for r in pl)
    utilisation = (used_area / total_plate_area * 100) if total_plate_area > 0 else 0.0
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

def find_oversized_parts(parts: List[Part], W: int, H: int, kerf: int) -> List[Part]:
    too_big: List[Part] = []
    for p in parts:
        w1, h1 = _apply_kerf(p.w, p.h, kerf)
        w2, h2 = _apply_kerf(p.h, p.w, kerf)
        if not ((w1 <= W and h1 <= H) or (w2 <= W and h2 <= H)):
            too_big.append(p)
    return too_big

# ---------- Rendering / Export ----------
def draw_plate_png(placed, plate_no: int, W: int, H: int, grid_step: int, rest_pct: float, count: int = 1):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    if grid_step > 0:
        for gx in range(0, W + 1, grid_step): ax.axvline(gx, color="lightgray", lw=0.4)
        for gy in range(0, H + 1, grid_step): ax.axhline(gy, color="lightgray", lw=0.4)
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="black", lw=1.5))
    legend: Dict[str, str] = {}
    for r in placed:
        rect = Rectangle((r["x"], r["y"]), r["w"], r["h"], fc=r["color"], ec="black", lw=0.8)
        ax.add_patch(rect)
        ax.text(r["x"] + r["w"] / 2, r["y"] + r["h"] / 2, f"{r['w']}x{r['h']}", ha="center", va="center", fontsize=8)
        legend[r["label"]] = r["color"]
    title_suffix = f" x{count}" if count > 1 else ""
    ax.set_title(f"Plaat {plate_no}{title_suffix} - {W}x{H} mm | Rest: {rest_pct:.1f}%", fontsize=13, weight="bold", pad=8)
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png", dpi=200); plt.close(fig); buf.seek(0)
    return buf, legend

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))

def pdf_safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")

def build_pdf_from_pages(plate_pages: List[Dict], parts: List[Part]) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    margin = 10; page_w = 210; usable_w = page_w - 2 * margin

    total_plates_by_group: Dict[Tuple[float, str], int] = defaultdict(int)
    for page in plate_pages:
        th = page.get("thickness", 0.0); mat = page.get("material", "")
        total_plates_by_group[(th, mat)] += int(page.get("count", 0))

    parts_by_group: Dict[Tuple[float, str], List[Part]] = defaultdict(list)
    for p in parts:
        parts_by_group[(float(p.thickness), str(p.material))].append(p)

    pdf.add_page()
    pdf.set_xy(margin, margin); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(usable_w, 8, pdf_safe("Plaatoptimalisatie – Overzicht per materiaal"), ln=1, align="C")
    pdf.ln(3)

    col_label_w = usable_w * 0.30
    col_w_w     = usable_w * 0.15
    col_h_w     = usable_w * 0.15
    col_qty_w   = usable_w * 0.15
    col_area_w  = usable_w * 0.25

    groups_sorted = sorted(parts_by_group.keys(), key=lambda k: (str(k[1]), float(k[0])))
    for (th, mat) in groups_sorted:
        group_parts = parts_by_group[(th, mat)]
        total_pl = total_plates_by_group.get((th, mat), 0)

        pdf.set_font("Helvetica", "B", 11); pdf.ln(3)
        pdf.cell(usable_w, 6, pdf_safe(f"Materiaal: {mat or '-'}"), ln=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(usable_w, 5, pdf_safe(f"Totaal verbruikte platen: {total_pl}"), ln=1)
        pdf.cell(usable_w, 5, pdf_safe(f"Dikte: {th} mm"), ln=1)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9); pdf.set_x(margin)
        pdf.cell(col_label_w, 6, pdf_safe("Label"),            border=1)
        pdf.cell(col_w_w,     6, pdf_safe("Breedte (mm)"),     border=1)
        pdf.cell(col_h_w,     6, pdf_safe("Hoogte (mm)"),      border=1)
        pdf.cell(col_qty_w,   6, pdf_safe("Aantal"),           border=1)
        pdf.cell(col_area_w,  6, pdf_safe("Totaal opp. (m2)"), border=1, ln=1)

        pdf.set_font("Helvetica", "", 9)
        for p in sorted(group_parts, key=lambda x: x.label):
            total_area_m2 = (p.area() * p.qty) / 1_000_000.0
            pdf.set_x(margin)
            pdf.cell(col_label_w, 5, pdf_safe(str(p.label)), border=1)
            pdf.cell(col_w_w,     5, pdf_safe(str(p.w)),     border=1)
            pdf.cell(col_h_w,     5, pdf_safe(str(p.h)),     border=1)
            pdf.cell(col_qty_w,   5, pdf_safe(str(p.qty)),   border=1)
            pdf.cell(col_area_w,  5, pdf_safe(f"{total_area_m2:.2f}"), border=1, ln=1)

    img_h = 110
    for page in plate_pages:
        pdf.add_page()
        title = f"Plaat-type {page['plate_no']} x{page['count']}"
        th = page.get("thickness", 0.0); mat = page.get("material", "")
        extra = []
        if th and th != 0.0: extra.append(f"{th}mm")
        if mat: extra.append(mat)
        if extra: title += " - " + " ".join(extra)
        pdf.set_xy(margin, margin); pdf.set_font("Helvetica", "B", 12)
        pdf.cell(usable_w, 8, pdf_safe(title), ln=1, align="C")

        img_y = pdf.get_y() + 2
        pdf.image(page["png_path"], x=margin, y=img_y, w=usable_w, h=img_h)

        y = img_y + img_h + 5
        pdf.set_xy(margin, y); pdf.set_font("Helvetica", "B", 10)
        pdf.cell(usable_w, 6, pdf_safe("Onderdelen overzicht (per plaat)"), ln=1)

        rows = page["rows"]; sorted_rows = sorted(rows.items(), key=lambda kv: kv[0])
        col1_w = usable_w * 0.7; col2_w = usable_w * 0.3

        pdf.set_font("Helvetica", "B", 9); pdf.set_xy(margin, pdf.get_y() + 1)
        pdf.cell(col1_w, 5, pdf_safe("Onderdeel (afm)"), border=1)
        pdf.cell(col2_w, 5, pdf_safe("Aantal"),          border=1, ln=1)

        pdf.set_font("Helvetica", "", 9)
        for label, qty in sorted_rows:
            pdf.set_x(margin)
            pdf.cell(col1_w, 5, pdf_safe(label),    border=1)
            pdf.cell(col2_w, 5, pdf_safe(str(qty)), border=1, ln=1)

        pdf.ln(3); pdf.set_font("Helvetica", "B", 9)
        pdf.cell(usable_w, 5, pdf_safe(f"Restmateriaal (per plaat): {page['rest_pct']:.1f}%"), ln=1)
        pdf.cell(usable_w, 5, pdf_safe(f"Aantal uit te voeren platen van dit type: {page['count']}"), ln=1)

        pdf.ln(3); pdf.set_font("Helvetica", "B", 10); pdf.cell(usable_w, 5, pdf_safe("Legenda"), ln=1)
        pdf.set_font("Helvetica", "", 9)
        for lbl, clr in page["legend"].items():
            r, g, b = hex_to_rgb(clr); y = pdf.get_y()
            pdf.set_xy(margin, y); pdf.set_fill_color(r, g, b); pdf.rect(margin, y, 4, 4, style="F")
            pdf.set_xy(margin + 6, y); pdf.cell(usable_w - 6, 4, pdf_safe(lbl), ln=1)

    out = io.BytesIO(); pdf.output(out); out.seek(0)
    return out.read()

def build_csv_from_plates(grouped, W: int, H: int) -> bytes:
    rows = []
    for plate_no, item in enumerate(grouped.values(), start=1):
        placed = item["placed"]; copies = item["count"]
        thickness = item.get("thickness", 0.0); material = item.get("material", "")
        for r in placed:
            rows.append(dict(
                plaat_type=plate_no, copies=copies, label=r["label"],
                x=r["x"], y=r["y"], w=r["w"], h=r["h"],
                dikte_mm=thickness, type_isolatie=material, plaat_breedte=W, plaat_hoogte=H
            ))
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

# ---------- Consolidatie ----------
def consolidate_plates(plates: List[List[Dict]], W: int, H: int, kerf: int, base_heuristic: str = "combined", extra_runs: int = 300) -> List[List[Dict]]:
    counter = defaultdict(int); meta = {}
    for pl in plates:
        for r in pl:
            key = (r["label"], r["w"], r["h"])
            counter[key] += 1
            meta[key] = dict(color=r["color"], thickness=r.get("thickness", 0.0), material=r.get("material", ""))
    repack_parts: List[Part] = []
    for (label, w, h), qty in counter.items():
        info = meta[(label, w, h)]
        repack_parts.append(Part(label=label, w=w, h=h, qty=qty, color=info["color"], thickness=float(info["thickness"]), material=str(info["material"])))
    best = pack_all(W, H, repack_parts, kerf, heuristic="all", runs=max(extra_runs, 200))
    return best if best else plates

# ---------- UI helpers ----------
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Instellingen")
        W = st.number_input("Plaatbreedte (mm)", 100, 20000, 3000, 10, key="sb_plaatbreedte")
        H = st.number_input("Plaathoogte (mm)", 100, 20000, 1500, 10, key="sb_plaathoogte")
        st.markdown("**Raster & kerf**")
        grid = st.number_input("Grid (mm)", 0, 1000, 100, 10, key="sb_grid")
        kerf = st.number_input("Kerf / zaagspleet (mm)", 0, 50, 0, 1, key="sb_kerf")
        with st.expander("🔧 Geavanceerde optimalisatie", expanded=False):
            runs = st.number_input("Optimalisatie-runs (Max-Rects)", 1, 200, 60, 1, key="sb_runs")
            heuristic_choice = st.selectbox(
                "Heuristiek Max-Rects", ["combined", "area", "short", "long", "all"],
                index=4, help="Gebruik 'all' voor beste resultaten.", key="sb_heuristic"
            )
        optimize_by_thickness_only = st.checkbox(
            "Optimaliseer per dikte (alles samen)", value=True,
            help="Alle onderdelen met dezelfde dikte in één pool optimaliseren.", key="sb_by_thickness"
        )
        post_consolidate = st.checkbox(
            "Extra consolidatie-pass (langzamer, vaak minder platen)", value=True, key="sb_consolidate"
        )
        extra_runs = st.number_input(
            "Extra runs bij consolidatie", 0, 2000, 400, 50, key="sb_extra_runs"
        )
    return W, H, grid, kerf, int(runs), heuristic_choice, optimize_by_thickness_only, post_consolidate, int(extra_runs)

def render_parts_input(default_colors) -> List[Part]:
    st.subheader("🧩 Onderdelen invoer")
    parts: List[Part] = []
    n = st.number_input("Aantal verschillende onderdelen", 1, 50, 2, 1, key="n_parts")
    for i in range(n):
        st.markdown(f"#### Onderdeel {i + 1}")
        cc = st.columns(7)
        label = cc[0].text_input("Naam", f"Onderdeel {i+1}", key=f"l{i}")
        w = cc[1].number_input("Breedte (mm)", 1, 20000, 500, 10, key=f"w{i}")
        h = cc[2].number_input("Hoogte (mm)", 1, 20000, 300, 10, key=f"h{i}")
        thickness = cc[3].number_input("Dikte (mm)", 0.0, 1000.0, 1.0, 0.5, key=f"t{i}")
        material = cc[4].text_input("Type isolatie", "", key=f"m{i}")
        qty = cc[5].number_input("Aantal", 1, 9999, 5, 1, key=f"q{i}")
        color = cc[6].color_picker("Kleur", default_colors[i % len(default_colors)], key=f"c{i}")
        parts.append(Part(label, int(w), int(h), int(qty), color, float(thickness), material))

    if parts:
        overview_df = pd.DataFrame(
            [[p.label, p.w, p.h, p.thickness, p.material, p.qty, (p.area() * p.qty) / 1_000_000.0] for p in parts],
            columns=["Label", "Breedte (mm)", "Hoogte (mm)", "Dikte (mm)", "Type isolatie", "Aantal", "Totaal opp. (m²)"],
        )
        st.dataframe(overview_df, use_container_width=True)

    return parts

# ---------- Hoofd UI ----------
st.title("🔪 Plaatoptimalisatie Tool")

default_colors = [
    "#A3CEF1","#90D26D","#F29E4C","#E59560","#B56576",
    "#6D597A","#355070","#43AA8B","#FFB5A7","#BDE0FE",
    "#84A98C","#F6BD60","#6C757D","#B08968","#A2D2FF",
]

W, H, grid, kerf, runs, heuristic_choice, optimize_by_thickness_only, post_consolidate, extra_runs = render_sidebar()
tab_input, tab_result = st.tabs(["📥 Invoer", "📐 Resultaat"])

with tab_input:
    parts = render_parts_input(default_colors)

with tab_result:
    st.subheader("📐 Optimalisatie uitvoeren")
    if not parts:
        st.info("Er zijn nog geen onderdelen ingevoerd. Ga naar het tabblad Invoer om te beginnen.")
    else:
        oversized = find_oversized_parts(parts, W, H, kerf)
        if oversized:
            st.error("De volgende onderdelen passen nooit op de gekozen plaat (ook niet met rotatie):")
            df_oversized = pd.DataFrame(
                [[p.label, p.w, p.h, p.thickness, p.material, p.qty] for p in oversized],
                columns=["Label", "Breedte (mm)", "Hoogte (mm)", "Dikte (mm)", "Type isolatie", "Aantal"],
            )
            st.dataframe(df_oversized, use_container_width=True)

        run = st.button("🚀 Optimaliseer", type="primary")

        if run:
            with st.spinner("Bezig met optimaliseren..."):
                # Poolen: standaard per DIKTE alles samen
                if optimize_by_thickness_only:
                    parts_by_group: Dict[Tuple[float], List[Part]] = defaultdict(list)
                    for p in parts:
                        parts_by_group[(float(p.thickness),)].append(p.copy())
                else:
                    parts_by_group: Dict[Tuple[float, str], List[Part]] = defaultdict(list)
                    for p in parts:
                        parts_by_group[(float(p.thickness), str(p.material))].append(p.copy())

                plates_by_group: Dict[Tuple, List[List[Dict]]] = {}
                all_plates_for_stats: List[List[Dict]] = []

                for key, plist in parts_by_group.items():
                    res = pack_all(W, H, [pp.copy() for pp in plist], kerf, heuristic=heuristic_choice, runs=int(runs))
                    if not res:
                        continue
                    if post_consolidate:
                        res2 = consolidate_plates(res, W, H, kerf, base_heuristic=heuristic_choice, extra_runs=int(extra_runs))
                        def score(plts):
                            total_rest = sum((W * H - sum(r["w"] * r["h"] for r in pl)) for pl in plts)
                            return (len(plts), total_rest)
                        if score(res2) < score(res):
                            res = res2
                    plates_by_group[key] = res
                    all_plates_for_stats.extend(res)

            if not all_plates_for_stats:
                st.warning("Er is geen plaatsing gevonden met de huidige instellingen.")
            else:
                stats = compute_global_stats(all_plates_for_stats, W, H, parts)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Totaal aantal platen", f"{stats['total_plates']}")
                c2.metric("Benutting", f"{stats['utilisation']:.1f} %")
                c3.metric("Afval (gem.)", f"{stats['waste_pct']:.1f} %")
                c4.metric("Totaal onderdelen-oppervlak", f"{stats['parts_total_area_m2']:.2f} m²")

                grouped = OrderedDict()
                for key, plates_group in plates_by_group.items():
                    for placed in plates_group:
                        used_area = sum(r["w"] * r["h"] for r in placed)
                        rest_pct = 100 - (used_area / (W * H) * 100) if (W > 0 and H > 0) else 0.0
                        sig = plate_signature(placed, group_key=key)
                        # key kan (thickness,) of (thickness, material)
                        if len(key) == 1:
                            thickness, material = key[0], ""
                        else:
                            thickness, material = key
                        if sig in grouped:
                            grouped[sig]["count"] += 1
                        else:
                            grouped[sig] = dict(
                                placed=placed, count=1, rest_pct=rest_pct,
                                group_key=key, thickness=thickness, material=material,
                            )

                total_plates = sum(item["count"] for item in grouped.values())
                st.info(f"**Totaal aantal platen (incl. herhalingen):** {total_plates}")

                plate_pages: List[Dict] = []
                for idx, item in enumerate(grouped.values(), start=1):
                    placed = item["placed"]; count = item["count"]; rest_pct = item["rest_pct"]
                    thickness = item.get("thickness", 0.0); material = item.get("material", "")
                    rows: Dict[str, int] = {}
                    for r in placed:
                        rows[f"{r['label']} ({r['w']}x{r['h']})"] = rows.get(f"{r['label']} ({r['w']}x{r['h']})", 0) + 1
                    buf, legend = draw_plate_png(placed, idx, W, H, grid, rest_pct, count=count)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(buf.getvalue()); png_path = tmp.name

                    plate_pages.append(dict(
                        plate_no=idx, count=count, rows=rows, rest_pct=rest_pct, legend=legend,
                        png_path=png_path, thickness=thickness, material=material
                    ))

                    colA, colB = st.columns([3, 2])
                    with colA:
                        extra = []
                        if thickness and thickness != 0.0: extra.append(f"{thickness} mm")
                        if material: extra.append(material)
                        extra_txt = f" ({', '.join(extra)})" if extra else ""
                        st.image(buf, caption=f"Plaat-type {idx} x{count}{extra_txt}", use_container_width=True)
                    with colB:
                        summary_df = pd.DataFrame([[k, v] for k, v in rows.items()], columns=["Onderdeel (afm)", "Aantal per plaat"]).sort_values("Onderdeel (afm)").reset_index(drop=True)
                        st.dataframe(summary_df, use_container_width=True)
                        st.markdown(f"**Restmateriaal (per plaat):** {rest_pct:.1f}%")
                        if extra:
                            st.markdown("**Materiaal**")
                            for line in extra: st.markdown(f"- {line}")
                        st.markdown(f"**Aantal uit te voeren platen van dit type:** {count}")
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
                    cdl1, cdl2 = st.columns(2)
                    with cdl1:
                        st.download_button(
                            "📄 Download PDF (overzicht + plaat-types)",
                            data=pdf_bytes, file_name="plaatindeling.pdf", mime="application/pdf"
                        )
                    with cdl2:
                        st.download_button(
                            "📥 Download CSV met alle posities",
                            data=csv_bytes, file_name="plaatindeling_posities.csv", mime="text/csv"
                        )
                finally:
                    for page in plate_pages:
                        try: os.remove(page["png_path"])
                        except Exception: pass



