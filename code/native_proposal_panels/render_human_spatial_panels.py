"""Build native PowerPoint human, spatial, COMPASS and ATAC panels from the source figure and the four analysis releases."""

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import math
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import colormaps, colors
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
import argparse

apar = argparse.ArgumentParser()
apar.add_argument("--figures", required=True, type=Path)
apar.add_argument("--snrna", required=True, type=Path)
apar.add_argument("--atac", required=True, type=Path)
apar.add_argument("--spatial", required=True, type=Path)
apar.add_argument("--compass", required=True, type=Path)
apar.add_argument("--out", required=True, type=Path)
args = apar.parse_args()
pres = Presentation()
pres.slide_width = Pt(540)
pres.slide_height = Pt(780)
ims = {}
with ZipFile(args.figures) as z:
    for n in [
        121,
        122,
        123,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        200,
        201,
        202,
        203,
    ]:
        ims[n] = Image.open(BytesIO(z.read(f"ppt/media/image{n}.png"))).convert("RGB")
from functools import lru_cache


@lru_cache(maxsize=None)
def thumb(n):
    im = ims[n].copy()
    im.thumbnail((670, 410))
    return im


def rich(x, y, w, h, msg, fs=7, b=False, al="left", c="191919", italic=False, rot=None):
    sh = s.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    sh.name = str(msg)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = {
        "center": PP_ALIGN.CENTER,
        "right": PP_ALIGN.RIGHT,
        "left": PP_ALIGN.LEFT,
    }[al]
    run = p.add_run()
    run.text = str(msg)
    run.font.name = "Arial"
    run.font.size = Pt(fs)
    run.font.bold = b
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(c)
    if rot is not None:
        sh.rotation = rot
    return sh


def vlabel(x, top, t, fs=7.2, h=32):
    # 270 degree font: centring the transformed textbox aligns each label with its tick.
    rich(x - h / 2, top + h / 2 - 4, h, 8, t, fs, al="right", italic=True, rot=270)


def rect(x, y, w, h, col, name, edge="", lw=0.25, circle=False):
    sh = s.shapes.add_shape(
        MSO_SHAPE.OVAL if circle else MSO_SHAPE.RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h)
    )
    sh.name = name
    if col:
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string(col)
    else:
        sh.fill.background()
    if edge:
        sh.line.color.rgb = RGBColor.from_string(edge)
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    sh._element.spPr.append(OxmlElement("a:effectLst"))
    return sh


def dot(x, y, diam, col, name, edge="", lw=0.2):
    return rect(x - diam / 2, y - diam / 2, diam, diam, col, name, edge, lw, True)


def line(x, y, x2, y2, col="666666", lw=0.4):
    sh = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x), Pt(y), Pt(x2), Pt(y2))
    sh.line.color.rgb = RGBColor.from_string(col)
    sh.line.width = Pt(lw)
    sh._element.spPr.append(OxmlElement("a:effectLst"))
    return sh


def img(x, y, w, h, image, name):
    data = BytesIO()
    image.save(data, "PNG")
    data.seek(0)
    sh = s.shapes.add_picture(data, Pt(x), Pt(y), Pt(w), Pt(h))
    sh.name = name
    return sh


def cut(n, c):
    im = ims[n]
    little = thumb(n)
    return im.crop(
        tuple(
            round(
                v
                * (im.width / little.width if j % 2 == 0 else im.height / little.height)
            )
            for j, v in enumerate(c)
        )
    )


def place(n, c, x, y, w, h, name, align="center"):
    im = cut(n, c) if c else ims[n]
    scale = min(w / im.width, h / im.height)
    ww, hh = im.width * scale, im.height * scale
    return img(
        x + (w - ww) / 2, y + (0 if align == "top" else (h - hh) / 2), ww, hh, im, name
    )


def letter(x, y, t):
    rich(x, y, 12, 13, t, 11.4, True)


def rgb(x):
    return colors.to_hex(x)[1:].upper()


def ramp(x, y, w, lo, hi, cmap="RdBu_r"):
    for j in range(32):
        rect(
            x + w * j / 32,
            y,
            w / 32 + 0.03,
            4,
            rgb(colormaps[cmap](j / 31)),
            f"Colour scale {j+1}",
        )
    for dx, val in [(0, lo), (0.5, "0"), (1, hi)]:
        rich(x + w * dx - 9, y + 4.4, 18, 8, val, 6.9, al="center")


def key(x, y, c, msg, sz=7):
    dot(x + 2.5, y + 4, 3.5, c, msg)
    rich(x + 7, y - 0.5, 110, 9, msg, sz)


def tab(base, file):
    return pd.read_csv(base / "tables" / file, sep="\t")


def colorat(n, x, y):
    return bytes(thumb(n).getpixel((int(x), int(y)))).hex().upper()


s = pres.slides.add_slide(pres.slide_layouts[6])
# 5A: keep both original unresolved classes and point colours.
letter(5, 3, "A")
rich(18, 3, 140, 12, "Cell-type annotations", 8.5, True)
place(121, (18, 20, 361, 392), 13, 18, 85, 89, "Nuclear UMAP, 11/2 source")
rich(25, 107, 63, 9, "UMAP 1", 7.3, al="center")
rich(-1, 56, 27, 8, "UMAP 2", 7.3, al="center", rot=270)
cats = [
    "Adipocyte",
    "EC",
    "FAP",
    "Lymphocyte",
    "MuSC",
    "Myeloid cell",
    "SMC",
    "Specialized MF",
    "Type I",
    "Type II",
    "Unresolved myonuclei",
    "Unresolved nuclei",
]
for i, t in enumerate(cats):
    key(99, 16.5 + i * 7.7, colorat(121, 379, 33 + 12.92 * i), t, 6.85)
# 5B and C directly use the released tables and original saturations, marker areas and significance coding.
letter(178, 3, "B")
rich(191, 3, 217, 12, "Cell-type marker expression", 8.5, True)
typ = [
    "Type I",
    "Type II",
    "Specialized MF",
    "MuSC",
    "FAP",
    "EC",
    "SMC",
    "Adipocyte",
    "Myeloid cell",
    "Lymphocyte",
]
v = tab(args.snrna, "celltype_marker_dotplot_values.tsv.gz")
genes = list(v.gene.drop_duplicates())
assert len(genes) == 28
x, y, w, h = 237, 18, 232, 66
for i, t in enumerate(typ):
    rich(182, y + h * (i + 0.5) / 10 - 4.2, 52, 8.4, t, 7.45, al="right")
for j, g in enumerate(genes):
    vlabel(x + w * (j + 0.5) / 28, y + h + 1, g, 7.05, 32)
for edge in (4.5, 8.5, 11.5, 15.5, 17.5, 19.5, 23.5, 25.5):
    xx = x + w * (edge + 0.5) / 28
    line(xx, y, xx, y + h, "CCCCCC", 0.35)
for _, r in v.iterrows():
    xx = x + w * (genes.index(r.gene) + 0.5) / 28
    yy = y + h * (typ.index(r.cell_type) + 0.5) / 10
    dot(
        xx,
        yy,
        0.56 * math.sqrt(7 + 105 * r.pct_expressing),
        rgb(colormaps["RdBu_r"]((min(2, max(-2, r.mean_z_by_gene)) + 2) / 4)),
        f"{r.cell_type} | {r.gene}",
        "888888",
        0.1,
    )
rect(x, y, w, h, "", "Marker expression plot", "444444", 0.45)
rich(475, 16, 59, 19, "Expression\n(gene z-score)", 6.9)
ramp(479, 39, 45, "−2", "2")
rich(475, 58, 59, 19, "Nuclei\nexpressing", 6.9)
for j, pc in enumerate([10, 50, 90]):
    yy = 83 + j * 9
    dot(
        482,
        yy,
        0.56 * math.sqrt(7 + 105 * pc / 100),
        "",
        "Nuclei expressing",
        "777777",
        0.45,
    )
    rich(490, yy - 4, 36, 8, f"{pc}%", 7.2)
letter(5, 116, "C")
rich(18, 116, 210, 12, "CoQ pathway expression vs young donors", 8.4, True)
st = tab(args.snrna, "COQ_statistics.tsv")
types = ["MuSC", "Type I", "Type II", "Specialized MF"]
st = st[st.cell_type.isin(types)]
gen = list(st.gene.drop_duplicates())
lim = max(1, np.nanpercentile(np.abs(st.log2_fold_change), 95))
x, y, w, h = 39, 132, 121, 95
assert len(gen) == 13
rect(x, y + h * gen.index("COQ8A") / 13, w, h / 13, "F7EEF4", "COQ8A highlight")
for j in [2, 4, 6]:
    line(x + w * j / 8, y, x + w * j / 8, y + h, "CCCCCC", 0.35)
for i, g in enumerate(gen):
    rich(7, y + h * (i + 0.5) / 13 - 4.1, 30, 8.2, g, 7.55, al="right", italic=True)
for _, r in st.iterrows():
    xx = x + w * (2 * types.index(r.cell_type) + ("<" in r.comparison) + 0.5) / 8
    yy = y + h * (gen.index(r.gene) + 0.5) / 13
    p = r.p_adjusted_bh_within_gene_celltype
    if not np.isfinite(p):
        continue
    dia = 0.57 * math.sqrt(22 + 33 * min(4, -math.log10(max(p, 1e-12))))
    dot(
        xx,
        yy,
        dia,
        rgb(colormaps["RdBu_r"](np.clip((r.log2_fold_change / lim + 1) / 2, 0, 1))),
        f"{r.gene} | {r.cell_type} | {r.comparison}",
        "9B1B67" if p < 0.05 else "5A5A5A",
        0.65 if p < 0.05 else 0.2,
    )
    if p < 0.05:
        rich(
            xx + 2.5,
            yy - 5.0,
            10,
            7,
            "***" if p < 0.001 else "**" if p < 0.01 else "*",
            5.8,
            True,
            c="9B1B67",
        )
rect(x, y, w, h, "", "Pathway dot plot", "444444", 0.4)
for i, t in enumerate(["MuSC", "Type I", "Type II", "Sp. MF"]):
    rich(x + i * w / 4, y + h + 9, w / 4, 9, t, 7.2, al="center")
    for j, bi in enumerate(["=", "<"]):
        rich(x + (i * 2 + j) * w / 8, y + h, w / 8, 9, bi, 7, al="center")
rich(167, 133, 76, 10, "log₂ fold change", 7.3)
ramp(170, 146, 46, "−" + f"{lim:.1f}", f"{lim:.1f}")
rich(167, 164, 84, 10, "BH-adjusted p", 7.3)
for i, pval in enumerate([0.32, 0.1, 0.01]):
    yy = 181 + i * 9
    dot(
        170,
        yy,
        0.57 * math.sqrt(22 + 33 * -math.log10(pval)),
        "",
        "q-size key",
        "555555",
        0.42,
    )
    rich(177, yy - 4, 36, 8, str(pval), 7.1)
rich(165, 202, 88, 10, "* adjusted p < 0.05", 6.9, c="9B1B67")
rich(165, 213, 85, 21, "=  older BI = 100\n<  older BI < 100", 7)
letter(245, 116, "D")
rich(258, 116, 229, 12, "COQ8A expression by donor", 8.4, True)
pb = tab(args.snrna, "COQ_donor_pseudobulk.tsv.gz")
pb = pb[(pb.gene == "COQ8A") & pb.cell_type.isin(types)]
gr = ["Young", "Older, BI=100", "Older, BI<100"]
cl = ["4C78A8", "F2A541", "D64B4B"]
for j, g in enumerate(gr):
    key(299 + j * 72, 130, cl[j], g, 6.8)
x0, y0, gw, gh = 278, 148, 248, 81
Y = lambda v: y0 + (6.7 - v) * gh / (6.7 + 0.22)
for z in range(7):
    line(x0, Y(z), x0 + gw, Y(z), "DDDDDD", 0.32)
    rich(264, Y(z) - 4, 12, 8, str(z), 6.9, al="right")
line(x0, y0, x0, y0 + gh)
line(x0, y0 + gh, x0 + gw, y0 + gh)
rich(237, 184, 49, 8, "ln(CPM + 1)", 7, al="center", rot=270)
for j, ct in enumerate(types):
    c = x0 + (j + 0.5) * gw / 4
    rich(
        c - 25,
        230,
        50,
        10,
        "Sp. MF" if ct == "Specialized MF" else ct,
        7.4,
        al="center",
    )
    for k, g in enumerate(gr):
        av = pb[(pb.cell_type == ct) & (pb.group == g)].log1p_cpm.dropna().to_numpy()
        unit = gw / 4 / 1.18
        cx = c + (k - 1) * 0.24 * unit
        if len(av) == 0:
            continue
        lq, md, hq = np.percentile(av, [25, 50, 75])
        iq = hq - lq
        wl = min(av[av >= lq - 1.5 * iq])
        wh = max(av[av <= hq + 1.5 * iq])
        chex = rgb(np.array(colors.to_rgb("#" + cl[k])) * 0.28 + 0.72)
        line(cx, Y(wl), cx, Y(wh), cl[k], 0.6)
        for vl in [wl, wh]:
            line(cx - 2.8, Y(vl), cx + 2.8, Y(vl), cl[k], 0.6)
        rect(
            cx - 0.10 * unit,
            Y(hq),
            0.20 * unit,
            max(0.08, Y(lq) - Y(hq)),
            chex,
            ct + " " + g,
            cl[k],
            0.5,
        )
        line(cx - 0.10 * unit, Y(md), cx + 0.10 * unit, Y(md), "202020", 0.7)
        for vi, axv in enumerate(av):
            dot(
                cx
                + (
                    np.linspace(-0.075 * unit, 0.075 * unit, len(av))[vi]
                    if len(av) > 1
                    else 0
                ),
                Y(axv),
                2.7,
                cl[k],
                ct + " " + g,
                "FFFFFF",
                0.22,
            )
    if j == 0:
        q = st[
            (st.gene == "COQ8A") & (st.cell_type == ct)
        ].p_adjusted_bh_within_gene_celltype
        assert np.allclose(q, 0.033, atol=0.001)
        for k in [0, 1]:
            cy = 149 + k * 6
            right = c + 0.24 * unit * k
            line(c - 0.24 * unit, cy, right, cy, "222222", 0.5)
            line(right, cy, right, cy + 2, "222222", 0.5)
            line(c - 0.24 * unit, cy, c - 0.24 * unit, cy + 2, "222222", 0.5)
        rich(281, 140, 98, 9, "Both BH-adjusted p = 0.033", 6.9)
# 5E/F: preserve the complete tissue area, scale bars, original colors and physical magnification.
letter(5, 247, "E")
letter(307, 247, "F")
maps = [
    (127, (10, 30, 331, 396)),
    (126, (12, 30, 364, 396)),
    (129, (10, 30, 331, 396)),
    (128, (12, 30, 364, 396)),
]
tiles = []
for n, box in maps:
    im = cut(n, box)
    ar = np.asarray(im)
    mask = ar.max(axis=2) < 60
    from scipy.ndimage import find_objects, label

    widths = []
    for rr in mask[int(im.height * 0.84) :]:
        sel = find_objects(label(rr)[0])
        widths.extend([part[0].stop - part[0].start for part in sel])
    bw = max(widths)
    assert bw > 140, (n, bw)
    tiles.append((n, im, bw))
scale = min(
    77 / max(im.height / bw for _, im, bw in tiles),
    71 / max(im.width / bw for _, im, bw in tiles),
)
for j, ((n, im, bw), x0) in enumerate(zip(tiles, [17, 92, 319, 394])):
    w0 = im.width * scale / bw
    h0 = im.height * scale / bw
    rich(x0 - 1, 247, 73, 11, ["Young", "Geriatric muscle"][j % 2], 7.3, True, "center")
    img(
        x0 + (71 - w0) / 2,
        260 + (78 - h0) / 2,
        w0,
        h0,
        im,
        "Tissue map with original 500 µm bar",
    )
    rect(x0, 260, 71, 78, "", "Spatial image frame", "444444", 0.55)
legendE = [
    "B cells",
    "Dendritic cells",
    "Endothelial cells",
    "Erythrocytes",
    "FAPs",
    "Fusing myocytes",
    "Monocytes/Macrophages",
    "MuSCs",
    "Myonuclei",
    "NK cells",
    "Neutrophils",
    "Pericytes and smooth muscle cells",
    "Schwann and Neural/Glial cells",
    "T cells",
    "Tenocytes",
]
rich(21, 338.5, 139, 8, "Scale bars, 500 µm", 6.8, al="center")
rich(323, 338.5, 139, 8, "Scale bars, 500 µm", 6.8, al="center")
for i, t in enumerate(legendE):
    key(168, 247 + i * 6.6, colorat(126, 388, 129 + 11.9 * i), t, 6.65)
for i, (cc, t) in enumerate(
    [
        ("8428BC", "Coq8a⁺ bead"),
        ("FA8B16", "Fusing myocytes"),
        ("287DC1", "MuSC label"),
        ("C0C0C0", "Other injury bead"),
    ]
):
    key(467, 267 + i * 17, cc, t, 6.6)
# Spatial quantitative panels use all original published release rows; glyphs, ticks, intervals and labels remain editable.
blue = "367BB1"
red = "D54C41"
ages = ["Young", "Geriatric"]
cols = [blue, red]
letter(5, 349, "G")
rich(18, 349, 214, 11, "Myogenic transcripts by bead label", 8.1, True)
markers = tab(args.spatial, "fusing_marker_audit.tsv")
genes = list(markers.gene.drop_duplicates())
mx = markers[["mean_log1p10k_fusing", "mean_log1p10k_other_injury"]].max().max()
x, y, w, h = 36, 371, 125, 58
rect(x, y, w / 2, h, "F2F7FB", "Young")
rect(x + w / 2, y, w / 2, h, "FDF2F2", "Geriatric")
for j, g in enumerate(genes):
    rich(6, y + h * (j + 0.5) / len(genes) - 4, 28, 8, g, 7, al="right", italic=True)
for i, age in enumerate(ages):
    rich(x + i * w / 2, y - 10, w / 2, 9, age, 7.4, True, "center")
    for _, r in markers[markers["sample"] == age].iterrows():
        for j, suffix in enumerate(["fusing", "other_injury"]):
            sz = 0.63 * math.sqrt(6 + 95 * r["detection_" + suffix])
            dot(
                x + (2 * i + j + 0.5) * w / 4,
                y + (genes.index(r.gene) + 0.5) * h / len(genes),
                sz,
                rgb(colormaps["magma"](r["mean_log1p10k_" + suffix] / mx)),
                f"{age} | {r.gene} | {suffix}",
            )
for i, t in enumerate(["Fusing", "Other", "Fusing", "Other"]):
    rich(x + i * w / 4, y + h + 1, w / 4, 8, t, 7, al="center")
rich(x, y + h + 9, w, 8, "Fusing myocytes / other injury", 6.7, al="center")
rich(169, 370, 75, 20, "Mean expression\nln(C/10k + 1)", 7.0)
for j in range(30):
    rect(
        170 + j * 1.9,
        395,
        1.92,
        4,
        rgb(colormaps["magma"](j / 29)),
        f"Expression scale {j}",
    )
rich(167, 399, 12, 8, "0", 6.7)
rich(216, 399, 22, 8, f"{mx:.2f}", 6.7)
rich(167, 411, 77, 9, "Detected beads", 6.9)
for j, val in enumerate([10, 30, 50]):
    xx = 168 + j * 24
    dot(xx + 2, 425, 0.63 * math.sqrt(6 + 95 * val / 100), "AAAAAA", f"{val}%")
    rich(xx + 5, 421, 20, 8, f"{val}%", 6.6)
letter(248, 349, "H")
rich(261, 349, 190, 11, "Coq8a detection by cell label", 8.1, True)
assn = tab(args.spatial, "coq8a_celltype_association.tsv")
labs = [
    "Fusing Myocytes",
    "Myonuclei",
    "MuSCs",
    "FAPs",
    "Monocytes/Macrophages",
    "Dendritic cells",
    "Endothelial cells",
    "Tenocytes",
]
shortlabs = [
    "Fusing myocytes",
    "Myonuclei",
    "MuSCs",
    "FAPs",
    "Monocytes/Mφ",
    "Dendritic cells",
    "Endothelial cells",
    "Tenocytes",
]
xx, yy, ww, hh = 321, 365, 150, 65
zs = assn[assn.cell_type.isin(labs)]
xmin = min(-1.5, np.log10(zs.odds_ratio_ci_low.min()) - 0.05)
xmax = max(1.5, np.log10(zs.odds_ratio_ci_high.max()) + 0.05)
X = lambda v: xx + ww * (np.log10(v) - xmin) / (xmax - xmin)
rect(xx, yy, ww, hh / 8, "F7EEF4", "Fusing myocytes")
line(X(1), yy, X(1), yy + hh, "888888", 0.45)
for j, cell in enumerate(labs):
    cy = yy + hh * (j + 0.5) / 8
    rich(249, cy - 4, 70, 8, shortlabs[j], 7.1, al="right")
    for i, age in enumerate(ages):
        rr = zs[(zs["sample"] == age) & (zs.cell_type == cell)].iloc[0]
        a, b = X(rr.odds_ratio_ci_low), X(rr.odds_ratio_ci_high)
        dy = cy + [-1.4, 1.4][i]
        line(a, dy, b, dy, cols[i], 0.65)
        line(a, dy - 1, a, dy + 1, cols[i], 0.55)
        line(b, dy - 1, b, dy + 1, cols[i], 0.55)
        dot(X(rr.odds_ratio_plot), dy, 3, cols[i], f"{age} | {cell}")
        if cell == labs[0]:
            dot(X(rr.odds_ratio_plot), dy, 4.6, "", "Significant", "9B1B67", 0.5)
            rich(
                475,
                393 + i * 9,
                59,
                8,
                f"q = {rr.p_adj_bh_within_section_15_labels:.5f}",
                6.6,
                c=cols[i],
            )
line(xx, yy + hh, xx + ww, yy + hh)
for val in [0.1, 1, 10]:
    rich(X(val) - 9, yy + hh + 0.5, 18, 8, str(val), 6.9, al="center")
rich(323, yy + hh + 9, 150, 8, "Odds ratio (95% CI)", 7.2, al="center")
for j, age in enumerate(ages):
    key(475, 369 + 9 * j, cols[j], age, 7)
letter(5, 449, "I")
letter(249, 449, "J")
rich(18, 449, 220, 10, "Coq8a-positive beads by region", 8.1, True)
rich(262, 449, 265, 10, "Coq8a-positive bead density near fusing myocytes", 8.1, True)
for panel in ["I", "J"]:
    if panel == "I":
        x, y, w, h = 34, 465, 176, 55
        lo, hi = 0.6, 1.6
        ticks = np.arange(0.6, 1.61, 0.2)
        xmin, xmax = -0.05, 1.05
    else:
        x, y, w, h = 296, 465, 209, 55
        lo, hi = 0.99, 1.26
        ticks = [1, 1.1, 1.2]
        xmin, xmax = 20, 160
    XX = lambda a: x + (a - xmin) * w / (xmax - xmin)
    YY = lambda a: y + (hi - a) * h / (hi - lo)
    line(x, y, x, y + h)
    line(x, y + h, x + w, y + h)
    for tk in ticks:
        rich(x - 20, YY(tk) - 4, 18, 8, f"{tk:.1f}", 7.0, al="right")
    for j, ag in enumerate(ages):
        if panel == "I":
            d = tab(args.spatial, "coq8a_region_detection.tsv")
            d = d[d["sample"] == ag].set_index("region").loc[["Outside", "Injury"]]
            xs = [0, 1]
            ys = list(d.coq8a_detection_pct)
        else:
            d = tab(args.spatial, "coq8a_fusion_neighborhood.tsv")
            d = d[d["sample"] == ag].sort_values("radius_um")
            xs = list(d.radius_um)
            ys = list(d.observed_to_random_ratio)
        for i, (ax, ay) in enumerate(zip(xs, ys)):
            if i:
                line(XX(xs[i - 1]), YY(ys[i - 1]), XX(ax), YY(ay), cols[j], 0.85)
            if j and panel == "J":
                rect(XX(ax) - 1.4, YY(ay) - 1.4, 2.8, 2.8, cols[j], "Geriatric")
            else:
                dot(XX(ax), YY(ay), 3, cols[j], ag)
            if panel == "J" and d.iloc[i].p_adj_bh_across_10_radius_tests < 0.05:
                rich(XX(ax) - 4, YY(ay) - 9, 8, 8, "*", 8, True, "center", cols[j])
        if panel == "I":
            key(127 + j * 47, 458, cols[j], ag, 6.8)
        else:
            rich(505, YY(ys[-1]) - 4, 34, 8, ag, 6.6, c=cols[j])
    if panel == "I":
        for j, t in enumerate(["Outside injury", "Injury"]):
            rich(XX(j) - 24, 521, 48, 8, t, 7, al="center")
        rich(0, 487, 29, 7, "Beads (%)", 6.8, al="center", rot=270)
    else:
        for t in [25, 50, 75, 100, 150]:
            rich(XX(t) - 9, 521, 18, 8, t, 7, al="center")
        rich(335, 529, 141, 8, "Radius around fusing myocytes (µm)", 7.1, al="center")
        rich(438, 461, 87, 8, "* BH-adjusted p < 0.05", 6.4)
        rich(237, 487, 60, 7, "Observed / random", 6.8, al="center", rot=270)
rich(
    5,
    535,
    524,
    9,
    "A–D: Lai et al., 2024.   E–J: Walter et al., 2024 (GSE266933).",
    7.15,
)
s = pres.slides.add_slide(pres.slide_layouts[6])
letter(5, 4, "A")
rich(18, 4, 280, 12, "Reaction potential in MuSCs", 9.2, True)
letter(346, 4, "B")
rich(360, 4, 172, 12, "CoQ synthesis in MuSCs", 9.2, True)
donors = tab(args.compass, "COMPASS_CoQ_donor_audit_20261102.tsv").sort_values(
    ["age_label", "donor_id"],
    key=lambda a: a.map({"Young": 0, "Older": 1}) if a.name == "age_label" else a,
)
assert donors.donor_id.tolist() == [
    "P13",
    "P26",
    "P5",
    "YM1",
    "YM2",
    "YM3",
    "YM4",
    "P17",
    "P21",
    "P23",
    "P27",
    "P29",
    "P3",
]
heat = thumb(200)
rx = [
    "CoQ synthesis",
    "PDH",
    "ACLY",
    "ACS",
    "Citrate synthase",
    "Aconitase",
    "IDH3",
    "IDH2",
    "OGDH",
    "SCS (ATP)",
    "SCS (GTP)",
    "SDH",
    "Fumarase",
    "MDH2",
    "Pyruvate carboxylase",
]
x = 87
y = 73
w = 224
h = 135
for gi, (start, count, title) in enumerate(
    [(0, 7, "Young (n = 7)"), (7, 6, "Older (n = 6)")]
):
    rich(x + start * w / 13, 18, count * w / 13, 12, title, 8.3, True, "center")
for k, lab in enumerate(["Age", "Cohort", "BI", "Nuclei"]):
    rich(x - 44, 32 + 9 * k, 41, 8, lab, 7.6, al="right")
for j, (_, r) in enumerate(donors.iterrows()):
    xx = x + j * w / 13
    dx = w / 13
    for k, text in enumerate(
        [
            "",
            "",
            ("NA" if pd.isna(r.barthel_index) else str(int(r.barthel_index))),
            str(int(r.n_nuclei)),
        ]
    ):
        # Sample each colour in blank cell background, away from values, directly from the approved archived heatmap.
        cc = colorat(200, 136 + (j + 0.10) * (559 - 136) / 13, [33, 47, 61, 76][k])
        rect(xx, 33 + k * 9, dx + 0.02, 7.8, cc, "Donor annotation " + r.donor_id)
        if text:
            rich(
                xx,
                32 + k * 9,
                dx,
                9,
                text,
                7.25,
                al="center",
                c=("FFFFFF" if sum(bytes.fromhex(cc)) / 3 < 110 else "111111"),
            )
    for k, enzyme in enumerate(rx):
        cc = colorat(
            200, 136 + (j + 0.5) * (559 - 136) / 13, 96 + (k + 0.5) * (384 - 96) / 15
        )
        rect(
            xx,
            y + k * h / 15,
            dx + 0.03,
            h / 15 + 0.03,
            cc,
            enzyme + " | " + r.donor_id,
        )
    vlabel(xx + dx / 2, y + h + 2, r.donor_id, 7.8, 22)
for i, t in enumerate(rx):
    rich(7, y + (i + 0.5) * h / 15 - 4.6, 78, 9.2, t, 8.25, al="right")
for xx, c, t in [
    (9, colorat(200, 149, 46), "European"),
    (85, colorat(200, 271, 46), "Asian Chinese"),
]:
    rect(xx, 241, 8, 6, c, t)
    rich(xx + 11, 238, 70, 11, t, 7.5)
rich(166, 238, 76, 11, "Reaction z-score", 7.5)
ramp(245, 241, 58, "−2", "2")
# The numeric values and original donor glyphs come directly from the archived COMPASS panel.
place(
    203,
    (88, 96, 442, 316),
    368,
    49,
    154,
    123,
    "Donor groups - original data points and medians",
)
# Preserve coordinate values also if source viewport is resized by the fit operation.
im = cut(203, (88, 96, 442, 316))
sc = min(154 / im.width, 123 / im.height)
yy = 49 + (123 - sc * im.height) / 2
hh = sc * im.height
for f0, lab in [
    ((106 - 96) / 220, "−10.8"),
    ((168 - 96) / 220, "−11.0"),
    ((232 - 96) / 220, "−11.2"),
    ((296 - 96) / 220, "−11.4"),
]:
    rich(340, yy + hh * f0 - 4.5, 26, 9, lab, 7.4, al="right")
rich(349, 21, 180, 11, "p = 0.014; q = 0.030", 8, al="center")
rich(397, yy + hh + 3, 35, 10, "Young", 8, al="center")
rich(469, yy + hh + 3, 35, 10, "Older", 8, al="center")
rich(361, yy - 12, 86, 9, "COMPASS score", 7.6)
for j, lab in enumerate(["European", "Asian Chinese"]):
    xx = 370 + j * 68
    rich(xx, yy + hh + 23, 10, 10, "●" if j == 0 else "▲", 8, True)
    rich(xx + 9, yy + hh + 23, 60, 10, lab, 7.2)
letter(5, 259, "C")
rich(18, 259, 240, 12, "All 565 candidate peaks", 9, True)
rich(280, 259, 240, 12, "Complete peak screen", 9, True)
rd = args.atac / "results/fixed_day0_D206"
peaks = pd.read_csv(rd / "all_peaks.tsv", sep="\t")
hits = pd.read_csv(rd / "selected_regions.tsv", sep="\t")
assert len(peaks) == 565 and peaks.n_pairs.eq(527).all()
# Read the original heat map texture; each of four columns preserves the order/scale of the 565 archived rows.
img(
    49,
    277,
    176,
    109,
    cut(201, (68, 34, 269, 208)),
    "Original peak accessibility matrix, 565 rows",
)
for j, t in enumerate(
    ["Source 1\nLow", "Source 1\nHigh", "Source 2\nLow", "Source 2\nHigh"]
):
    rich(49 + 44 * j - 2, 387, 48, 16, t, 7.1, al="center")
rich(27, 273, 19, 9, "1", 7.1, al="right")
rich(25, 381, 21, 9, "565", 7.1, al="right")
rich(-12, 331, 91, 8, "Mean accessibility rank", 7.5, al="center", rot=270)
rich(11, 405, 82, 11, "Accessible nuclei (%)", 7.2)
for i in range(40):
    rect(
        99 + 2.8 * i,
        407,
        2.83,
        5,
        rgb(colormaps["Blues"](i / 39)),
        f"Accessible nuclei {i}",
    )
for i in [0, 50, 100]:
    rich(99 + 112 * i / 100 - 9, 413, 18, 8, str(i), 6.9, al="center")
x, y, w, h = 297, 277, 227, 110
xx = np.log2((peaks.high + 0.5) / (peaks.low + 0.5))
yy = -np.log10(peaks.p)
xl = min(-2.5, float(xx.min()) - 0.1)
xh = max(3.3, float(xx.max()) + 0.1)
yh = max(4.65, float(yy.max()) + 0.1)
XC = lambda v: x + w * (v - xl) / (xh - xl)
YC = lambda v: y + h * (1 - v / yh)
for r in sorted(peaks.index, key=lambda i: peaks.q565[i] < 0.05):
    sig = peaks.q565[r] < 0.05
    cc = ("BC5144" if peaks.FC[r] > 1 else "3276A7") if sig else "C4C9CA"
    dot(XC(xx[r]), YC(yy[r]), 2.8 if sig else 1.25, cc, peaks.peak[r])
line(x, y, x, y + h)
line(x, y + h, x + w, y + h)
line(XC(0), y, XC(0), y + h, "9A9A9A", 0.4)
for t in [-2, -1, 0, 1, 2, 3]:
    rich(XC(t) - 8, y + h + 2, 16, 8, str(t), 7, al="center")
for t in range(5):
    rich(x - 15, YC(t) - 4, 12, 9, str(t), 7.4, al="right")
rich(233, 327, 81, 9, "−log₁₀ paired peak p", 7.4, al="center", rot=270)
rich(332, 403, 167, 10, "log₂ accessibility ratio (high / low)", 7.6, al="center")
offsets = {
    "MYOD1": (8, -9),
    "CSRP3": (6, 5),
    "CACNA1H": (10, -6),
    "BOC": (4, -9),
    "NOTCH1": (-26, 4),
    "CAV3": (-10, 10),
}
for r in hits[hits.q25 < 0.05].itertuples():
    j = peaks.index[peaks.peak == r.top_peak][0]
    cx, cy = XC(xx[j]), YC(yy[j])
    dx, dy = offsets[r.gene]
    line(cx, cy, cx + dx, cy + dy + 3, "777777", 0.35)
    rich(cx + dx, cy + dy, 41, 8, r.gene, 6.9)
rich(466, 327, 63, 9, "Peak q < 0.05", 7.2)
key(467, 337, "BC5144", "Opening", 7)
key(467, 347, "3276A7", "Closing", 7)
letter(5, 423, "D")
rich(18, 423, 141, 12, "MYOD1 locus", 9.0, True)
# Match source profile aggregation and 100-bp Gaussian smoothing; create two editable PowerPoint polylines.
from scipy.ndimage import gaussian_filter1d

pf = pd.read_csv(rd / "fragment_profiles.tsv.gz", sep="\t")
windows = pd.read_csv(rd / "profile_windows.tsv", sep="\t")
win = windows[(windows.kind == "locus") & (windows.id == "MYOD1")].iloc[0]
pr = (
    pf[(pf.kind == "locus") & (pf.id == "MYOD1")]
    .groupby(["group", "bin"])
    .per100_nuclei_per100bp.mean()
    .reset_index()
)
x, y, w, h = 56, 444, 467, 53
start, stop = win.start / 1e6, win.end / 1e6
X = lambda v: x + w * (v - start) / (stop - start)
Y = lambda v: y + h * (1 - v / 22)
rect(
    X(17.653349), y, X(17.654252) - X(17.653349), h, "F7D59A", "Highlighted MYOD1 peak"
)
for group, cc in [("low", "818181"), ("high", "2C80BA")]:
    z = pr[pr.group == group].sort_values("bin")
    left = win.start + 100 * z.bin.to_numpy()
    xp = (left + np.minimum(100, win.end - left) / 2) / 1e6
    yp = gaussian_filter1d(z.per100_nuclei_per100bp.to_numpy(), 1)
    assert max(yp) < 22
    coords = [(round(X(a) * 100), round(Y(b) * 100)) for a, b in zip(xp, yp)]
    ff = s.shapes.build_freeform(coords[0][0], coords[0][1], scale=Pt(1) / 100)
    ff.add_line_segments(coords[1:], close=False)
    sh = ff.convert_to_shape()
    sh.name = "COQ8A " + group + " | 100-bp bins"
    sh.fill.background()
    sh.line.color.rgb = RGBColor.from_string(cc)
    sh.line.width = Pt(0.8)
    sh._element.spPr.append(OxmlElement("a:effectLst"))
line(x, y, x, y + h)
line(x, y + h, x + w, y + h)
for val in [0, 5, 10, 15, 20]:
    rich(36, Y(val) - 4, 17, 8, str(val), 7.25, al="right")
for val in [17.65, 17.66, 17.67, 17.68, 17.69, 17.70, 17.71, 17.72]:
    if val >= start and val <= stop:
        rich(X(val) - 15, y + h + 1, 30, 9, f"{val:.2f}", 7.3, al="center")
rich(190, 508, 171, 10, "Chromosome 11 position (Mb)", 7.8, al="center")
rich(
    -13,
    468,
    86,
    16,
    "ATAC insertions\n/ 100 nuclei / 100 bp",
    6.8,
    al="center",
    rot=270,
)
for i, (cc, t) in enumerate([("818181", "COQ8A low"), ("2C80BA", "COQ8A high")]):
    line(242 + i * 78, 437, 253 + i * 78, 437, cc, 1.05)
    rich(256 + i * 78, 433, 59, 9, t, 7.2)
tss = X(win.center / 1e6)
line(tss, y, tss, y + h, "666666", 0.4)
rich(tss - 40, y - 8, 66, 8, "MYOD1 TSS", 7.1, al="center")
rich(
    5,
    522,
    526,
    10,
    "A–B: Lai et al., 2024; COMPASS: Wagner et al., 2021.   C–D: Wang et al., 2023 (GSE208248).",
    7.3,
)
args.out.parent.mkdir(parents=True, exist_ok=True)
pres.save(args.out)
