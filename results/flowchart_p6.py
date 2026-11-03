"""
flowchart_p6.py — flowchart penelitian (Figure 1 naskah JUTIF) dengan simbol standar.

Simbol mengikuti konvensi flowchart (ANSI/ISO 5807, seperti draw.io/RAPTOR):
  terminal (Start/End) = persegi panjang bersudut bulat penuh, proses = persegi panjang,
  data = jajaran genjang, keputusan = belah ketupat, konektor antar-kolom = lingkaran "A".

Satu definisi simpul & panah menghasilkan DUA berkas dengan tata letak identik:
  artikel/figures/fig_p6_flowchart.png     untuk naskah (300 dpi, ~ukuran cetak)
  artikel/figures/fig_p6_flowchart.drawio  untuk disunting di draw.io / app.diagrams.net

Isi tiap kotak diambil dari bab Method; jangan menambah tahap yang tidak ada di sana.

    python results/flowchart_p6.py
"""
from __future__ import annotations

import html
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "artikel" / "figures"
W, H = 6.5, 6.55                       # inci; lebar = lebar teks halaman
FONT = "Arial"
UK = 8.0                                # ukuran huruf (pt) — tercetak hampir 1:1
# Polos (permintaan user 2026-09-28): garis hitam, isi putih untuk semua simbol.
WARNA = {k: "#ffffff" for k in ("terminal", "proses", "data", "keputusan", "konektor")}
GARIS = "#000000"

C1, C2 = 1.65, 4.90                     # pusat kolom kiri & kanan
# id: (jenis, cx, cy, lebar, tinggi, teks) — koordinat inci, y dihitung dari ATAS
SIMPUL = {
    "mulai": ("terminal", C1, 0.35, 1.30, 0.40, "Start"),
    "korpus": ("data", C1, 0.95, 2.60, 0.46, "Alodokter patient–physician\nquestion–answer corpus"),
    "praproses": ("proses", C1, 1.60, 2.60, 0.50, "Preprocessing: 12 quality filters,\naudit and targeted repairs"),
    "split": ("data", C1, 2.25, 2.60, 0.50, "Train 20,000 / validation 2,947 /\ntest 2,998"),
    "pilot": ("proses", C1, 2.95, 2.60, 0.55, "Pilot fine-tuning\n(Qwen3.5 0.8B and 2B; 250 steps)"),
    "cek": ("keputusan", C1, 3.85, 2.20, 0.85, "Pilot passes\nall four checks?"),
    "latih": ("proses", C1, 4.80, 2.60, 0.55, "Full LoRA fine-tuning (bf16, 3 epochs,\nearly stopping, best validation loss)"),
    "ekspor": ("proses", C1, 5.55, 2.60, 0.55, "Export: merge LoRA matrices →\n16-bit model → GGUF Q4_K_M"),
    "A1": ("konektor", C1, 6.22, 0.36, 0.36, "A"),
    "A2": ("konektor", C2, 0.35, 0.36, 0.36, "A"),
    "tier1": ("proses", C2, 1.02, 2.80, 0.62, "Tier 1: token-F1, ROUGE-L, BERTScore;\ninter-physician agreement; ablations\n(RQ1, RQ2)"),
    "tier2": ("proses", C2, 1.82, 2.80, 0.62, "Tier 2: blind paired LLM judge;\nred-flag escalation analysis\n(RQ4)"),
    "device": ("proses", C2, 2.62, 2.80, 0.62, "On-device test: Q4_K_M under llama.cpp\non a laptop and an Android smartphone\n(RQ3)"),
    "gerbang": ("keputusan", C2, 3.60, 2.50, 0.90, "Peak RAM ≤ 55% of the\nsmartphone class RAM?"),
    "layak": ("proses", 4.15, 4.62, 1.35, 0.50, "Size passes\nthis class"),
    "tidak": ("proses", 5.60, 4.62, 1.30, 0.50, "Size fails\nthis class"),
    "banding": ("proses", C2, 5.45, 2.80, 0.55, "Compare quality, safety and device\ncost across the three sizes"),
    "selesai": ("terminal", C2, 6.22, 1.30, 0.40, "End"),
}


def tepi(i, sisi, simpul=None):
    _, cx, cy, w, h, _ = (simpul or SIMPUL)[i]
    return {"atas": (cx, cy - h / 2), "bawah": (cx, cy + h / 2),
            "kiri": (cx - w / 2, cy), "kanan": (cx + w / 2, cy)}[sisi]


# Panah: (dari, sisi, ke, sisi, titik belok, label, posisi label, berujung panah)
PANAH = [
    ("mulai", "bawah", "korpus", "atas", [], "", None, True),
    ("korpus", "bawah", "praproses", "atas", [], "", None, True),
    ("praproses", "bawah", "split", "atas", [], "", None, True),
    ("split", "bawah", "pilot", "atas", [], "", None, True),
    ("pilot", "bawah", "cek", "atas", [], "", None, True),
    ("cek", "bawah", "latih", "atas", [], "Yes", (C1 + 0.12, 4.36), True),
    ("cek", "kiri", "pilot", "kiri", [(0.15, 3.85), (0.15, 2.95)], "No", (0.40, 3.74), True),
    ("latih", "bawah", "ekspor", "atas", [], "", None, True),
    ("ekspor", "bawah", "A1", "atas", [], "", None, True),
    ("A2", "bawah", "tier1", "atas", [], "", None, True),
    ("tier1", "bawah", "tier2", "atas", [], "", None, True),
    ("tier2", "bawah", "device", "atas", [], "", None, True),
    ("device", "bawah", "gerbang", "atas", [], "", None, True),
    ("gerbang", "bawah", "layak", "atas", [(C2, 4.20), (4.15, 4.20)], "Yes", (C2 - 0.28, 4.13), True),
    ("gerbang", "kanan", "tidak", "kanan", [(6.40, 3.60), (6.40, 4.62)], "No", (6.27, 3.49), True),
    ("layak", "bawah", "banding", "atas", [(4.15, 5.02), (C2, 5.02)], "", None, True),
    ("tidak", "bawah", "banding", "atas", [(5.60, 5.02), (C2, 5.02)], "", None, True),
    ("banding", "bawah", "selesai", "atas", [], "", None, True),
]


# ------------------------------------------------------------------ PNG (matplotlib)
def gambar_png(simpul=None, panah=None, W=W, H=H, UK=UK, nama="fig_p6_flowchart.png") -> pathlib.Path:
    simpul, panah = simpul or SIMPUL, panah or PANAH
    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                   # y dari atas, sama dengan draw.io
    ax.axis("off")
    lw = 0.9

    for i, (jenis, cx, cy, w, h, teks) in simpul.items():
        x0, y0 = cx - w / 2, cy - h / 2
        warna = WARNA[jenis]
        if jenis == "terminal":
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                        fc=warna, ec=GARIS, lw=lw))
        elif jenis == "proses":
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="square,pad=0", fc=warna, ec=GARIS, lw=lw))
        elif jenis == "data":
            s = 0.13
            ax.add_patch(Polygon([(x0 + s, y0), (x0 + w, y0), (x0 + w - s, y0 + h), (x0, y0 + h)],
                                 closed=True, fc=warna, ec=GARIS, lw=lw))
        elif jenis == "keputusan":
            ax.add_patch(Polygon([(cx, y0), (x0 + w, cy), (cx, y0 + h), (x0, cy)],
                                 closed=True, fc=warna, ec=GARIS, lw=lw))
        elif jenis == "konektor":
            ax.add_patch(Circle((cx, cy), w / 2, fc=warna, ec=GARIS, lw=lw))
        tebal = "bold" if jenis in ("terminal", "konektor") else "normal"
        ax.text(cx, cy, teks, ha="center", va="center", fontsize=UK, color=GARIS,
                fontweight=tebal, linespacing=1.2)

    for dari, sd, ke, sk, belok, label, pos, ujung in panah:
        titik = [tepi(dari, sd, simpul)] + belok + [tepi(ke, sk, simpul)]
        # Dua cabang yang bertemu di satu titik: ujung panah hanya pada ruas terakhir.
        for (x1, y1), (x2, y2) in zip(titik[:-2], titik[1:-1]):
            ax.plot([x1, x2], [y1, y2], color=GARIS, lw=lw, solid_capstyle="butt")
        (x1, y1), (x2, y2) = titik[-2], titik[-1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>" if ujung else "-", color=GARIS, lw=lw,
                                    mutation_scale=9, shrinkA=0, shrinkB=0))
        if label:
            ax.text(*pos, label, fontsize=UK - 0.5, ha="center", va="center", color=GARIS,
                    fontstyle="italic", bbox=dict(fc="white", ec="none", pad=0.5))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / nama
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ draw.io (mxGraph XML)
PX = 96  # piksel per inci di draw.io
GAYA = {
    "terminal": "rounded=1;arcSize=50;whiteSpace=wrap;html=1;fontStyle=1;",
    "proses": "rounded=0;whiteSpace=wrap;html=1;",
    "data": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=12;",
    "keputusan": "rhombus;whiteSpace=wrap;html=1;",
    "konektor": "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontStyle=1;",
}
SISI_KE_XY = {"atas": (0.5, 0), "bawah": (0.5, 1), "kiri": (0, 0.5), "kanan": (1, 0.5)}


def tulis_drawio() -> pathlib.Path:
    sel = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for i, (jenis, cx, cy, w, h, teks) in SIMPUL.items():
        gaya = (GAYA[jenis] + f"fillColor={WARNA[jenis]};strokeColor={GARIS};"
                f"fontFamily={FONT};fontSize={int(UK * 1.33)};")
        nilai = html.escape(teks).replace("\n", "&lt;br&gt;")
        sel.append(f'<mxCell id="{i}" value="{nilai}" style="{gaya}" vertex="1" parent="1">'
                   f'<mxGeometry x="{(cx - w / 2) * PX:.1f}" y="{(cy - h / 2) * PX:.1f}" '
                   f'width="{w * PX:.1f}" height="{h * PX:.1f}" as="geometry"/></mxCell>')
    for n, (dari, sd, ke, sk, belok, label, _, ujung) in enumerate(PANAH):
        ex, ey = SISI_KE_XY[sd]
        nx, ny = SISI_KE_XY[sk]
        gaya = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={GARIS};"
                f"endArrow={'block' if ujung else 'none'};endFill=1;fontFamily={FONT};"
                f"fontSize={int(UK * 1.33)};fontStyle=2;"
                f"exitX={ex};exitY={ey};entryX={nx};entryY={ny};")
        titik = "".join(f'<mxPoint x="{x * PX:.1f}" y="{y * PX:.1f}"/>' for x, y in belok)
        arr = f'<Array as="points">{titik}</Array>' if belok else ""
        sel.append(f'<mxCell id="e{n}" value="{label}" style="{gaya}" edge="1" parent="1" '
                   f'source="{dari}" target="{ke}"><mxGeometry relative="1" as="geometry">{arr}'
                   f'</mxGeometry></mxCell>')
    xml = ('<mxfile host="flowchart_p6.py"><diagram name="Research flowchart">'
           f'<mxGraphModel dx="800" dy="800" grid="1" gridSize="10" page="1" '
           f'pageWidth="{int(W * PX)}" pageHeight="{int(H * PX)}" background="#ffffff">'
           f'<root>{"".join(sel)}</root></mxGraphModel></diagram></mxfile>')
    p = OUT / "fig_p6_flowchart.drawio"
    p.write_text(xml, encoding="utf-8")
    return p


# ------------------------------------------------------------------ versi satu kolom (RESTI)
# Isi kotak SAMA dengan SIMPUL di atas, hanya baris dipatah ulang supaya muat di satu kolom
# template RESTI (lebar kolom 7,8 cm = 3,07 inci) dan tercetak 1:1 dengan huruf 7 pt.
W1, H1, UK1 = 3.07, 6.12, 7.0
K1, K2 = 0.86, 2.33                     # pusat sub-kolom kiri & kanan
SIMPUL_1KOL = {
    "mulai": ("terminal", K1, 0.19, 0.80, 0.28, "Start"),
    "korpus": ("data", K1, 0.66, 1.42, 0.46, "Alodokter\npatient–physician\nquestion–answer\ncorpus"),
    "praproses": ("proses", K1, 1.26, 1.42, 0.54, "Preprocessing:\n12 quality filters,\naudit and\ntargeted repairs"),
    "split": ("data", K1, 1.90, 1.42, 0.54, "Train 20,000 /\nvalidation 2,947 /\ntest 2,998"),
    "pilot": ("proses", K1, 2.54, 1.42, 0.54, "Pilot fine-tuning\n(Qwen3.5 0.8B and 2B;\n250 steps)"),
    "cek": ("keputusan", K1, 3.36, 1.40, 0.82, "Pilot passes\nall four\nchecks?"),
    "latih": ("proses", K1, 4.22, 1.42, 0.66, "Full LoRA fine-tuning\n(bf16, 3 epochs, early\nstopping, best\nvalidation loss)"),
    "ekspor": ("proses", K1, 4.98, 1.42, 0.66, "Export: merge LoRA\nmatrices → 16-bit\nmodel → GGUF\nQ4_K_M"),
    "A1": ("konektor", K1, 5.56, 0.28, 0.28, "A"),
    "A2": ("konektor", K2, 0.22, 0.28, 0.28, "A"),
    "tier1": ("proses", K2, 0.86, 1.34, 0.80, "Tier 1: token-F1,\nROUGE-L, BERTScore;\ninter-physician\nagreement; ablations\n(RQ1, RQ2)"),
    "tier2": ("proses", K2, 1.74, 1.34, 0.66, "Tier 2: blind paired\nLLM judge; red-flag\nescalation analysis\n(RQ4)"),
    "device": ("proses", K2, 2.62, 1.34, 0.80, "On-device test:\nQ4_K_M under\nllama.cpp on a laptop\nand an Android smartphone\n(RQ3)"),
    "gerbang": ("keputusan", K2, 3.62, 1.34, 0.96, "Peak RAM ≤ 55%\nof the\nsmartphone\nclass RAM?"),
    "layak": ("proses", 1.985, 4.66, 0.64, 0.54, "Size\npasses this\nclass"),
    "tidak": ("proses", 2.675, 4.66, 0.64, 0.54, "Size\nfails this\nclass"),
    "banding": ("proses", K2, 5.40, 1.34, 0.54, "Compare quality,\nsafety and device cost\nacross the three sizes"),
    "selesai": ("terminal", K2, 5.94, 0.80, 0.28, "End"),
}
PANAH_1KOL = [
    ("mulai", "bawah", "korpus", "atas", [], "", None, True),
    ("korpus", "bawah", "praproses", "atas", [], "", None, True),
    ("praproses", "bawah", "split", "atas", [], "", None, True),
    ("split", "bawah", "pilot", "atas", [], "", None, True),
    ("pilot", "bawah", "cek", "atas", [], "", None, True),
    ("cek", "bawah", "latih", "atas", [], "Yes", (K1 + 0.13, 3.83), True),
    ("cek", "kiri", "pilot", "kiri", [(0.06, 3.36), (0.06, 2.54)], "No", (0.10, 3.20), True),
    ("latih", "bawah", "ekspor", "atas", [], "", None, True),
    ("ekspor", "bawah", "A1", "atas", [], "", None, True),
    ("A2", "bawah", "tier1", "atas", [], "", None, True),
    ("tier1", "bawah", "tier2", "atas", [], "", None, True),
    ("tier2", "bawah", "device", "atas", [], "", None, True),
    ("device", "bawah", "gerbang", "atas", [], "", None, True),
    ("gerbang", "bawah", "layak", "atas", [(K2, 4.25), (1.985, 4.25)], "Yes", (K2 - 0.20, 4.17), True),
    ("gerbang", "kanan", "tidak", "atas", [(3.00, 4.25), (2.675, 4.25)], "No", (2.92, 3.86), True),
    ("layak", "bawah", "banding", "atas", [(1.985, 5.03), (K2, 5.03)], "", None, True),
    ("tidak", "bawah", "banding", "atas", [(2.675, 5.03), (K2, 5.03)], "", None, True),
    ("banding", "bawah", "selesai", "atas", [], "", None, True),
]


if __name__ == "__main__":
    for p in (gambar_png(), tulis_drawio(),
              gambar_png(SIMPUL_1KOL, PANAH_1KOL, W1, H1, UK1, "fig_p6_flowchart_1kol.png")):
        print(f"tersimpan: {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
