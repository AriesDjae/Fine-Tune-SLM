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
WARNA = {"terminal": "#dbe7f5", "proses": "#ffffff", "data": "#eaf4e4",
         "keputusan": "#fff3d1", "konektor": "#ffffff"}
GARIS = "#1f1f1f"

C1, C2 = 1.65, 4.90                     # pusat kolom kiri & kanan
# id: (jenis, cx, cy, lebar, tinggi, teks) — koordinat inci, y dihitung dari ATAS
SIMPUL = {
    "mulai": ("terminal", C1, 0.35, 1.30, 0.40, "Start"),
    "korpus": ("data", C1, 0.95, 2.60, 0.46, "Alodokter patient–doctor QA corpus"),
    "praproses": ("proses", C1, 1.60, 2.60, 0.50, "Preprocessing: 12 quality filters,\nrecall audit and fixes"),
    "split": ("data", C1, 2.25, 2.60, 0.50, "Train 20,000 / validation 2,947 /\ntest 2,998"),
    "pilot": ("proses", C1, 2.95, 2.60, 0.55, "Pilot fine-tuning per size\n(Qwen3.5 0.8B, 2B, 4B; 250 steps)"),
    "cek": ("keputusan", C1, 3.85, 2.20, 0.85, "Pilot passes\nall four checks?"),
    "latih": ("proses", C1, 4.80, 2.60, 0.55, "Full LoRA fine-tuning (bf16, 3 epochs,\nearly stopping, best validation loss)"),
    "ekspor": ("proses", C1, 5.55, 2.60, 0.55, "Export: merge adapters → 16-bit model;\nGGUF Q4_K_M and Q8_0"),
    "A1": ("konektor", C1, 6.22, 0.36, 0.36, "A"),
    "A2": ("konektor", C2, 0.35, 0.36, 0.36, "A"),
    "tier1": ("proses", C2, 1.02, 2.80, 0.62, "Tier 1: token-F1, ROUGE-L, BERTScore;\ninter-physician agreement; ablations\n(RQ1, RQ2)"),
    "tier2": ("proses", C2, 1.82, 2.80, 0.62, "Tier 2: blind paired LLM judge;\nred-flag escalation analysis\n(RQ4)"),
    "device": ("proses", C2, 2.62, 2.80, 0.62, "On-device test: Q4_K_M under llama.cpp\non a laptop and an Android phone\n(RQ3)"),
    "gerbang": ("keputusan", C2, 3.60, 2.50, 0.90, "Peak RAM within the\nbudget of the device class?"),
    "layak": ("proses", 4.15, 4.62, 1.35, 0.50, "Size in the\nfeasible set"),
    "tidak": ("proses", 5.60, 4.62, 1.30, 0.50, "Size excluded\nfor that class"),
    "banding": ("proses", C2, 5.45, 2.80, 0.55, "Compare quality, safety and device\ncost across the three sizes"),
    "selesai": ("terminal", C2, 6.22, 1.30, 0.40, "End"),
}


def tepi(i, sisi):
    _, cx, cy, w, h, _ = SIMPUL[i]
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
def gambar_png() -> pathlib.Path:
    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                   # y dari atas, sama dengan draw.io
    ax.axis("off")
    lw = 0.9

    for i, (jenis, cx, cy, w, h, teks) in SIMPUL.items():
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

    for dari, sd, ke, sk, belok, label, pos, ujung in PANAH:
        titik = [tepi(dari, sd)] + belok + [tepi(ke, sk)]
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
    p = OUT / "fig_p6_flowchart.png"
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


if __name__ == "__main__":
    for p in (gambar_png(), tulis_drawio()):
        print(f"tersimpan: {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
