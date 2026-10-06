"""Slide progres paper (16:9) — satu big task, 12 small task, bar per task.

Render: python presentasi/progress_paper.py presentasi/progress_paper_YYYYMMDD.png
Task "Menyunting naskah" dihitung otomatis dari artikel/TemplateRESTI2026_rev.docx (kata prosa
yang tidak ada di template asli versi git HEAD) dibagi sasaran RESTI ±5.500 kata.
"""
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
DOCX = ROOT / "artikel" / "TemplateRESTI2026_rev.docx"  # file kerja penulis
SASARAN_KATA = 5500
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]


def kata_naskah():
    """Kata prosa yang ditulis penulis di TemplateRESTI2026_rev.docx: paragraf badan naskah
    yang tidak ada di versi template asli (git HEAD), tanpa judul bab dan caption.
    Mengembalikan (kata, paragraf instruksi template yang belum dihapus)."""
    import io
    import subprocess
    import docx
    asli = subprocess.run(["git", "show", "HEAD:artikel/TemplateRESTI2026_rev.docx"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    teks_asli = {p.text.strip() for p in docx.Document(io.BytesIO(asli)).paragraphs}
    mulai, kata, sisa = False, 0, 0
    for p in docx.Document(DOCX).paragraphs:
        t = p.text.strip()
        if t.endswith("Introduction"):
            mulai = True
        if t.startswith("References"):
            break
        if not mulai or not t or p.style.name.lower().startswith(("heading", "caption")):
            continue
        if t in teks_asli:
            sisa += len(t.split()) > 8
            continue
        kata += len(t.split())
    return kata, sisa


kata, penanda = kata_naskah()
p_sunting = min(100, round(100 * kata / SASARAN_KATA))

# (task, progres %, keterangan di kanan)
TASKS = [
    ("Menentukan jurnal sasaran", 100, "Selesai · RESTI (SINTA 2, Scopus Q3)"),
    ("Menyelesaikan eksperimen & analisis hasil", 100, "Selesai · RQ1–RQ4; angka dicek ulang"),
    ("Mengumpulkan & memverifikasi referensi", 100, "Selesai · 38 dikutip, 34 ber-DOI"),
    ("Menulis draf lengkap Abstract–Conclusion", 100, "Selesai · versi RESTI ±5.900 kata"),
    ("Membuat gambar & tabel", 100, "Selesai · 5 tabel, 3 gambar"),
    ("Merevisi gaya bahasa (masukan dosen)", 100, "Selesai"),
    ("Menyiapkan arena validasi dokter", 100, "Selesai · 100 pasangan jawaban"),
    ("Menyunting naskah di template RESTI", p_sunting,
     f"{kata:,} / ±{SASARAN_KATA:,} kata".replace(",", ".") + f" · {penanda} paragraf instruksi tersisa"),
    ("Mengecek kalimat sitasi ke sumber asli", 12, "5 / 43 kalimat"),
    ("Mengumpulkan penilaian dokter", 25, "25 / 100 pasangan · 4 dokter"),
    ("Submit ke RESTI", 0, "Belum"),
    ("Mengunggah preprint arXiv", 0, "Belum"),
]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
TRACK, FILL = "#e9e8e2", "#2a78d6"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

n = len(TASKS)
selesai = sum(1 for _, p, _ in TASKS if p == 100)
persen = round(100 * selesai / n)
hari = datetime.date.today()
tanggal = f"{hari.day} {BULAN[hari.month - 1]} {hari.year}"

fig = plt.figure(figsize=(13.333, 7.5), facecolor=SURFACE)

# Kepala slide
fig.text(0.045, 0.905, "Progress Report: Paper", fontsize=30, fontweight="bold", color=INK)
fig.text(0.045, 0.855,
         f"Big task: naskah artikel jurnal untuk RESTI (SINTA 2, Scopus Q3)  ·  per {tanggal}",
         fontsize=14, color=INK2)
fig.text(0.955, 0.875, f"{persen}%", fontsize=54, fontweight="bold", color=INK,
         ha="right", va="center")
fig.text(0.825, 0.892, f"{selesai} dari {n}", fontsize=15, fontweight="bold",
         color=INK, ha="right", va="center")
fig.text(0.825, 0.858, "small task selesai", fontsize=13, color=INK2,
         ha="right", va="center")

# Bar per task
ax = fig.add_axes([0.375, 0.07, 0.36, 0.73], facecolor=SURFACE)
y = list(range(n))[::-1]
h = 0.46
ax.barh(y, [100] * n, height=h, color=TRACK, linewidth=0)
ax.barh(y, [p for _, p, _ in TASKS], height=h, color=FILL, linewidth=0)

for yi, (nama, p, ket) in zip(y, TASKS):
    ax.text(-3, yi, nama, ha="right", va="center", fontsize=14, color=INK)
    ax.text(103, yi, ket, ha="left", va="center", fontsize=12.5,
            color=INK2 if p == 100 else INK,
            fontweight="normal" if p == 100 else "bold")

ax.set_xlim(0, 100)
ax.set_ylim(-0.6, n - 0.4)
ax.set_xticks([0, 50, 100])
ax.set_xticklabels(["0%", "50%", "100%"], fontsize=11, color=MUTED)
ax.set_yticks([])
ax.tick_params(axis="x", length=0, pad=4)
for s in ax.spines.values():
    s.set_visible(False)

fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(OUT, f"{selesai}/{n} = {persen}%  ·  naskah {kata} kata, {penanda} paragraf instruksi")
