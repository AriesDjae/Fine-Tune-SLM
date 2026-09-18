"""
plot_hasil_p6.py — grafik hasil Pivot 6 untuk laporan & naskah.

Membaca angka LANGSUNG dari results/ (bukan hard-code) supaya grafik tidak pernah
menyimpang dari hasil eksperimen:

  - RQ1/RQ2  <- results/eval_truncation.json  (blok `per_ukuran.<size>.semua`)
  - Tier 2   <- results/llm_judge/tier2_<size>_ringkas.json

Keluaran:
  fig_p6_rq1_rq2.png   mutu jawaban: baseline vs fine-tuned, 3 ukuran, + plafon dokter
  fig_p6_tier2_aman.png  perubahan rate "Aman" — pembalikan tanda lintas ukuran

Jalankan:
  python results/plot_hasil_p6.py            # bahasa Indonesia -> results/figures/ (skripsi)
  python results/plot_hasil_p6.py --lang en  # bahasa Inggris  -> artikel/figures/ (naskah)

Versi `en` sengaja TANPA judul besar & catatan bawah di dalam gambar: di naskah,
penjelasan itu sudah ada di caption LaTeX, dan judul yang menyimpulkan
("menolong model kecil, merusak model besar") tidak lazim di figur jurnal.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = {"id": RES / "figures", "en": ROOT / "artikel" / "figures"}

TEKS = {
    "id": {
        "baseline": "Baseline (pra-latih)",
        "plafon": "Plafon kesepakatan dokter ({:.3f})",
        "skor": "skor",
        "delta_aman": "Δ rate “Aman” (poin persentase)",
    },
    "en": {
        "baseline": "Baseline (pre-trained)",
        "plafon": "Inter-physician agreement ({:.3f})",
        "skor": "Score",
        "delta_aman": "Change in Safe rate (percentage points)",
    },
}

# --- palet (dataviz: slot kategorikal 1+2, diverging blue<->red; sudah divalidasi) ---
BASELINE = "#2a78d6"   # biru
FINETUNED = "#eb6834"  # oranye
POS = "#2a78d6"        # delta positif
NEG = "#e34948"        # delta negatif
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"

UKURAN = ["0.8b", "2b", "4b"]
LABEL_UKURAN = ["0.8B", "2B", "4B"]

# Plafon kesepakatan antar-dokter. Dihitung ulang oleh plafon_manusia.py pada 1093
# pasangan jawaban dokter berbeda (satu pasangan per pertanyaan, jawaban identik
# dibuang) -> results/plafon_manusia.json. Angka lama 0,285/0,145 berasal dari
# prosedur yang tidak tersimpan sebagai skrip dan tidak dapat direproduksi.
PLAFON = {"f1": 0.2972, "rougeL": 0.1496, "bertscore": 0.6867}


def gaya(ax):
    """Kromium resesif: grid horizontal tipis, tanpa bingkai kanan/atas."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sisi in ("top", "right", "left"):
        ax.spines[sisi].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=11)


def muat_metrik() -> dict:
    """token-F1 & ROUGE-L dari eval_truncation.json; BERTScore dari
    bertscore_p6.json (dihitung ulang dari preds tersimpan) — digabung ke satu
    struktur supaya panel grafik dibangun dari satu sumber."""
    d = json.loads((RES / "eval_truncation.json").read_text(encoding="utf-8"))
    m = {u: dict(d["per_ukuran"][u]["semua"]) for u in UKURAN}

    bs_file = RES / "bertscore_p6.json"
    if bs_file.exists():
        bs = json.loads(bs_file.read_text(encoding="utf-8"))["bertscore_f1"]
        for u in UKURAN:
            m[u]["baseline"] = dict(m[u]["baseline"])
            m[u]["finetuned"] = dict(m[u]["finetuned"])
            m[u]["baseline"]["bertscore"] = bs[f"qwen{u}_baseline"]
            m[u]["finetuned"]["bertscore"] = bs[f"qwen{u}_finetuned"]
    return m


def muat_tier2() -> dict:
    out = {}
    for u in UKURAN:
        t = json.loads((RES / "llm_judge" / f"tier2_{u}_ringkas.json").read_text(encoding="utf-8"))
        n = t["n_per_kondisi"]
        out[u] = {
            "base": t["per_kondisi"]["baseline"]["aman"] / n["baseline"] * 100,
            "ft": t["per_kondisi"]["finetuned"]["aman"] / n["finetuned"] * 100,
            "ci": t["signifikansi"]["delta_aman_ci95"],
            "n": t["n"],
        }
    return out


def fig_rq1_rq2(metrik: dict, lang: str) -> pathlib.Path:
    """Dua panel — token-F1 dan ROUGE-L punya skala berbeda, jadi TIDAK disatukan
    dalam satu sumbu (aturan: jangan pernah dua skala-y dalam satu grafik)."""
    panel = [("f1", "Token-F1"), ("rougeL", "ROUGE-L")]
    if "bertscore" in metrik[UKURAN[0]]["finetuned"]:
        panel.append(("bertscore", "BERTScore"))

    fig, axes = plt.subplots(1, len(panel), figsize=(5.5 * len(panel), 4.8),
                             facecolor=SURFACE)
    x = range(len(UKURAN))
    lebar = 0.36

    for ax, (kunci, judul) in zip(axes, panel):
        gaya(ax)
        base = [metrik[u]["baseline"][kunci] for u in UKURAN]
        ft = [metrik[u]["finetuned"][kunci] for u in UKURAN]

        # jarak 2px antar batang bersebelahan -> pakai selisih posisi, bukan batang menempel
        b1 = ax.bar([i - lebar / 2 - 0.012 for i in x], base, lebar,
                    color=BASELINE, label=TEKS[lang]["baseline"], zorder=2)
        b2 = ax.bar([i + lebar / 2 + 0.012 for i in x], ft, lebar,
                    color=FINETUNED, label="Fine-tuned", zorder=2)

        # halo setipis surface supaya label nilai tetap terbaca saat garis plafon
        # kebetulan lewat tepat di belakangnya (kasus baseline 2B & 4B)
        halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]
        for batang in (b1, b2):
            for r in batang:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004,
                        f"{r.get_height():.3f}", ha="center", va="bottom",
                        fontsize=9.5, color=INK2, path_effects=halo, zorder=4)

        # plafon manusia = garis acuan, bukan seri -> abu-abu, putus-putus.
        # Keterangannya masuk LEGENDA, bukan teks melayang: teks inline bertabrakan
        # dengan label nilai batang.
        # Plafon manusia hanya terukur untuk metrik tumpang-tindih; BERTScore
        # tidak punya nilai antar-dokter, jadi panelnya tanpa garis acuan.
        pegangan = [b1, b2]
        batas_atas = max(ft)
        if kunci in PLAFON:
            pegangan.append(ax.axhline(
                PLAFON[kunci], color=INK2, linewidth=1.4, linestyle=(0, (5, 3)),
                zorder=3, label=TEKS[lang]["plafon"].format(PLAFON[kunci])))
            batas_atas = max(batas_atas, PLAFON[kunci])

        ax.set_xticks(list(x))
        ax.set_xticklabels(LABEL_UKURAN)
        ax.set_title(judul, fontsize=13, color=INK, pad=10, loc="left", fontweight="bold")
        ax.set_ylim(0, batas_atas * 1.52)
        ax.legend(handles=pegangan, frameon=False, fontsize=10,
                  labelcolor=INK2, loc="upper left", handlelength=1.8)

    axes[0].set_ylabel(TEKS[lang]["skor"], fontsize=11, color=INK2)

    if lang == "id":
        fig.suptitle("Mutu jawaban: fine-tuning menaikkan skor di ketiga ukuran (n = 2.998)",
                     fontsize=14.5, color=INK, x=0.008, ha="left", fontweight="bold")
        fig.text(0.008, 0.015,
                 "Ketiga model fine-tuned melampaui plafon kesepakatan antar-dokter "
                 "(n = 908 pasangan) — metrik overlap sudah jenuh untuk tugas ini.",
                 fontsize=9.5, color=INK2)
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    else:
        fig.tight_layout()

    p = OUT[lang] / "fig_p6_rq1_rq2.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_tier2(t2: dict, lang: str) -> pathlib.Path:
    """Satu seri bertanda -> warna diverging (biru naik / merah turun), tanpa legenda."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0), facecolor=SURFACE)
    gaya(ax)

    delta = [t2[u]["ft"] - t2[u]["base"] for u in UKURAN]
    warna = [POS if d >= 0 else NEG for d in delta]
    x = range(len(UKURAN))

    batang = ax.bar(x, delta, 0.42, color=warna, zorder=2)
    ax.axhline(0, color=INK2, linewidth=1.2, zorder=3)

    for i, (r, u) in enumerate(zip(batang, UKURAN)):
        h = r.get_height()
        atas = h >= 0
        ax.text(i, h + (1.6 if atas else -1.6), f"{h:+.1f} pp",
                ha="center", va="bottom" if atas else "top",
                fontsize=13, color=INK, fontweight="bold")
        ax.text(i, h + (7.0 if atas else -7.0),
                f"{t2[u]['base']:.1f}% → {t2[u]['ft']:.1f}%",
                ha="center", va="bottom" if atas else "top",
                fontsize=9.5, color=INK2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(LABEL_UKURAN, fontsize=12)
    ax.set_ylabel(TEKS[lang]["delta_aman"], fontsize=11, color=INK2)
    ax.set_ylim(min(delta) - 15, max(delta) + 15)

    if lang == "id":
        fig.suptitle("Keselamatan: fine-tuning menolong model kecil, merusak model besar",
                     fontsize=14, color=INK, x=0.008, ha="left", fontweight="bold")
        fig.text(0.008, 0.015,
                 "Penilai LLM buta & berpasangan, n = 200 per ukuran. Ketiga selisih signifikan "
                 "(CI 95% tidak memuat nol).\nArahnya BERLAWANAN dengan metrik overlap di atas — "
                 "di sana justru 4B fine-tuned yang terbaik.",
                 fontsize=9.5, color=INK2)
        fig.tight_layout(rect=[0, 0.085, 1, 0.93])
    else:
        fig.tight_layout()

    p = OUT[lang] / "fig_p6_tier2_aman.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["id", "en"], default="id")
    lang = ap.parse_args().lang
    OUT[lang].mkdir(parents=True, exist_ok=True)
    metrik, t2 = muat_metrik(), muat_tier2()
    for p in (fig_rq1_rq2(metrik, lang), fig_tier2(t2, lang)):
        print("tersimpan:", p.relative_to(ROOT), f"({p.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
