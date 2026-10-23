"""
plot_hasil_p6.py — grafik hasil Pivot 6 untuk laporan & naskah.

Membaca angka LANGSUNG dari results/ (bukan hard-code) supaya grafik tidak pernah
menyimpang dari hasil eksperimen:

  - RQ1/RQ2  <- results/eval_truncation.json  (blok `per_ukuran.<size>.semua`)
  - Tier 2   <- results/llm_judge/tier2_<size>_ringkas.json
  - RQ3      <- results_bench/bench_<size>.json (laptop) & results_bench_hp/ (HP)

Keluaran:
  fig_p6_rq1_rq2.png   mutu jawaban: baseline vs fine-tuned, 3 ukuran, + plafon dokter
  fig_p6_tier2_aman.png  perubahan rate "Aman" — pembalikan tanda lintas ukuran
  fig_p6_rq3.png         puncak RAM (+ anggaran kelas HP) dan latensi, laptop vs HP

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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
# "skripsi" = teks Indonesia tanpa judul/catatan di dalam gambar (keterangan ada di caption LaTeX).
OUT = {"id": RES / "figures", "en": ROOT / "artikel" / "figures",
       "skripsi": ROOT / "skripsi" / "latex" / "gambar"}

TEKS = {
    "id": {
        "baseline": "Baseline (pra-latih)",
        "plafon": "Plafon kesepakatan dokter ({:.3f})",
        "skor": "skor",
        "delta_aman": "Δ rate “Aman” (poin persentase)",
        "laptop": "Laptop CPU (12 thread)",
        "hp": "HP POCO X3 GT (4 thread)",
        "ram": "Puncak RAM saat generate (GB)",
        "lat": "Latensi per jawaban (detik)",
        "judul_ram": "Memori",
        "judul_lat": "Kecepatan",
        "anggaran": "anggaran HP {} GB",
        "token3": "HP, pemblokir 3-gram (belum diulang DRY+)",
    },
    "en": {
        "baseline": "Baseline",
        "plafon": "Inter-physician agreement ({:.3f})",
        "skor": "Score",
        "delta_aman": "Change in Safe rate (pp)",
        "laptop": "Laptop CPU (12 threads)",
        "hp": "Android smartphone (4 threads)",
        "ram": "Peak RAM during generation (GB)",
        "lat": "Latency per answer (s)",
        "judul_ram": "Memory",
        "judul_lat": "Speed",
        "anggaran": "{} GB smartphone budget",
        "token3": "Smartphone, 3-gram block",
    },
}

# --- palet (dataviz: slot kategorikal 1+2, diverging blue<->red; sudah divalidasi) ---
BASELINE = "#2a78d6"   # biru
FINETUNED = "#eb6834"  # oranye
POS = "#2a78d6"        # delta positif
NEG = "#e34948"        # delta negatif
# Perangkat memakai slot 3 & 7, BUKAN 1 & 2: biru/oranye di naskah sudah berarti
# baseline/fine-tuned. Pasangan ini divalidasi (CVD ΔE 31, normal ΔE 36); aqua di
# bawah 3:1 terhadap surface -> label nilai wajib ada di tiap batang.
LAPTOP = "#1baf7a"     # aqua
HP = "#4a3aa7"         # violet
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"

UKURAN = ["0.8b", "2b", "4b"]
LABEL_UKURAN = ["0.8B", "2B", "4B"]

# Kanvas versi "en" = ukuran cetak di naskah RESTI (buat_kerangka_resti_docx.py: fig:rq1
# 13,5 cm, fig:tier2 7,8 cm), jadi ukuran huruf di sini = ukuran huruf tercetak (7-8 pt).
# Kanvas lama (8,4 dan 5,6 in) diperkecil Word 0,55-0,63x -> label tercetak ~5 pt.
# fig_p6_rq3 tidak dipakai di docx RESTI, jadi kanvasnya dibiarkan.
KANVAS_EN = {"rq1": (5.31, 2.35), "tier2": (3.07, 1.85), "rq3": (8.0, 3.6)}

# Plafon kesepakatan antar-dokter. Dihitung ulang oleh plafon_manusia.py pada 1093
# pasangan jawaban dokter berbeda (satu pasangan per pertanyaan, jawaban identik
# dibuang) -> results/plafon_manusia.json. Angka lama 0,285/0,145 berasal dari
# prosedur yang tidak tersimpan sebagai skrip dan tidak dapat direproduksi.
PLAFON = {"f1": 0.2972, "rougeL": 0.1496, "bertscore": 0.6867}


def gaya(ax, labelsize=11):
    """Kromium resesif: grid horizontal tipis, tanpa bingkai kanan/atas."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sisi in ("top", "right", "left"):
        ax.spines[sisi].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=labelsize)


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

    en = lang == "en"
    ukuran = KANVAS_EN["rq1"] if en else (5.5 * len(panel), 4.8)
    fig, axes = plt.subplots(1, len(panel), figsize=ukuran, facecolor=SURFACE)
    x = range(len(UKURAN))
    lebar = 0.36

    for ax, (kunci, judul) in zip(axes, panel):
        gaya(ax, labelsize=7 if en else 11)
        base = [metrik[u]["baseline"][kunci] for u in UKURAN]
        ft = [metrik[u]["finetuned"][kunci] for u in UKURAN]

        # jarak 2px antar batang bersebelahan -> pakai selisih posisi, bukan batang menempel
        b1 = ax.bar([i - lebar / 2 - 0.012 for i in x], base, lebar,
                    color=BASELINE, label=TEKS[lang]["baseline"], zorder=2)
        b2 = ax.bar([i + lebar / 2 + 0.012 for i in x], ft, lebar,
                    color=FINETUNED, label="Fine-tuned", zorder=2)

        # Versi "en" (naskah) TANPA angka di tiap batang: nilainya sama persis dengan
        # Tabel 2, dan di ukuran cetak 18 label tidak muat (saling tumpuk dan tertimpa
        # garis plafon, mis. BERTScore baseline 0,658-0,665 vs garis 0,687). Pedoman
        # dataviz: jangan angka di tiap batang; sumbu + tabel membawa nilainya.
        if not en:
            halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]
            for batang in (b1, b2):
                for r in batang:
                    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004,
                            f"{r.get_height():.3f}", ha="center", va="bottom",
                            fontsize=9.5, color=INK2, path_effects=halo, zorder=4)

        # plafon manusia = garis acuan, bukan seri -> abu-abu, putus-putus.
        # Keterangannya masuk LEGENDA, bukan teks melayang: teks inline bertabrakan
        # dengan label nilai batang.
        pegangan = [b1, b2]
        batas_atas = max(ft)
        if kunci in PLAFON:
            pegangan.append(ax.axhline(
                PLAFON[kunci], color=INK2, linewidth=1.0 if en else 1.4, linestyle=(0, (5, 3)),
                zorder=3, label=TEKS[lang]["plafon"].format(PLAFON[kunci])))
            batas_atas = max(batas_atas, PLAFON[kunci])

        ax.set_xticks(list(x))
        ax.set_xticklabels(LABEL_UKURAN)
        ax.set_title(judul, fontsize=8 if en else 13, color=INK, pad=4 if en else 10,
                     loc="left", fontweight="bold")
        if en:
            ax.set_ylim(0, batas_atas * 1.12)
        else:
            ax.set_ylim(0, batas_atas * 1.52)
            ax.legend(handles=pegangan, frameon=False, fontsize=10,
                      labelcolor=INK2, loc="upper left", handlelength=1.8)

    axes[0].set_ylabel(TEKS[lang]["skor"], fontsize=7 if en else 11, color=INK2)

    if lang == "id":
        fig.suptitle("Mutu jawaban: fine-tuning menaikkan skor di ketiga ukuran (n = 2.998)",
                     fontsize=14.5, color=INK, x=0.008, ha="left", fontweight="bold")
        fig.text(0.008, 0.015,
                 "Model fine-tuned mencapai plafon kesepakatan antar-dokter (n = 1.093 pasangan); "
                 "hanya 4B yang melampauinya pada token-F1.",
                 fontsize=9.5, color=INK2)
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    else:
        garis = mlines.Line2D([], [], color=INK2, linewidth=1.0, linestyle=(0, (5, 3)),
                              label="Inter-physician agreement")
        fig.legend(handles=[b1, b2, garis], frameon=False, fontsize=7, labelcolor=INK2,
                   loc="upper center", ncol=3, handlelength=1.8, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.4, w_pad=1.2)

    p = OUT[lang] / "fig_p6_rq1_rq2.png"
    fig.savefig(p, dpi=300 if lang == "en" else 200, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_tier2(t2: dict, lang: str) -> pathlib.Path:
    """Satu seri bertanda -> warna diverging (biru naik / merah turun), tanpa legenda."""
    en = lang == "en"
    fig, ax = plt.subplots(figsize=KANVAS_EN["tier2"] if en else (8.2, 5.0),
                           facecolor=SURFACE)
    gaya(ax, labelsize=7 if en else 11)

    delta = [t2[u]["ft"] - t2[u]["base"] for u in UKURAN]
    warna = [POS if d >= 0 else NEG for d in delta]
    x = range(len(UKURAN))

    batang = ax.bar(x, delta, 0.42, color=warna, zorder=2)
    ax.axhline(0, color=INK2, linewidth=0.9 if en else 1.2, zorder=3)

    # Jarak label dalam POIN (bukan satuan data), supaya tetap pas di kanvas mana pun.
    fs_d, fs_s = (8, 6.5) if en else (13, 9.5)
    for i, (r, u) in enumerate(zip(batang, UKURAN)):
        h = r.get_height()
        arah = 1 if h >= 0 else -1
        va = "bottom" if arah > 0 else "top"
        # tanda minus tipografis (−), sama dengan sumbu dan teks naskah
        ax.annotate(f"{h:+.1f} pp".replace("-", "−"), (i, h), xytext=(0, arah * 3),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=fs_d, color=INK, fontweight="bold")
        ax.annotate(f"{t2[u]['base']:.1f}% → {t2[u]['ft']:.1f}%", (i, h),
                    xytext=(0, arah * (3 + fs_d + 2)), textcoords="offset points",
                    ha="center", va=va, fontsize=fs_s, color=INK2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(LABEL_UKURAN, fontsize=7.5 if en else 12)
    ax.set_ylabel(TEKS[lang]["delta_aman"], fontsize=7 if en else 11, color=INK2)
    ax.set_ylim(min(delta) - 15, max(delta) + 15)
    if en:
        # ruang kiri-kanan supaya label "93.5% → 76.0%" tidak menempel di tepi kanvas
        ax.set_xlim(-0.65, len(UKURAN) - 0.35)

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
        fig.tight_layout(pad=0.4)

    p = OUT[lang] / "fig_p6_tier2_aman.png"
    fig.savefig(p, dpi=300 if lang == "en" else 200, facecolor=SURFACE)
    plt.close(fig)
    return p


def muat_bench() -> dict:
    """Ringkasan benchmark on-device. `token3` = run yang masih memakai pemblokir
    3-gram (kunci `decoding` tidak ada) — latensinya ditandai arsir di grafik,
    dan tanda itu hilang sendiri begitu run DRY+ disalin ke results_bench_hp/."""
    out = {}
    for perangkat, folder in (("laptop", "results_bench"), ("hp", "results_bench_hp")):
        out[perangkat] = {}
        for u in UKURAN:
            d = json.loads((ROOT / folder / f"bench_{u}.json").read_text(encoding="utf-8"))
            out[perangkat][u] = {
                "ram": d["ram_GB"]["puncak_infer"],
                "lat": d["waktu"]["latensi_s_mean"],
                "tps": d["waktu"]["tok_per_s_mean"],
                "token3": d["waktu"].get("decoding") != "dry_plus",
                "anggaran": {int(float(k.split()[1])): v["diasumsikan_tersedia_GB"]
                             for k, v in d["gerbang_ram"].items()},
            }
    return out


def fig_rq3(b: dict, lang: str) -> pathlib.Path:
    """Dua panel, bukan dua sumbu-y: GB dan detik tidak boleh berbagi skala."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=KANVAS_EN["rq3"] if lang == "en" else (12.5, 5.0),
                                   facecolor=SURFACE)
    x = range(len(UKURAN))
    lebar = 0.36
    geser = lebar / 2 + 0.012
    halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]

    def batang(ax, kunci, fmt):
        pegangan = []
        for perangkat, warna, arah in (("laptop", LAPTOP, -1), ("hp", HP, 1)):
            nilai = [b[perangkat][u][kunci] for u in UKURAN]
            rs = ax.bar([i + arah * geser for i in x], nilai, lebar, color=warna, zorder=2)
            pegangan.append(mpatches.Patch(facecolor=warna, label=TEKS[lang][perangkat]))
            for u, r in zip(UKURAN, rs):
                if kunci == "lat" and perangkat == "hp" and b["hp"][u]["token3"]:
                    r.set_hatch("////")
                    r.set_edgecolor(SURFACE)
                    r.set_linewidth(0)
                ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                        " " + fmt(r.get_height()), ha="center", va="bottom",
                        fontsize=9.5, color=INK2, path_effects=halo, zorder=4)
        ax.set_xticks(list(x))
        ax.set_xticklabels(LABEL_UKURAN, fontsize=12)
        return pegangan

    # --- panel memori + garis anggaran kelas HP ---
    gaya(ax1)
    peg = batang(ax1, "ram", lambda v: f"{v:.2f}")
    anggaran = b["hp"]["0.8b"]["anggaran"]
    for kelas, gb in sorted(anggaran.items()):
        ax1.axhline(gb, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax1.text(len(UKURAN) - 0.58, gb, TEKS[lang]["anggaran"].format(kelas),
                 ha="left", va="center", fontsize=8.5, color=INK2,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    ax1.set_ylim(0, max(b["laptop"][u]["ram"] for u in UKURAN) * 1.15)
    ax1.set_xlim(-0.6, len(UKURAN) + 0.45)
    ax1.set_ylabel(TEKS[lang]["ram"], fontsize=11, color=INK2)
    ax1.set_title(TEKS[lang]["judul_ram"], fontsize=13, color=INK, pad=10,
                  loc="left", fontweight="bold")
    if lang != "en":
        ax1.legend(handles=peg, frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")

    # --- panel latensi ---
    gaya(ax2)
    peg = batang(ax2, "lat", lambda v: f"{v:.1f} s")
    if any(b["hp"][u]["token3"] for u in UKURAN):
        peg.append(mpatches.Patch(facecolor=HP, hatch="////", edgecolor=SURFACE,
                                  linewidth=0, label=TEKS[lang]["token3"]))
    ax2.set_ylim(0, max(b["hp"][u]["lat"] for u in UKURAN) * 1.15)
    ax2.set_ylabel(TEKS[lang]["lat"], fontsize=11, color=INK2)
    ax2.set_title(TEKS[lang]["judul_lat"], fontsize=13, color=INK, pad=10,
                  loc="left", fontweight="bold")
    if lang != "en":
        ax2.legend(handles=peg, frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")

    if lang == "id":
        fig.suptitle("Deploy offline: memori muat, kecepatan yang menyaring (Q4_K_M, n = 12 pertanyaan)",
                     fontsize=14, color=INK, x=0.008, ha="left", fontweight="bold")
        fig.text(0.008, 0.015,
                 "Garis putus-putus = memori yang diasumsikan tersedia untuk satu aplikasi "
                 "(55% RAM total). Puncak RAM di HP 23–37% lebih rendah daripada laptop.",
                 fontsize=9.5, color=INK2)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    else:
        fig.legend(handles=peg, frameon=False, fontsize=10, labelcolor=INK2,
                   loc="upper center", ncol=len(peg), bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.9])

    p = OUT[lang] / "fig_p6_rq3.png"
    fig.savefig(p, dpi=300 if lang == "en" else 200, facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["id", "en", "skripsi"], default="id")
    lang = ap.parse_args().lang
    if lang == "skripsi":
        TEKS["skripsi"] = TEKS["id"]
    OUT[lang].mkdir(parents=True, exist_ok=True)
    metrik, t2, bench = muat_metrik(), muat_tier2(), muat_bench()
    for p in (fig_rq1_rq2(metrik, lang), fig_tier2(t2, lang), fig_rq3(bench, lang)):
        print("tersimpan:", p.relative_to(ROOT), f"({p.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
