"""
plot_loss_p6.py — kurva loss pelatihan ketiga model Pivot 6 (0.8B / 2B / 4B).

Sumber: log_history.json tiap run di HF `AriesDjaenuri/p6-hasil` (terbaca tanpa token),
disalin ke results/loss_p6/ supaya grafik bisa dibuat ulang tanpa jaringan.

    python results/plot_loss_p6.py                 # -> skripsi/latex/gambar/ + results/figures/
    python results/plot_loss_p6.py --lang en       # bahasa Inggris -> artikel/figures/ (naskah)

Sumbu-x memakai epoch, bukan step: 2B dan 4B dihentikan EarlyStopping pada 2,3 epoch,
dan ketidaksetaraan anggaran epoch itu justru yang harus terlihat di grafik.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_hasil_p6 import GRID, HP, INK, INK2, SURFACE, gaya

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "loss_p6"
OUT = [ROOT / "skripsi" / "latex" / "gambar", ROOT / "results" / "figures"]
UKURAN = [("0.8b", "0,8B"), ("2b", "2B"), ("4b", "4B")]
LATIH = "#a9a7a1"   # abu: loss latih = konteks, bukan fokus
TEKS = {
    "id": {"latih": "Loss latih", "val": "Loss validasi", "terbaik": "terbaik", "henti": "epoch 2,3",
           "ukuran": ["0,8B", "2B", "4B"]},
    "en": {"latih": "Training loss", "val": "Validation loss", "terbaik": "best", "henti": "epoch 2.3",
           "ukuran": ["0.8B", "2B", "4B"]},
}


def angka(x: float, lang: str) -> str:
    return f"{x:.4f}".replace(".", ",") if lang == "id" else f"{x:.4f}"


def ribuan(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", ".") if lang == "id" else f"{n:,}"


def muat(u: str) -> list[dict]:
    p = LOG / f"log_history_{u}.json"
    if not p.exists():
        from huggingface_hub import hf_hub_download
        sumber = hf_hub_download("AriesDjaenuri/p6-hasil", f"qwen35_{u}_p6/results/log_history.json")
        LOG.mkdir(parents=True, exist_ok=True)
        p.write_text(pathlib.Path(sumber).read_text(encoding="utf-8"), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    lang = "en" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en" else "id"
    T = TEKS[lang]
    out = [ROOT / "artikel" / "figures"] if lang == "en" else OUT
    # naskah: kanvas ~ukuran cetak (6.5 in) supaya huruf tidak tercetak ~4 pt
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3) if lang == "en" else (15, 4.8),
                             sharey=True, facecolor=SURFACE)
    for ax, (u, _), nama in zip(axes, UKURAN, T["ukuran"]):
        log = muat(u)
        latih = [(e["epoch"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
        val = [(e["epoch"], e["eval_loss"], e["step"]) for e in log if "eval_loss" in e]
        terbaik = min(val, key=lambda x: x[1])

        gaya(ax)
        ax.plot(*zip(*latih), color=LATIH, linewidth=1.0, label=T["latih"], zorder=2)
        ax.plot([v[0] for v in val], [v[1] for v in val], color=HP, linewidth=2.2,
                marker="o", markersize=3.5, label=T["val"], zorder=3)
        ax.scatter([terbaik[0]], [terbaik[1]], s=120, facecolors="none", edgecolors=INK,
                   linewidths=1.6, zorder=4)
        # skripsi: desimal koma & ribuan titik; naskah Inggris: titik & koma
        teks = f"{T['terbaik']} {angka(terbaik[1], lang)}\n(step {ribuan(terbaik[2], lang)})"
        ax.annotate(teks,
                    (terbaik[0], terbaik[1]), textcoords="offset points",
                    # naskah (kanvas kecil): teks naik di atas kurva supaya tak tertimpa garis
                    xytext=(4, 30) if lang == "en" else (-8, 16), va="bottom" if lang == "en" else "baseline",
                    ha="right", fontsize=9.5, color=INK)
        akhir = latih[-1][0]
        if akhir < 2.9:
            ax.axvline(akhir, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.text(akhir, 2.62, " EarlyStopping\n " + T["henti"],
                    fontsize=9, color=INK2, va="top", ha="left")
        ax.set_xlim(0, 3.05)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Epoch", fontsize=11, color=INK2)
        ax.set_title(nama, fontsize=13, color=INK, pad=8, loc="left", fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_ylabel("Loss (cross-entropy)", fontsize=11, color=INK2)
    axes[0].legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    for d in out:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "fig_p6_loss.png"
        fig.savefig(p, dpi=300 if lang == "en" else 200, facecolor=SURFACE)
        print(f"tersimpan: {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
