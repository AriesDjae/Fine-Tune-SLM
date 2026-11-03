"""
r2_figure2.py — Figure 2 naskah dengan baseline setelah clean_greeting (R2-07a).

Gaya dan kode gambar = results/plot_hasil_p6.fig_rq1_rq2 (versi "en"), tidak disalin.
Satu-satunya beda: token-F1/ROUGE-L baseline dari r2_clean_greeting.json, bukan dari
results/eval_truncation.json (yang dihitung tanpa clean_greeting). BERTScore tetap dari
results/bertscore_p6.json. Gambar lama artikel/figures/fig_p6_rq1_rq2.png tidak disentuh.

    python analisis_revisi_resti/r2_figure2.py
"""
import json
import pathlib
import sys

DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DIR.parent / "results"))
import plot_hasil_p6 as P  # noqa: E402

BARU = DIR / "fig_p6_rq1_rq2_clean_greeting.png"
tabel = json.loads((DIR / "r2_clean_greeting.json").read_text(encoding="utf-8"))["R2_06_table2"]

metrik = P.muat_metrik()
for u in P.UKURAN:
    for m in ("f1", "rougeL"):
        metrik[u]["baseline"][m] = tabel[f"{u}|{m}"]["baseline"]
        metrik[u]["finetuned"][m] = tabel[f"{u}|{m}"]["finetuned"]

P.OUT["en"] = DIR / "_tmp_fig"
P.OUT["en"].mkdir(exist_ok=True)
sementara = P.fig_rq1_rq2(metrik, "en")
sementara.replace(BARU)
P.OUT["en"].rmdir()
print(BARU, {u: {m: metrik[u]["baseline"][m] for m in ("f1", "rougeL", "bertscore")} for u in P.UKURAN})
