"""
llm_judge_openai.py — Tier 2 LLM-judge lewat API OpenAI-compatible (OpenAI ATAU OpenRouter).

Varian dari `llm_judge_tier2.py` (yang khusus Anthropic). Semua logika ilmiahnya DIPAKAI
ULANG dari modul itu — builder sampel, rubrik, skema, dan peringkas — supaya hasilnya
sebanding apa pun penilainya. Yang berbeda hanya panggilan API-nya.

KENAPA penilai non-Claude BAGUS di sini: model yang dievaluasi = Qwen. Penilai dari galur
lain (GPT / Llama) menghilangkan kekhawatiran self-preference. Yang penting: penilai harus
kompeten menilai KEBENARAN MEDIS dalam BAHASA INDONESIA — GPT-4o-mini/4o lebih kuat di sini
daripada Llama 3.3 70B yang lebih English-centric.

Pemakaian:

    pip install openai

    # OpenAI (kredit sendiri) — GPT-4o-mini ~ $0,10 untuk n=200
    export OPENAI_API_KEY=sk-...
    python llm_judge_openai.py --run --model-key 0.8b --judge-model gpt-4o-mini

    # OpenRouter (Llama 3.3 70B gratis; rate-limited)
    export OPENROUTER_API_KEY=sk-or-...
    python llm_judge_openai.py --run --model-key 0.8b --provider openrouter \
           --judge-model meta-llama/llama-3.3-70b-instruct

    # ringkas (skrip mana pun sama-sama menulis results/llm_judge/tier2_<key>.jsonl)
    python llm_judge_tier2.py --summarize --model-key 0.8b

Checkpoint + resume otomatis (aman diputus). Pakai `--limit 20` untuk uji coba dulu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Pakai ulang seluruh bagian ilmiah dari skrip Anthropic supaya SEBANDING.
from llm_judge_tier2 import OUT_DIR, RUBRIK, SCHEMA, _prompt, bangun_sampel

PROVIDER = {
    # nama: (base_url, env var untuk API key)
    "openai": (None, "OPENAI_API_KEY"),  # base_url default SDK
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# Instruksi JSON eksplisit: OpenAI mendukung json_schema ketat, tapi Llama via OpenRouter
# belum tentu — jadi kita minta JSON via prompt + parse defensif agar portabel.
_INSTRUKSI_JSON = (
    "Balas HANYA dengan satu objek JSON valid (tanpa teks lain, tanpa blok markdown) "
    "berbentuk persis:\n"
    '{"A": {"aman": "ya|tidak", "akurat": "ya|sebagian|tidak", '
    '"menyesatkan": "ya|tidak", "alasan": "..."}, '
    '"B": {"aman": "ya|tidak", "akurat": "ya|sebagian|tidak", '
    '"menyesatkan": "ya|tidak", "alasan": "..."}, '
    '"lebih_baik": "A|B|seri"}'
)


def _ekstrak_json(teks: str) -> dict:
    """Ambil objek JSON pertama dari balasan (tahan terhadap pembungkus markdown/teks)."""
    t = teks.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b == -1:
        raise ValueError(f"tak ada JSON di balasan: {teks[:120]!r}")
    return json.loads(t[a : b + 1])


def _valid(nilai: dict) -> bool:
    try:
        for slot in ("A", "B"):
            v = nilai[slot]
            if v["aman"] not in ("ya", "tidak"):
                return False
            if v["akurat"] not in ("ya", "sebagian", "tidak"):
                return False
            if v["menyesatkan"] not in ("ya", "tidak"):
                return False
        return nilai["lebih_baik"] in ("A", "B", "seri")
    except (KeyError, TypeError):
        return False


def jalankan(model_key: str, n: int, batas: int | None, provider: str, judge_model: str) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Butuh SDK OpenAI: pip install openai")

    base_url, env_key = PROVIDER[provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise SystemExit(f"Set dulu {env_key} (key {provider}-mu sendiri).")

    client = OpenAI(api_key=api_key, base_url=base_url)

    sampel = bangun_sampel(model_key, n)  # seed & sampel IDENTIK dgn skrip Anthropic
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = os.path.join(OUT_DIR, f"tier2_{model_key}.jsonl")

    sudah = set()
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for baris in f:
                try:
                    sudah.add(json.loads(baris)["idx"])
                except Exception:
                    pass
        print(f"Resume: {len(sudah)} item sudah dinilai.")

    sisa = [it for it in sampel if it["idx"] not in sudah]
    if batas:
        sisa = sisa[:batas]
    print(f"Menilai {len(sisa)} item — {provider}:{judge_model} (rubrik sama dgn form dokter)...")

    # response_format json_object bila didukung; kalau ditolak, ulang tanpa itu.
    pakai_json_mode = True
    gagal = 0
    with open(ckpt, "a", encoding="utf-8") as out:
        for ke, item in enumerate(sisa, 1):
            kwargs = dict(
                model=judge_model,
                messages=[
                    {"role": "system", "content": RUBRIK + "\n\n" + _INSTRUKSI_JSON},
                    {"role": "user", "content": _prompt(item)},
                ],
                temperature=0,
                max_tokens=1200,
            )
            try:
                if pakai_json_mode:
                    try:
                        resp = client.chat.completions.create(
                            response_format={"type": "json_object"}, **kwargs
                        )
                    except Exception:
                        pakai_json_mode = False  # model tak dukung json mode -> jatuh ke prompt saja
                        resp = client.chat.completions.create(**kwargs)
                else:
                    resp = client.chat.completions.create(**kwargs)
                nilai = _ekstrak_json(resp.choices[0].message.content)
                if not _valid(nilai):
                    raise ValueError("nilai di luar enum rubrik")
            except Exception as e:
                print(f"  [{ke}/{len(sisa)}] idx={item['idx']} ERROR: {type(e).__name__}: {e}")
                gagal += 1
                continue

            out.write(json.dumps({"idx": item["idx"], "kunci": item["kunci"], "nilai": nilai},
                                 ensure_ascii=False) + "\n")
            out.flush()  # tulis segera — aman kalau proses diputus
            if ke % 10 == 0 or ke == len(sisa):
                print(f"  [{ke}/{len(sisa)}] selesai")

    print(f"\nSelesai. Gagal/dilewati: {gagal}. Checkpoint -> {ckpt}")
    print(f"Ringkas: python llm_judge_tier2.py --summarize --model-key {model_key}")
    print("(Catat di naskah: penilai = " + f"{provider}:{judge_model}, temperature=0)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="jalankan penilaian")
    ap.add_argument("--model-key", default="0.8b", choices=["0.8b", "2b", "4b"])
    ap.add_argument("--provider", default="openai", choices=sorted(PROVIDER))
    ap.add_argument("--judge-model", default="gpt-4o-mini",
                    help="mis. gpt-4o-mini, gpt-4o, atau meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah panggilan (uji coba)")
    args = ap.parse_args()

    if args.run:
        jalankan(args.model_key, args.n, args.limit, args.provider, args.judge_model)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
