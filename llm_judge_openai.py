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

    # KUNCI API: jangan tempel di chat/perintah. Simpan di file `openai.key`
    # (otomatis ter-gitignore lewat pola *.key). Skrip membacanya sendiri.
    #   1) buat file openai.key berisi HANYA key-nya
    #   2) jalankan:
    python llm_judge_openai.py --run --model-key 0.8b --judge-model gpt-4o

    # uji coba dulu 20 item (murah) sebelum full 200:
    python llm_judge_openai.py --run --model-key 0.8b --judge-model gpt-4o --limit 20

    # OpenRouter (Llama 3.3 70B gratis; rate-limited) — key di file openrouter.key
    python llm_judge_openai.py --run --model-key 0.8b --provider openrouter \
           --judge-model meta-llama/llama-3.3-70b-instruct

    # ringkas (skrip mana pun sama-sama menulis results/llm_judge/tier2_<key>.jsonl)
    python llm_judge_tier2.py --summarize --model-key 0.8b

DEFAULT OBJEKTIF: referensi dokter TIDAK disodorkan ke penilai (fine-tuned lebih mirip
referensi -> menyodorkannya bisa memberi keunggulan semu). Aktifkan --show-reference
hanya bila ingin membandingkan. Checkpoint + resume otomatis (aman diputus).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Pakai ulang seluruh bagian ilmiah dari skrip Anthropic supaya SEBANDING.
from llm_judge_tier2 import (OUT_DIR, REF_NOTE, RUBRIK, SCHEMA, _prompt, baca_kunci,
                             bangun_sampel, ckpt_path)

PROVIDER = {
    # nama: (base_url, env var, nama file kunci, prefiks kunci yang sah)
    "openai": (None, "OPENAI_API_KEY", "openai.key", ("sk-",)),  # base_url default SDK
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter.key", ("sk-",)),
    # Google AI Studio menyediakan endpoint OpenAI-compatible, jadi seluruh logika di
    # bawah dipakai apa adanya. Kunci gratisnya dari aistudio.google.com/apikey (tanpa
    # billing, tanpa Google Cloud) dan berawalan 'AIza', bukan 'sk-'.
    # KENAPA GEMINI: model yang dievaluasi = Qwen, penilai utama = GPT-4o. Gemini berasal
    # dari lab KETIGA, sehingga kesepakatan Gemini<->GPT-4o adalah bukti lintas-galur —
    # jauh lebih sulit dibantah daripada dua model satu keluarga.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",
               "GEMINI_API_KEY", "gemini.key", ("AIza",)),
}

# Tier gratis Gemini membatasi permintaan per menit; beri jeda default agar tak terus
# menabrak 429. Penyedia berbayar tidak perlu jeda.
JEDA_DEFAULT = {"gemini": 4.0}


# Pembaca kunci dipakai bersama dengan jalur Anthropic (didefinisikan di llm_judge_tier2).
# Alias dipertahankan supaya pemanggil lama (judge_run.py) tetap jalan.
_api_key = baca_kunci

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


def _panggil(client, kwargs: dict, state: dict):
    """Panggil chat completion; pakai json_object mode bila model mendukungnya.

    Hanya MATIKAN json mode secara permanen kalau errornya jelas soal `response_format`
    (mis. Llama via OpenRouter tak dukung) — error lain (jaringan/rate-limit) DILEMPAR
    ulang supaya ditangani sbg kegagalan per-item, tidak meracuni sisa run.
    """
    if state["json"]:
        try:
            return client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "response_format" in msg or "json_object" in msg or "json mode" in msg:
                state["json"] = False
                print("  (model tak dukung json mode -> lanjut via instruksi prompt saja)")
                return client.chat.completions.create(**kwargs)
            raise
    return client.chat.completions.create(**kwargs)


def _kuota_habis(e: Exception) -> bool:
    """Bedakan 429 SEMENTARA (rate limit, layak diulang) dari 429 PERMANEN (kuota habis).

    Ini bukan detail kosmetik: mengulang-ulang 'insufficient_quota' dengan backoff akan
    menggantung run berjam-jam tanpa hasil — persis jebakan yang menghabiskan waktu saat
    kredit OpenAI habis di tengah ablation.
    """
    m = str(e).lower()
    return "insufficient_quota" in m or "exceeded your current quota" in m or "billing" in m


def _batas_laju(e: Exception) -> bool:
    m = str(e).lower()
    return ("rate limit" in m or "rate_limit" in m or "429" in m
            or "resource_exhausted" in m) and not _kuota_habis(e)


def jalankan(model_key: str, n: int, batas: int | None, provider: str, judge_model: str,
             show_reference: bool, max_tokens: int, strip: bool = False, tag: str = "",
             jeda: float | None = None) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Butuh SDK OpenAI: pip install openai")

    base_url, env_key, key_file, prefiks = PROVIDER[provider]
    api_key = _api_key(env_key, key_file, prefiks)
    if not api_key:
        raise SystemExit(
            f"Kunci {provider} tak ditemukan.\n"
            f"Simpan kunci di file '{key_file}' (ter-gitignore), atau set env {env_key}.\n"
            f"Kunci {provider} berawalan {' / '.join(prefiks)}.\n"
            f"JANGAN tempel kunci di perintah/chat."
        )
    if jeda is None:
        jeda = JEDA_DEFAULT.get(provider, 0.0)

    client = OpenAI(api_key=api_key, base_url=base_url)

    sampel = bangun_sampel(model_key, n, strip=strip)  # seed & sampel IDENTIK dgn skrip Anthropic
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = ckpt_path(model_key, tag)

    sudah = set()
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for baris in f:
                try:
                    sudah.add(json.loads(baris)["idx"])
                except Exception:
                    pass
        print(f"Resume: {len(sudah)} item sudah dinilai.")

    sistem = RUBRIK + (REF_NOTE if show_reference else "") + "\n\n" + _INSTRUKSI_JSON
    sisa = [it for it in sampel if it["idx"] not in sudah]
    if batas:
        sisa = sisa[:batas]
    ref_txt = "DENGAN referensi" if show_reference else "TANPA referensi (objektif)"
    strip_txt = " | STRIP-DISCLAIMER (kedua kondisi)" if strip else ""
    tag_txt = f" | run='{tag}'" if tag else ""
    jeda_txt = f" | jeda {jeda:.1f}s/panggilan" if jeda else ""
    print(f"Menilai {len(sisa)} item — {provider}:{judge_model} | {ref_txt} | "
          f"temperature=0{strip_txt}{tag_txt}{jeda_txt}")

    state = {"json": True}  # apakah json_object mode masih dipakai
    gagal = 0
    with open(ckpt, "a", encoding="utf-8") as out:
        for ke, item in enumerate(sisa, 1):
            kwargs = dict(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sistem},
                    {"role": "user", "content": _prompt(item, show_reference)},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            if jeda and ke > 1:
                time.sleep(jeda)
            nilai = None
            for percobaan in range(5):
                try:
                    resp = _panggil(client, kwargs, state)
                    nilai = _ekstrak_json(resp.choices[0].message.content)
                    if not _valid(nilai):
                        raise ValueError("nilai di luar enum rubrik")
                    break
                except Exception as e:
                    if _kuota_habis(e):
                        # Permanen — jangan diulang, dan jangan bakar sisa daftar.
                        print(f"  [{ke}/{len(sisa)}] KUOTA HABIS: {e}")
                        print(f"\nDIHENTIKAN di item ke-{ke}. Isi ulang kredit lalu jalankan "
                              f"perintah yang sama — {len(sudah)} item yang sudah dinilai "
                              f"TIDAK akan diulang.")
                        raise SystemExit(1)
                    if _batas_laju(e) and percobaan < 4:
                        tunggu = 10 * (percobaan + 1)  # 10s, 20s, 30s, 40s
                        print(f"  [{ke}/{len(sisa)}] kena batas laju, tunggu {tunggu}s "
                              f"(percobaan {percobaan + 1}/4)")
                        time.sleep(tunggu)
                        continue
                    print(f"  [{ke}/{len(sisa)}] idx={item['idx']} ERROR: {type(e).__name__}: {e}")
                    break
            if nilai is None:
                gagal += 1
                continue

            out.write(json.dumps({"idx": item["idx"], "kunci": item["kunci"], "nilai": nilai,
                                  "judge": f"{provider}:{judge_model}"},
                                 ensure_ascii=False) + "\n")
            out.flush()  # tulis segera — aman kalau proses diputus
            if ke % 10 == 0 or ke == len(sisa):
                print(f"  [{ke}/{len(sisa)}] selesai")

    print(f"\nSelesai. Gagal/dilewati: {gagal}. Checkpoint -> {ckpt}")
    print(f"Ringkas: python llm_judge_tier2.py --summarize --model-key {model_key}"
          + (f" --tag {tag}" if tag else ""))
    print(f"(Catat di naskah: penilai = {provider}:{judge_model}, temperature=0, "
          f"{'dengan' if show_reference else 'tanpa'} referensi, "
          f"{'strip-disclaimer, ' if strip else ''}blind & paired, seed=42)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="jalankan penilaian")
    ap.add_argument("--model-key", default="0.8b", choices=["0.8b", "2b", "4b"])
    ap.add_argument("--provider", default="openai", choices=sorted(PROVIDER))
    ap.add_argument("--judge-model", default="gpt-4o",
                    help="mis. gpt-4o, gpt-4o-mini, atau meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--show-reference", action="store_true",
                    help="sodorkan jawaban referensi dokter ke penilai (default TIDAK; "
                         "menyodorkannya bisa memberi keunggulan semu ke fine-tuned)")
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="batas token keluaran/panggilan; kecil = hemat biaya output (default 700)")
    ap.add_argument("--tag", default="",
                    help="penanda run terpisah (mis. 'strip', 'gpt5'); kosong = run utama")
    ap.add_argument("--strip-disclaimer", action="store_true",
                    help="ABLATION: buang kalimat disclaimer dari KEDUA kondisi sebelum dinilai")
    ap.add_argument("--jeda", type=float, default=None,
                    help="jeda detik antar panggilan; default 4s utk gemini (tier gratis "
                         "dibatasi per menit), 0 utk penyedia berbayar")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah panggilan (uji coba)")
    args = ap.parse_args()

    if args.run:
        jalankan(args.model_key, args.n, args.limit, args.provider, args.judge_model,
                 args.show_reference, args.max_tokens, args.strip_disclaimer, args.tag,
                 args.jeda)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
