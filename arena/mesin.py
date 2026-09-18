"""
mesin.py — satu-satunya tempat jawaban arena Tier 3 dihasilkan.

Dipakai `scripts/pragenerate_arena.py` (bank) dan `scripts/proses_usulan.py` (usulan
dokter). Keduanya harus memakai mesin yang SAMA, kalau tidak dua dokter bisa menilai
teks yang tidak sebanding.

Ini penerus `webapp/runpod/handler.py` setelah jalur RunPod dipensiunkan
(lihat `webapp/RENCANA_BANK.md`). Perbedaannya cuma satu: tidak ada lagi pembungkus
serverless dan tidak ada lagi salinan roster — parameter diimpor LANGSUNG dari
`hf_space/config.py`, yang memang satu-satunya sumber kebenaran. Berkas ini hanya
berjalan di laptop peneliti, jadi tidak ada persoalan `KEY_MAP` bocor ke server;
sebagai gantinya, `KEY_MAP` tidak pernah disentuh di sini dan tidak pernah ikut ke
berkas keluaran.

Empat hal yang haram diubah (warisan `hf_space/KONTRAK_RUNPOD.md` § 5):

1. `system_prompt` verbatim dari config.
2. Greedy + `seed=42` + `no_repeat_ngram=3` sebagai pemblokir KERAS.
3. Chat template asal tiap model lewat `chat_format.build_prompt()`, bukan template GGUF.
4. Label A-F tidak pernah dipetakan ke nama model di sini.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time

import numpy as np

_AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _AKAR)
sys.path.insert(0, os.path.join(_AKAR, "hf_space"))

from chat_format import MODEL_MODES, build_prompt, clean_greeting   # noqa: E402
from config import MODELS, PARAMS_GENERASI                          # noqa: E402  (hf_space/)

MODE = MODEL_MODES["qwen"]          # roster Pivot 6 seluruhnya Qwen3.5
LABELS = list(MODELS)

# Ditetapkan saat model dimuat, jadi tidak bisa berbeda per pertanyaan.
N_CTX = int(os.environ.get("N_CTX", PARAMS_GENERASI.get("n_ctx", 4096)))
N_THREADS = int(os.environ.get("N_THREADS", "0")) or None    # None = bawaan llama.cpp
CACHE = os.environ.get("HF_HOME") or os.path.join(_AKAR, ".local_models")

_llm: dict = {}
_tok: dict = {}

# Metrik per label, terkumpul sepanjang proses: dipakai skrip bank untuk menulis
# results/bench_cpu_lokal.json. Bukan pengganti uji HP 3-4 GB yang diminta RQ3 —
# ini mengisi baris "perangkat CPU-only" yang sekarang kosong sama sekali, dan
# gratis karena jalurnya sama persis dengan yang dipakai membuat bank.
metrik: dict[str, dict] = {}


def _log(*a):
    print(*a, flush=True)


def _rss_mb() -> float | None:
    """Memori proses saat ini, MB. None kalau psutil tidak ada — angka RQ3 hilang,
    pembuatan bank tetap jalan."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Identitas protokol & pertanyaan
# --------------------------------------------------------------------------- #
def normalisasi(pertanyaan: str) -> str:
    """Bentuk kanonik untuk hash. Tanda baca SENGAJA tidak dibuang: model menjawab
    "Apakah berbahaya?" dan "Apakah berbahaya" secara berbeda, jadi keduanya memang
    dua entri bank, bukan satu."""
    return re.sub(r"\s+", " ", (pertanyaan or "").strip()).lower()


def hash_pertanyaan(pertanyaan: str) -> str:
    return hashlib.sha1(normalisasi(pertanyaan).encode("utf-8")).hexdigest()[:16]


def versi_protokol(p: dict | None = None) -> str:
    """Sidik jari parameter yang menentukan bentuk jawaban.

    Ikut jadi nama folder bank. Mengubah max_tokens / system prompt / kuantisasi /
    no_repeat_ngram akan MENGGANTI versinya, sehingga bank lama ditinggalkan alih-alih
    tercampur diam-diam dengan yang baru — tanpa ini, sebagian dokter menilai protokol A
    dan sebagian protokol B tanpa jejak apa pun.
    """
    p = p or PARAMS_GENERASI
    sidik = hashlib.sha1(p["system_prompt"].encode("utf-8")).hexdigest()[:6]
    return (f"q4km-{p['max_tokens']}-ngram{p['no_repeat_ngram']}"
            f"-t{p['temperature']}-seed{p['seed']}-sys{sidik}")


# --------------------------------------------------------------------------- #
# Pemblokir n-gram — WAJIB, bukan hiasan
# --------------------------------------------------------------------------- #
def _pemblokir_ngram(n: int):
    """Replika `no_repeat_ngram_size` milik eval.py sebagai logits processor.

    llama.cpp hanya punya repeat_penalty (hukuman lunak pada token yang sudah muncul).
    eval.py memakai pemblokiran KERAS: token yang akan melengkapi n-gram yang pernah
    muncul diberi probabilitas nol. Tanpa ini, greedy decoding pada model kecil terkunci
    di siklus kalimat yang sama sampai jatah token habis — teramati nyata pada dua label
    di 1024 token. Disalin apa adanya dari webapp/runpod/handler.py.
    """
    def proses(input_ids, scores):
        ids = np.asarray(input_ids)
        if ids.size < n:
            return scores
        batas = ids.size - n + 1
        cocok = np.ones(batas, dtype=bool)
        awalan = ids[-(n - 1):]
        for j in range(n - 1):
            cocok &= ids[j:j + batas] == awalan[j]
        terlarang = ids[np.flatnonzero(cocok) + (n - 1)]
        if terlarang.size:
            scores[terlarang] = -np.inf
        return scores

    return proses


try:
    from llama_cpp import LogitsProcessorList
    NGRAM_TERSEDIA = True
except ImportError:
    LogitsProcessorList = None
    NGRAM_TERSEDIA = False


def prosesor(n: int):
    """None kalau pemblokir tidak bisa aktif — pemanggil WAJIB memeriksanya dan berhenti,
    bukan melanjutkan diam-diam (lihat memori `decoding-ngram-konsistensi`)."""
    if not n or n <= 1 or not NGRAM_TERSEDIA:
        return None
    return LogitsProcessorList([_pemblokir_ngram(n)])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def muat(label: str) -> None:
    """Muat satu label. Sengaja satu-satu: 6 GGUF sekaligus tidak muat di laptop, dan
    skrip bank memang berjalan per model (lihat pragenerate_arena.py)."""
    if label in _llm:
        return
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    cfg = MODELS[label]
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    jalur = hf_hub_download(cfg["repo_gguf"], cfg["berkas"], cache_dir=CACHE)
    _tok[label] = AutoTokenizer.from_pretrained(cfg["tokenizer"], cache_dir=CACHE)
    kw = {"n_threads": N_THREADS} if N_THREADS else {}
    _llm[label] = Llama(model_path=jalur, n_ctx=N_CTX, n_gpu_layers=0,
                        seed=PARAMS_GENERASI["seed"], verbose=False, **kw)
    lama = time.time() - t0
    metrik.setdefault(label, {"berkas": cfg["berkas"]}).update({
        "muat_detik": round(lama, 1),
        "gguf_mb": round(os.path.getsize(jalur) / 1e6, 1),
        "rss_mb": _rss_mb(),          # setelah bobot masuk memori
        "n_ctx": N_CTX, "n_threads": N_THREADS or "bawaan",
    })
    _log(f"  muat {label}: {lama:.1f}s ({cfg['berkas']})")


def bongkar(label: str) -> None:
    import gc
    _llm.pop(label, None)
    _tok.pop(label, None)
    gc.collect()


def jawab(label: str, pertanyaan: str, p: dict | None = None, proc=None) -> dict:
    """Satu jawaban + metanya. `proc` boleh dioper supaya tidak dibuat ulang tiap panggilan."""
    p = p or PARAMS_GENERASI
    muat(label)
    prompt = build_prompt(_tok[label], pertanyaan, MODE, system=p["system_prompt"])
    t0 = time.time()
    keluar = _llm[label].create_completion(
        prompt,
        max_tokens=p["max_tokens"], temperature=p["temperature"], top_p=p["top_p"],
        top_k=p["top_k"], repeat_penalty=p["repeat_penalty"], seed=p["seed"],
        # Jaring pengaman kalau metadata EOS salah satu GGUF tidak terbawa; roster
        # Pivot 6 seluruhnya ChatML, jadi satu token ini cukup.
        stop=["<|im_end|>"],
        logits_processor=proc if proc is not None else prosesor(p["no_repeat_ngram"]))
    pilihan = keluar["choices"][0]
    alasan = pilihan.get("finish_reason")
    lama = time.time() - t0
    tok = (keluar.get("usage") or {}).get("completion_tokens") or 0
    m = metrik.setdefault(label, {})
    m["n_jawaban"] = m.get("n_jawaban", 0) + 1
    m["total_token"] = m.get("total_token", 0) + tok
    m["total_detik"] = round(m.get("total_detik", 0.0) + lama, 1)
    m["tok_per_detik"] = round(m["total_token"] / m["total_detik"], 2) if m["total_detik"] else None
    m["rss_puncak_mb"] = max(m.get("rss_puncak_mb") or 0, _rss_mb() or 0) or None
    return {
        "teks": clean_greeting(pilihan["text"]),
        "terpotong": alasan == "length",
        "stop": "batas_token" if alasan == "length" else "selesai",
        "detik": round(lama, 1),
        "token": (keluar.get("usage") or {}).get("completion_tokens"),
    }
