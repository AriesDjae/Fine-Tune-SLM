"""
app.py — arena penilaian dokter, seluruhnya di dalam satu Space Docker.

Menggantikan pasangan Vercel + RunPod: halaman yang dilihat dokter PERSIS
`public/index.html` yang sudah dipakai sebelumnya, kontrak API-nya juga sama
(`/api/awal`, `/api/tanya`, `/api/status`, `/api/batal`, `/api/simpan`), hanya saja
inferensinya sekarang berjalan di proses ini lewat llama.cpp, dan penilaian ditulis
langsung ke HF Dataset privat.

DUA SUMBER JAWABAN, satu protokol:

1. **Bank** (`bank/<versi_protokol>/<hash>.json`) — jawaban pra-generate di laptop
   peneliti. Kalau pertanyaannya ada di bank, jawabannya disajikan seketika, tanpa
   inferensi sama sekali. Sah karena decoding deterministik (greedy, seed 42,
   `no_repeat_ngram=3`): jawaban live akan identik.
2. **Live** — kalau tidak ada di bank, keenam model dijalankan di sini.

Sifat yang tidak boleh hilang, warisan `arsip/pivot6_jalur-lama/hf_space/KONTRAK_RUNPOD.md`:

- `SYSTEM_PROMPT` & `PARAMS_GENERASI` verbatim dari `arena/config.py`.
- Pemblokir n-gram KERAS; kalau tidak bisa aktif, job GAGAL, tidak dilanjutkan diam-diam.
- Label A-F buta: peta ke nama model asli TIDAK ADA di berkas ini maupun di mana pun
  di dalam Space. Tirai dibuka belakangan saat analisis lewat `arena/config.py`.
- Urutan kartu diacak DI SERVER tiap pertanyaan, dan `urutan_tampil` ikut disimpan.
- Job gagal → browser tidak boleh bisa menyimpan skor.
"""

from __future__ import annotations

import datetime
import gc
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid

import numpy as np
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat_format import MODEL_MODES, build_prompt, clean_greeting

MODE = MODEL_MODES["qwen"]

# =========================================================================== #
# SALINAN VERBATIM dari arena/config.py. Disalin, bukan diimpor — Space hanya
# membundel isi direktori ini. Kalau config.py berubah, SALIN ULANG ke sini;
# `scripts/uji_konsistensi.py` membandingkan keduanya dan gagal bila menyimpang.
# =========================================================================== #
LABELS = ["Model A", "Model B", "Model C", "Model D", "Model E", "Model F"]

MODELS = {
    "Model A": {"repo_gguf": "AriesDjaenuri/qwen35-2b-medical-id-GGUF",
                "berkas": "Qwen3.5-2B.Q4_K_M.gguf",
                "tokenizer": "AriesDjaenuri/qwen35-2b-medical-id"},
    "Model B": {"repo_gguf": "unsloth/Qwen3.5-0.8B-GGUF",
                "berkas": "Qwen3.5-0.8B-Q4_K_M.gguf",
                "tokenizer": "unsloth/Qwen3.5-0.8B"},
    "Model C": {"repo_gguf": "AriesDjaenuri/qwen35-4b-medical-id-GGUF",
                "berkas": "Qwen3.5-4B.Q4_K_M.gguf",
                "tokenizer": "AriesDjaenuri/qwen35-4b-medical-id"},
    "Model D": {"repo_gguf": "AriesDjaenuri/qwen35-0.8b-medical-id-GGUF",
                "berkas": "Qwen3.5-0.8B.Q4_K_M.gguf",
                "tokenizer": "AriesDjaenuri/qwen35-0.8b-medical-id"},
    "Model E": {"repo_gguf": "unsloth/Qwen3.5-4B-GGUF",
                "berkas": "Qwen3.5-4B-Q4_K_M.gguf",
                "tokenizer": "unsloth/Qwen3.5-4B"},
    "Model F": {"repo_gguf": "unsloth/Qwen3.5-2B-GGUF",
                "berkas": "Qwen3.5-2B-Q4_K_M.gguf",
                "tokenizer": "unsloth/Qwen3.5-2B"},
}

SYSTEM_PROMPT = (
    "Anda adalah asisten informasi kesehatan berbahasa Indonesia. Berikan penjelasan "
    "yang jelas, akurat, dan hati-hati berdasarkan pengetahuan klinis yang mapan. "
    "Selalu ingatkan bahwa informasi ini bukan pengganti konsultasi langsung dengan dokter."
)

PARAMS_GENERASI = {
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "seed": 42,
    "no_repeat_ngram": 3,
    "n_ctx": 4096,
    "system_prompt": SYSTEM_PROMPT,
}

ACAK_POSISI = True
SKALA = [1, 2, 3, 4, 5]
HF_DATASET_REPO = "AriesDjaenuri/p6-penilaian-dokter"

CONTOH_PERTANYAAN = [
    "Dok, saya demam sejak 3 hari disertai batuk kering. Apakah berbahaya?",
    "Bolehkah minum parasetamol dan ibuprofen bersamaan?",
    "Anak saya demam 38,5 derajat setelah imunisasi. Normalkah?",
    "Saya penderita diabetes, akhir-akhir ini kaki sering kesemutan dan ada luka kecil "
    "yang lama sembuh. Apa yang harus saya lakukan?",
    "Ada benjolan di payudara yang tidak terasa nyeri, sebaiknya saya periksa ke dokter apa?",
    "Antibiotik saya masih bersisa dari sakit sebelumnya, bolehkah saya minum lagi "
    "sekarang karena batuk pilek?",
]
# =========================================================================== #

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
N_CTX = int(os.environ.get("N_CTX", PARAMS_GENERASI["n_ctx"]))
N_THREADS = int(os.environ.get("N_THREADS", "0")) or None
# 0 = CPU murni (perangkat keras cpu-upgrade). Di t4-medium setel GPU_LAYERS=99 lewat
# variabel Space; keenam GGUF berjumlah 9,3 GB, masih muat di T4 16 GB.
GPU_LAYERS = int(os.environ.get("GPU_LAYERS", "0"))
# Di CPU 32 GB keenam model muat menetap (9,3 GB bobot + KV cache); turunkan lewat
# variabel Space kalau perangkat kerasnya lebih kecil.
MAKS_DIMUAT = int(os.environ.get("MAKS_DIMUAT", "6"))

# Kontainer melihat inti HOST (mis. 64) padahal kuota cgroup-nya 8, dan ggml memakai
# OpenMP yang menentukan jumlah thread dari inti yang terlihat. HF sendiri sudah
# menyetel OMP_NUM_THREADS, jadi di sini hanya dihormati bila operator memaksanya.
# Harus sebelum llama_cpp dimuat; setelah itu tidak berpengaruh.
if N_THREADS:
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS)


def _log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Identitas protokol & pertanyaan — harus sama persis dengan arena/mesin.py,
# kalau tidak berkas bank tidak akan pernah ketemu.
# --------------------------------------------------------------------------- #
def normalisasi(pertanyaan: str) -> str:
    return re.sub(r"\s+", " ", (pertanyaan or "").strip()).lower()


def hash_pertanyaan(pertanyaan: str) -> str:
    return hashlib.sha1(normalisasi(pertanyaan).encode("utf-8")).hexdigest()[:16]


def versi_protokol(p: dict | None = None) -> str:
    p = p or PARAMS_GENERASI
    sidik = hashlib.sha1(p["system_prompt"].encode("utf-8")).hexdigest()[:6]
    return (f"q4km-{p['max_tokens']}-ngram{p['no_repeat_ngram']}"
            f"-t{p['temperature']}-seed{p['seed']}-sys{sidik}")


VERSI = versi_protokol()
DIR_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank", VERSI)


def _sinkron_bank_dari_dataset() -> int:
    """Tarik bank yang pernah ditulis Space ini pada run sebelumnya.

    Disk Space bersifat sementara: setiap kali Space tidur lalu bangun, seluruh isinya
    kembali ke isi image. Tanpa sinkronisasi ini, jawaban yang susah payah dihitung di
    T4 berbayar hilang dan dihitung ulang dari nol saat pertanyaan yang sama muncul lagi.
    Berkas bank yang ikut di dalam image (hasil pra-generate laptop) tidak tertimpa —
    `snapshot_download` hanya menambahkan yang belum ada.
    """
    if not HF_TOKEN:
        return 0
    try:
        from huggingface_hub import snapshot_download
        # local_dir = akar aplikasi, karena pola unduhnya sudah memuat "bank/<versi>/".
        snapshot_download(HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN,
                          local_dir=os.path.dirname(os.path.dirname(DIR_BANK)),
                          allow_patterns=[f"bank/{VERSI}/*.json"])
    except Exception as e:
        _log(f"[bank] sinkron dari dataset gagal (tidak fatal): {e}")
    return len(os.listdir(DIR_BANK)) if os.path.isdir(DIR_BANK) else 0


def _simpan_ke_bank(pertanyaan: str, jawaban: dict) -> None:
    """Tulis jawaban live ke bank — lokal untuk giliran berikutnya, dan ke dataset
    supaya tidak ikut hilang saat Space tidur. Bentuk berkasnya sama persis dengan
    keluaran `scripts/pragenerate_arena.py`, jadi bank dari dua sumber bisa bercampur
    tanpa cabang kode di mana pun."""
    isi = {
        "pertanyaan": pertanyaan,
        "hash": hash_pertanyaan(pertanyaan),
        "versi_protokol": VERSI,
        "dibuat": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "mesin": f"hf-space-{'gpu' if GPU_LAYERS else 'cpu'}",
        "params_dipakai": PARAMS_GENERASI,
        "pemblokir_ngram": True,
        "jawaban": jawaban,
    }
    nama = f"{isi['hash']}.json"
    try:
        os.makedirs(DIR_BANK, exist_ok=True)
        with open(os.path.join(DIR_BANK, nama), "w", encoding="utf-8") as f:
            json.dump(isi, f, ensure_ascii=False, indent=1)
    except OSError as e:
        _log(f"[bank] gagal menulis lokal: {e}")
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=json.dumps(isi, ensure_ascii=False, indent=1).encode("utf-8"),
            path_in_repo=f"bank/{VERSI}/{nama}", repo_id=HF_DATASET_REPO,
            repo_type="dataset", commit_message=f"bank live {isi['hash']}")
    except Exception as e:
        _log(f"[bank] gagal mengunggah: {e}")


def dari_bank(pertanyaan: str) -> dict | None:
    """Jawaban pra-generate untuk pertanyaan ini, atau None.

    Hanya diterima bila LENGKAP (enam label). Bank setengah jadi lebih berbahaya
    daripada tidak ada bank: dokter akan menilai kartu kosong sebagai jawaban buruk.
    """
    jalur = os.path.join(DIR_BANK, f"{hash_pertanyaan(pertanyaan)}.json")
    if not os.path.exists(jalur):
        return None
    try:
        with open(jalur, encoding="utf-8") as f:
            isi = json.load(f)
    except (OSError, ValueError):
        return None
    jawaban = isi.get("jawaban") or {}
    if not all(l in jawaban for l in LABELS):
        return None
    return isi


# --------------------------------------------------------------------------- #
# Pemblokir n-gram — WAJIB
# --------------------------------------------------------------------------- #
def _pemblokir_ngram(n: int):
    """Replika `no_repeat_ngram_size` milik eval.py sebagai logits processor.

    llama.cpp hanya punya repeat_penalty (hukuman lunak). eval.py memblokir KERAS:
    token yang akan melengkapi n-gram yang pernah muncul diberi probabilitas nol.
    Tanpa ini model kecil terkunci mengulang kalimat sampai jatah token habis.
    Disalin apa adanya dari arena/mesin.py.
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


def prosesor(n: int):
    if not n or n <= 1:
        return None
    try:
        from llama_cpp import LogitsProcessorList
    except ImportError:
        return None
    return LogitsProcessorList([_pemblokir_ngram(n)])


# --------------------------------------------------------------------------- #
# Muatan model
# --------------------------------------------------------------------------- #
_MUAT: dict[str, dict] = {}
_URUT: list[str] = []
_KUNCI_MUAT = threading.Lock()


def _muat(label: str):
    with _KUNCI_MUAT:
        if label in _MUAT:
            _URUT.remove(label)
            _URUT.append(label)
            return _MUAT[label]["llm"], _MUAT[label]["tok"]

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        cfg = MODELS[label]
        t0 = time.time()
        jalur = hf_hub_download(cfg["repo_gguf"], cfg["berkas"])
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        kw = {"n_threads": N_THREADS} if N_THREADS else {}
        llm = Llama(model_path=jalur, n_ctx=N_CTX, n_gpu_layers=GPU_LAYERS,
                    seed=PARAMS_GENERASI["seed"], verbose=False, **kw)

        while len(_URUT) >= MAKS_DIMUAT:
            _MUAT.pop(_URUT.pop(0), None)
            gc.collect()

        _MUAT[label] = {"llm": llm, "tok": tok}
        _URUT.append(label)
        _log(f"[muat] {label} {time.time() - t0:.1f}s gpu_layers={GPU_LAYERS}")
        return llm, tok


def _jawab(label: str, pertanyaan: str, proc) -> dict:
    llm, tok = _muat(label)
    p = PARAMS_GENERASI
    prompt = build_prompt(tok, pertanyaan, MODE, system=p["system_prompt"])
    t0 = time.time()
    keluar = llm.create_completion(
        prompt, max_tokens=p["max_tokens"], temperature=p["temperature"],
        top_p=p["top_p"], top_k=p["top_k"], repeat_penalty=p["repeat_penalty"],
        seed=p["seed"],
        # Jaring pengaman bila metadata EOS suatu GGUF tidak terbawa; roster Pivot 6
        # seluruhnya ChatML, jadi satu token ini cukup.
        stop=["<|im_end|>"], logits_processor=proc)
    pilihan = keluar["choices"][0]
    return {"teks": clean_greeting(pilihan["text"]),
            "terpotong": pilihan.get("finish_reason") == "length",
            "detik": round(time.time() - t0, 1),
            "token": (keluar.get("usage") or {}).get("completion_tokens")}


# --------------------------------------------------------------------------- #
# Job — /tanya memulai, /status mengambil potongan yang sudah jadi
# --------------------------------------------------------------------------- #
_JOB: dict[str, dict] = {}
_KUNCI_JOB = threading.Lock()

# SATU generasi pada satu waktu untuk seluruh Space. Bukan pilihan gaya:
# objek `Llama` llama.cpp tidak aman dipakai dua thread sekaligus, dan pembuangan LRU
# di `_muat` bisa membuang model yang sedang dipakai thread lain — dua dokter yang
# menekan "tanya" bersamaan cukup untuk membuat jawaban tercampur atau proses mati.
# Job kedua menunggu, dan browser diberi tahu lewat medan `antre`.
_KUNCI_GENERASI = threading.Lock()
TTL_JOB = 3600          # buang job lama supaya proses panjang tidak menumpuk memori


def _kerjakan(id_job: str, pertanyaan: str, urutan: list[str], bank: dict | None):
    proc = prosesor(PARAMS_GENERASI["no_repeat_ngram"])
    if proc is None and bank is None:
        with _KUNCI_JOB:
            _JOB[id_job]["galat"] = (
                "Pemblokir n-gram tidak tersedia di pemasangan llama-cpp-python ini. "
                "Job dihentikan: tanpa pemblokir, jawaban tidak sebanding dengan protokol "
                "yang dilaporkan di naskah.")
            _JOB[id_job]["selesai"] = True
        return

    with _KUNCI_JOB:
        _JOB[id_job]["potongan"].append({
            "jenis": "siap", "pemblokir_ngram": True,
            "params_dipakai": {**{k: v for k, v in PARAMS_GENERASI.items()
                                  if k != "system_prompt"},
                               "sumber": "bank" if bank else "live",
                               "gpu_layers": GPU_LAYERS}})

    # Jawaban bank tidak menyentuh mesin sama sekali, jadi tidak perlu ikut antre.
    kunci = None if bank else _KUNCI_GENERASI
    if kunci is not None:
        with _KUNCI_JOB:
            _JOB[id_job]["antre"] = True
        kunci.acquire()
        with _KUNCI_JOB:
            _JOB[id_job]["antre"] = False
    try:
        for posisi, label in enumerate(urutan):
            with _KUNCI_JOB:
                if _JOB.get(id_job, {}).get("batal"):
                    return
            try:
                hasil = (bank["jawaban"][label] if bank else
                         _jawab(label, pertanyaan, proc))
            except Exception as e:
                # Satu model gagal = SELURUH job gagal. Melanjutkan dengan kartu kosong
                # justru berbahaya: dokter melihat kartu tanpa jawaban tanpa tahu
                # sebabnya, dan menilainya sebagai jawaban buruk.
                _log(f"[job {id_job}] {label} gagal: {e}")
                with _KUNCI_JOB:
                    _JOB[id_job]["galat"] = (
                        f"{label} gagal dijawab ({type(e).__name__}: {e}). "
                        "Job dihentikan supaya tidak ada kartu kosong yang ikut dinilai.")
                    _JOB[id_job]["selesai"] = True
                return
            with _KUNCI_JOB:
                _JOB[id_job]["potongan"].append({
                    "jenis": "jawaban", "label": label, "posisi": posisi,
                    "teks": hasil.get("teks", ""), "terpotong": bool(hasil.get("terpotong")),
                    "detik": hasil.get("detik"), "token": hasil.get("token")})
    finally:
        if kunci is not None:
            kunci.release()

    with _KUNCI_JOB:
        _JOB[id_job]["selesai"] = True
        jadi = {p["label"]: {k: p[k] for k in ("teks", "terpotong", "detik", "token")}
                for p in _JOB[id_job]["semua"] + _JOB[id_job]["potongan"]
                if p.get("jenis") == "jawaban"}

    # Hanya job live yang layak disimpan; yang dari bank memang sudah ada di sana.
    if bank is None and len(jadi) == len(LABELS):
        _simpan_ke_bank(pertanyaan, jadi)


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
rute = APIRouter()

# Kode akses sesi — dibagikan ke dokter di luar aplikasi. BUKAN autentikasi sungguhan:
# ia tidak mengenali siapa yang memakai dan bisa diteruskan ke orang lain. Gunanya
# sempit dan jelas: Space yang publik tanpa ini berarti siapa pun yang menemukan
# tautannya bisa (a) menjalankan enam model dan membakar jam GPU berbayar, dan
# (b) menulis baris penilaian palsu ke dataset. Kosong = gerbang mati (untuk uji lokal).
KODE_AKSES = os.environ.get("KODE_AKSES", "").strip()


def _kode_salah(request: Request) -> JSONResponse | None:
    if not KODE_AKSES:
        return None
    kirim = (request.headers.get("X-Kode-Akses") or
             request.query_params.get("kode") or "").strip()
    if kirim == KODE_AKSES:
        return None
    return JSONResponse({"galat": "Kode akses salah."}, status_code=401)


def _siap() -> tuple[bool, str]:
    if not HF_TOKEN:
        return False, "HF_TOKEN belum dipasang di Secret Space — penilaian tidak bisa disimpan"
    n_bank = len(os.listdir(DIR_BANK)) if os.path.isdir(DIR_BANK) else 0
    tempat = "GPU" if GPU_LAYERS else "CPU"
    return True, f"siap ({tempat}, bank {n_bank} pertanyaan)"


@rute.get("/healthz")
def healthz():
    siap, alasan = _siap()
    return {"ok": True, "siap": siap, "alasan": alasan, "mock": False,
            "versi_protokol": VERSI}


@rute.get("/awal")
def awal(request: Request):
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    siap, alasan = _siap()
    return {"siap": siap, "alasan": alasan, "mock": False,
            "labels": LABELS, "skala": SKALA, "contoh": CONTOH_PERTANYAAN}


class Tanya(BaseModel):
    pertanyaan: str


@rute.post("/tanya")
def tanya(badan: Tanya, request: Request):
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    siap, alasan = _siap()
    if not siap:
        return JSONResponse({"galat": alasan}, status_code=503)

    pertanyaan = (badan.pertanyaan or "").strip()
    if not pertanyaan:
        return JSONResponse({"galat": "Tuliskan pertanyaannya dulu."}, status_code=400)

    urutan = LABELS[:]
    if ACAK_POSISI:
        # Diacak di server: kalau browser yang mengacak, urutannya bisa diutak-atik
        # dari konsol dan efek posisi tidak lagi terkendali.
        random.shuffle(urutan)

    bank = dari_bank(pertanyaan)
    id_job = uuid.uuid4().hex[:12]
    sekarang = time.time()
    with _KUNCI_JOB:
        # Job lama dibuang di sini, bukan lewat thread pembersih: Space ini bisa tidur
        # kapan saja, jadi pembersihan yang menempel pada lalu lintas lebih jujur.
        for lama in [k for k, v in _JOB.items() if sekarang - v["dibuat"] > TTL_JOB]:
            _JOB.pop(lama, None)
        _JOB[id_job] = {"potongan": [], "selesai": False, "batal": False,
                        "galat": None, "urutan": urutan, "semua": [],
                        "antre": False, "sumber": "bank" if bank else "live",
                        "dibuat": sekarang}
    threading.Thread(target=_kerjakan, args=(id_job, pertanyaan, urutan, bank),
                     daemon=True).start()
    return {"id": id_job, "urutan": urutan, "sumber": "bank" if bank else "live"}


@rute.get("/status")
def status(id: str, request: Request):
    """Ambil potongan yang sudah jadi. MENGONSUMSI: potongan yang sudah diambil tidak
    muncul lagi, jadi browser yang mengakumulasi — sama seperti kontrak RunPod lama.
    Saat selesai, daftar UTUH ikut dikirim lewat `semua` untuk rekonsiliasi."""
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    with _KUNCI_JOB:
        keadaan = _JOB.get(id)
        if keadaan is None:
            return JSONResponse({"galat": "job tidak dikenal"}, status_code=404)
        if keadaan["galat"]:
            return JSONResponse({"galat": keadaan["galat"]}, status_code=502)
        potongan = keadaan["potongan"]
        keadaan["potongan"] = []
        keadaan["semua"].extend(potongan)
        hasil = {"selesai": keadaan["selesai"],
                 "antre": bool(keadaan.get("antre")), "potongan": potongan}
        if keadaan["selesai"]:
            hasil["semua"] = [p for p in keadaan["semua"] if p.get("jenis") == "jawaban"]
    return hasil


@rute.post("/batal")
def batal(id: str):
    with _KUNCI_JOB:
        if id in _JOB:
            _JOB[id]["batal"] = True
    return {"ok": True}


# Sama dengan POLA_EMAIL di public/index.html — keduanya harus berubah bersamaan.
POLA_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Penilaian(BaseModel):
    id_sesi: str
    penilai: str
    pertanyaan: str
    urutan: list[str]
    skor: dict[str, int]
    tidak_aman: list[str] = []
    jawaban: dict[str, str]
    terpotong: list[str] = []
    detik: dict[str, float | None] = {}
    token: dict[str, int | None] = {}
    email: str = ""
    saran: str = ""
    sistem: dict = {}
    peristiwa: list[dict] = []


@rute.post("/simpan")
def simpan(p: Penilaian, request: Request):
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    if not HF_TOKEN:
        return JSONResponse({"galat": "HF_TOKEN belum dipasang di Secret Space"},
                            status_code=503)
    if not (p.penilai or "").strip():
        return JSONResponse({"galat": "Isi dulu nama penilai."}, status_code=400)
    email = (p.email or "").strip()
    if not email:
        return JSONResponse({"galat": "Isi dulu email Anda."}, status_code=400)
    if not POLA_EMAIL.match(email):
        return JSONResponse({"galat": "Format email belum benar — contoh: andi@contoh.com."},
                            status_code=400)
    saran = (p.saran or "").strip()
    if len(re.findall(r"[^\W_]", saran)) < 3:
        return JSONResponse({"galat": "Isi dulu saran Anda."}, status_code=400)

    # Validasi diulang di server. Yang di browser hanya kenyamanan; ini yang menentukan,
    # karena tanpa ini satu permintaan buatan tangan bisa menyisipkan baris tanpa skor.
    belum = [x for x in p.urutan if x not in p.skor]
    if belum:
        return JSONResponse({"galat": f"Belum dinilai: {', '.join(belum)}."}, status_code=400)
    liar = [x for x in p.urutan if p.skor[x] not in SKALA]
    if liar:
        return JSONResponse(
            {"galat": f"Skor di luar skala {SKALA[0]}-{SKALA[-1]}: {', '.join(liar)}."},
            status_code=400)
    kosong = [x for x in p.urutan if not (p.jawaban.get(x) or "").strip()]
    if kosong:
        return JSONResponse(
            {"galat": f"Ada kartu tanpa jawaban ({', '.join(kosong)}) - kirim ulang."},
            status_code=400)

    waktu = datetime.datetime.now(datetime.timezone.utc)
    rekaman = {
        "waktu_utc": waktu.isoformat(timespec="seconds"),
        "id_sesi": p.id_sesi,
        "penilai": p.penilai.strip(),
        "pertanyaan": p.pertanyaan,
        # label buta saja — nama asli model TIDAK ikut tersimpan.
        "skor": {k: int(v) for k, v in p.skor.items()},
        "tidak_aman": p.tidak_aman,
        "jawaban": p.jawaban,
        "terpotong": p.terpotong,
        "detik": p.detik,
        "token": p.token,
        "urutan_tampil": p.urutan,     # untuk memeriksa efek posisi saat analisis
        # Email hanya di sini (dataset privat) — tidak di pesan commit, tidak di log sesi.
        # Baris sebelum 2026-09-25 memakai kunci `catatan` (opsional) dan tanpa email.
        "email": email,
        "saran": saran,
        "sistem": {"system_prompt": SYSTEM_PROMPT,
                   "max_tokens": PARAMS_GENERASI["max_tokens"],
                   "temperature": PARAMS_GENERASI["temperature"],
                   "kuantisasi": "Q4_K_M", "acak_posisi": ACAK_POSISI, "skala": SKALA,
                   "backend": f"hf-space-{'gpu' if GPU_LAYERS else 'cpu'}",
                   "frontend": "hf-space",
                   "versi_protokol": VERSI,
                   "diminta": PARAMS_GENERASI,
                   "dipakai": p.sistem.get("params_dipakai"),
                   "pemblokir_ngram": p.sistem.get("pemblokir_ngram")},
    }

    nama = f"rating_{waktu.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=json.dumps(rekaman, ensure_ascii=False, indent=1).encode("utf-8"),
            path_in_repo=f"ratings/{nama}", repo_id=HF_DATASET_REPO, repo_type="dataset",
            commit_message=f"penilaian {p.penilai.strip()} {waktu.strftime('%Y-%m-%d %H:%M')}")
        if p.peristiwa:
            api.upload_file(
                path_or_fileobj="\n".join(json.dumps(x, ensure_ascii=False)
                                          for x in p.peristiwa).encode("utf-8"),
                path_in_repo=f"logs/sesi_{p.id_sesi}.jsonl",
                repo_id=HF_DATASET_REPO, repo_type="dataset",
                commit_message=f"log sesi {p.id_sesi}")
    except Exception as e:
        return JSONResponse({"galat": f"gagal menyimpan ke HF Dataset: {type(e).__name__}: {e}"},
                            status_code=502)

    return {"ok": True, "berkas": nama, "waktu": waktu.strftime("%H:%M"),
            "ditandai": len(p.tidak_aman)}


# --------------------------------------------------------------------------- #
# Tugas 2 — anotasi tanda bahaya (red-flag) pada PERTANYAAN pasien.
#
# Berbeda dari penilaian jawaban di atas: di sini dokter hanya membaca keluhan pasien
# dan memutuskan apakah perlu penanganan segera (Y/T). Hasilnya dipakai menghitung
# recall/FPR detektor red-flag (redflag_annotate.py --score). Daftar pertanyaannya
# (redflag_soal.json) TIDAK memuat keluaran detektor maupun strata — penilaian buta.
#
# Space membagikan pertanyaan per giliran kepada dokter mana pun yang datang, menurut
# urutan di berkas. Pertanyaan ber-`target` 2 dinilai dua dokter berbeda.
# --------------------------------------------------------------------------- #
def _muat_soal_rf() -> dict | None:
    """Daftar pertanyaan diambil dari dataset PRIVAT, bukan disalin ke repo Space: Space
    ini publik, sedangkan isinya 400 keluhan pasien dari test set."""
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_DATASET_REPO, "redflag_soal.json", repo_type="dataset",
                               token=HF_TOKEN)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _log(f"[redflag] daftar pertanyaan tidak bisa dimuat: {e}")
        return None


_RF = _muat_soal_rf()
_RF_ID = {s["id"]: s for s in _RF["soal"]} if _RF else {}
_RF_LABEL: dict[str, set[str]] = {}          # id -> email yang sudah menilai
_RF_PESAN: dict[str, dict[str, float]] = {}  # id -> {email: batas waktu reservasi}
_RF_KUNCI = threading.Lock()
RF_PER_GILIRAN = 10
RF_TTL_PESAN = 30 * 60
# Waktu dokter terbatas: tiap dokter paling banyak menilai sekian pertanyaan (±12 menit).
RF_MAKS_PER_DOKTER = int(os.environ.get("RF_MAKS_PER_DOKTER", "30"))


def _catat_label_rf(rek: dict) -> None:
    if rek.get("uji"):
        return  # uji fungsional tidak boleh membuat pertanyaan dianggap sudah dinilai
    email = rek["email"].lower()
    for lab in rek["label"]:
        _RF_LABEL.setdefault(lab["id"], set()).add(email)


def _sinkron_redflag() -> int:
    """Muat label yang sudah tersimpan di dataset, supaya Space yang baru bangun tidak
    membagikan ulang pertanyaan yang sudah dinilai."""
    if not (HF_TOKEN and _RF):
        return 0
    try:
        from huggingface_hub import snapshot_download
        folder = snapshot_download(HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN,
                                   allow_patterns=["redflag/*.json"])
        d = os.path.join(folder, "redflag")
        n = 0
        for nama in os.listdir(d) if os.path.isdir(d) else []:
            with open(os.path.join(d, nama), encoding="utf-8") as f:
                _catat_label_rf(json.load(f))
            n += 1
        return n
    except Exception as e:
        _log(f"[redflag] sinkron dari dataset gagal (tidak fatal): {e}")
        return 0


def _progres_rf(email: str = "") -> dict:
    n = [len(_RF_LABEL.get(s["id"], ())) for s in _RF["soal"]]
    saya = sum(email in v for v in _RF_LABEL.values()) if email else 0
    return {"total": len(n), "selesai": sum(x >= 1 for x in n), "dua_dokter": sum(x >= 2 for x in n),
            "saya": saya, "kuota": RF_MAKS_PER_DOKTER}


@rute.get("/redflag/awal")
def redflag_awal(request: Request):
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    if not _RF:
        return JSONResponse({"galat": "Daftar pertanyaan Tugas 2 belum dipasang."}, status_code=503)
    with _RF_KUNCI:
        return {"kategori": _RF["kategori"], "per_giliran": RF_PER_GILIRAN, **_progres_rf()}


@rute.get("/redflag/ambil")
def redflag_ambil(email: str, request: Request):
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    email = (email or "").strip().lower()
    if not POLA_EMAIL.match(email):
        return JSONResponse({"galat": "Isi email yang benar dulu."}, status_code=400)
    sekarang = time.time()
    with _RF_KUNCI:
        sisa_kuota = RF_MAKS_PER_DOKTER - sum(email in v for v in _RF_LABEL.values())
        calon = []
        for urut, s in enumerate(_RF["soal"]):
            sudah = _RF_LABEL.get(s["id"], set())
            if email in sudah:
                continue
            pesan = {e: t for e, t in _RF_PESAN.get(s["id"], {}).items()
                     if t > sekarang and e != email}
            # Reservasi mencegah dua dokter yang bekerja bersamaan mendapat pertanyaan
            # yang sama; kedaluwarsa sendiri kalau dokter menutup halaman.
            terisi = len(sudah) + len(pesan)
            if terisi >= s["target"]:
                continue
            calon.append((terisi, urut, s, pesan))
        # Yang belum dinilai siapa pun didahulukan: dengan 2 dokter semua pertanyaan
        # dinilai sekali, baru dokter berikutnya memberi penilaian kedua.
        calon.sort(key=lambda x: x[:2])
        ambil = []
        for _, _, s, pesan in calon[:max(0, min(RF_PER_GILIRAN, sisa_kuota))]:
            pesan[email] = sekarang + RF_TTL_PESAN
            _RF_PESAN[s["id"]] = pesan
            ambil.append({"id": s["id"], "pertanyaan": s["pertanyaan"]})
        return {"soal": ambil, **_progres_rf(email)}


class LabelRF(BaseModel):
    id: str
    label: str
    kategori: str = ""
    catatan: str = ""


class SimpanRF(BaseModel):
    penilai: str
    email: str
    label: list[LabelRF]
    uji: bool = False


@rute.post("/redflag/simpan")
def redflag_simpan(p: SimpanRF, request: Request):
    tolak = _kode_salah(request)
    if tolak is not None:
        return tolak
    if not HF_TOKEN:
        return JSONResponse({"galat": "HF_TOKEN belum dipasang di Secret Space"}, status_code=503)
    email = (p.email or "").strip().lower()
    if not (p.penilai or "").strip():
        return JSONResponse({"galat": "Isi dulu nama penilai."}, status_code=400)
    if not POLA_EMAIL.match(email):
        return JSONResponse({"galat": "Format email belum benar — contoh: andi@contoh.com."},
                            status_code=400)
    if not 1 <= len(p.label) <= 2 * RF_PER_GILIRAN:
        return JSONResponse({"galat": "Tidak ada penilaian untuk disimpan."}, status_code=400)
    ids = [x.id for x in p.label]
    asing = [i for i in ids if i not in _RF_ID]
    if asing or len(set(ids)) != len(ids):
        return JSONResponse({"galat": "Daftar pertanyaan tidak dikenal — muat ulang halaman."},
                            status_code=400)
    kosong = [i for i, x in zip(ids, p.label) if x.label.strip().upper() not in {"Y", "T"}]
    if kosong:
        return JSONResponse({"galat": f"Belum dijawab Y/T: {len(kosong)} pertanyaan."},
                            status_code=400)
    liar = [x.kategori for x in p.label if x.kategori and x.kategori not in _RF["kategori"]]
    if liar:
        return JSONResponse({"galat": "Kategori tidak dikenal."}, status_code=400)

    waktu = datetime.datetime.now(datetime.timezone.utc)
    rek = {
        "waktu_utc": waktu.isoformat(timespec="seconds"),
        "penilai": p.penilai.strip(),
        "email": email,
        "uji": bool(p.uji),
        "label": [{"id": x.id, "label": x.label.strip().upper(),
                   "kategori": x.kategori if x.label.strip().upper() == "Y" else "",
                   "catatan": x.catatan.strip()} for x in p.label],
    }
    nama = f"rf_{waktu.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=json.dumps(rek, ensure_ascii=False, indent=1).encode("utf-8"),
            path_in_repo=f"redflag/{nama}", repo_id=HF_DATASET_REPO, repo_type="dataset",
            commit_message=f"tanda bahaya {p.penilai.strip()} ({len(p.label)} pertanyaan)")
    except Exception as e:
        return JSONResponse({"galat": f"gagal menyimpan ke HF Dataset: {type(e).__name__}: {e}"},
                            status_code=502)
    with _RF_KUNCI:
        _catat_label_rf(rek)
        for i in ids:
            _RF_PESAN.get(i, {}).pop(email, None)
        prog = _progres_rf(email)
    return {"ok": True, "berkas": nama, "waktu": waktu.strftime("%H:%M"), **prog}


# Didaftarkan dua kali supaya halaman yang sama bisa dipakai apa adanya, baik saat
# di-serve dari Space ini maupun saat diletakkan di belakang proxy yang memangkas /api.
app.include_router(rute, prefix="/api")
app.include_router(rute)

# Halaman dokter = berkas yang sama persis dengan yang dipakai di Vercel.
_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
app.mount("/", StaticFiles(directory=_PUBLIC, html=True), name="public")

def _dukungan_gpu() -> bool | None:
    """Apakah llama.cpp yang TERPASANG punya dukungan offload GPU sama sekali.

    Ini yang membedakan "GPU_LAYERS diisi" dari "GPU benar-benar dipakai": wheel CPU
    menerima `n_gpu_layers=99` tanpa protes dan diam-diam menjalankan semuanya di CPU.
    Di perangkat keras GPU berbayar, kegagalan senyap seperti itu berarti membayar
    tarif GPU untuk kecepatan CPU.
    """
    try:
        from llama_cpp import llama_supports_gpu_offload
        return bool(llama_supports_gpu_offload())
    except Exception:
        return None


_N_BANK = _sinkron_bank_dari_dataset()
_N_RF = _sinkron_redflag()
_GPU_OK = _dukungan_gpu()
if GPU_LAYERS and _GPU_OK is False:
    _log("[arena] ⚠ GPU_LAYERS>0 tetapi llama-cpp-python TANPA dukungan GPU — "
         "image ini memakai wheel CPU. Jalankan: python scripts/space_mode.py gpu")
if not GPU_LAYERS and _GPU_OK:
    _log("[arena] catatan: image berdukungan GPU tetapi GPU_LAYERS=0 — "
         "lapisan tetap di CPU.")
_log(f"[arena] versi protokol {VERSI} | bank {DIR_BANK} ({_N_BANK} berkas) "
     f"| gpu_layers={GPU_LAYERS} (dukungan GPU: {_GPU_OK}) | maks dimuat {MAKS_DIMUAT} "
     f"| kode akses {'AKTIF' if KODE_AKSES else 'MATI'} "
     f"| tanda bahaya {len(_RF_ID)} soal, {_N_RF} berkas label")
