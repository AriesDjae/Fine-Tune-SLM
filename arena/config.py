"""
config.py — semua tombol yang boleh diubah tanpa menyentuh app.py.

BLIND: dokter hanya melihat label "Model A".."Model F". Nama asli ada di KEY_MAP dan
TIDAK PERNAH dikirim ke UI maupun ikut tersimpan di berkas penilaian — pembukaan tirai
dilakukan belakangan saat analisis (lihat DEPLOY.md Bagian 6).
"""

# ====== repo Hugging Face ======
# Penampung penilaian dokter. Dibuat otomatis oleh app.py (butuh Secret HF_TOKEN
# ber-scope write). Dibuat PRIVAT karena isinya memuat nama/identitas penilai.
HF_DATASET_REPO = "AriesDjaenuri/p6-penilaian-dokter"

# ====== roster 6 model (Pivot 6: 3 ukuran x baseline/fine-tuned) ======
# Label A-F sengaja DIACAK terhadap ukuran & kondisi supaya urutan label tidak
# membocorkan apa pun (bukan A<B<C menurut ukuran, bukan pula baseline-lalu-finetuned).
#
#   repo_gguf   : repo berisi berkas GGUF Q4_K_M
#   berkas      : nama berkas GGUF persis seperti di repo itu
#   tokenizer   : repo HF yang chat template-nya dipakai untuk merender prompt.
#                 WAJIB repo asal model itu sendiri — inilah yang menjaga
#                 train == eval == demo (lihat chat_format.py).
MODELS = {
    "Model A": {
        "repo_gguf": "AriesDjaenuri/qwen35-2b-medical-id-GGUF",
        "berkas": "Qwen3.5-2B.Q4_K_M.gguf",
        "tokenizer": "AriesDjaenuri/qwen35-2b-medical-id",
    },
    "Model B": {
        "repo_gguf": "unsloth/Qwen3.5-0.8B-GGUF",
        "berkas": "Qwen3.5-0.8B-Q4_K_M.gguf",
        "tokenizer": "unsloth/Qwen3.5-0.8B",
    },
    "Model C": {
        "repo_gguf": "AriesDjaenuri/qwen35-4b-medical-id-GGUF",
        "berkas": "Qwen3.5-4B.Q4_K_M.gguf",
        "tokenizer": "AriesDjaenuri/qwen35-4b-medical-id",
    },
    "Model D": {
        "repo_gguf": "AriesDjaenuri/qwen35-0.8b-medical-id-GGUF",
        "berkas": "Qwen3.5-0.8B.Q4_K_M.gguf",
        "tokenizer": "AriesDjaenuri/qwen35-0.8b-medical-id",
    },
    "Model E": {
        "repo_gguf": "unsloth/Qwen3.5-4B-GGUF",
        "berkas": "Qwen3.5-4B-Q4_K_M.gguf",
        "tokenizer": "unsloth/Qwen3.5-4B",
    },
    "Model F": {
        "repo_gguf": "unsloth/Qwen3.5-2B-GGUF",
        "berkas": "Qwen3.5-2B-Q4_K_M.gguf",
        "tokenizer": "unsloth/Qwen3.5-2B",
    },
}

# ★ RAHASIA — hanya untuk analisis, JANGAN ditampilkan di UI.
KEY_MAP = {
    "Model A": "Qwen3.5-2B fine-tuned",
    "Model B": "Qwen3.5-0.8B baseline",
    "Model C": "Qwen3.5-4B fine-tuned",
    "Model D": "Qwen3.5-0.8B fine-tuned",
    "Model E": "Qwen3.5-4B baseline",
    "Model F": "Qwen3.5-2B baseline",
}

# ====== prompt ======
# PERSIS system prompt yang dipakai di Data/processed_id/test.jsonl, yaitu prompt yang
# dipakai saat training, saat eval.py, dan saat penilaian Tier 2. Mengubahnya berarti
# dokter menilai sistem yang berbeda dari yang dilaporkan di naskah.
#
# Catatan pembingkaian (CLAUDE.md): sasaran deploy = HP masyarakat umum di wilayah tanpa
# jaringan, BUKAN Puskesmas dengan nakes sebagai penyaring.
SYSTEM_PROMPT = (
    "Anda adalah asisten informasi kesehatan berbahasa Indonesia. Berikan penjelasan "
    "yang jelas, akurat, dan hati-hati berdasarkan pengetahuan klinis yang mapan. "
    "Selalu ingatkan bahwa informasi ini bukan pengganti konsultasi langsung dengan dokter."
)

# ====== parameter generasi ======
# Greedy = replika protokol FAIR eval.py (do_sample=False).
# Menaikkan TEMPERATURE membuat jawaban di sini tak sebanding dengan angka di naskah.
#
# MAX_TOKENS SENGAJA BERBEDA dari eval.py (256). Alasannya ada di hasil sendiri:
# pada batas 256, keluaran baseline terpotong 89,9-91,2% sementara fine-tuned hanya
# 39,1-63,1% (results/eval_truncation.json). Dokter yang membaca jawaban terpotong
# akan menghukum pemotongannya, bukan mutunya — dan justru dimensi keamanan yang
# paling terpengaruh. Untuk Tier 3 kelengkapan jawaban lebih penting daripada
# kesebandingan token-per-token dengan Tier 1/Tier 2.
#
# 1024: melampaui panjang record latih terpanjang (p99 ~931 token untuk SELURUH
# record, prompt + jawaban), jadi praktis tidak ada jawaban alami yang tersentuh
# batas ini. Ongkosnya waktu tunggu — 6 model per pertanyaan.
#
# Ini menaikkan batas, BUKAN menghapusnya: model yang terjebak mengulang bisa tetap
# menabraknya. Kartu yang kena tetap diberi label "terpotong".
#
# WAJIB DISEBUT DI NASKAH sebagai penyimpangan protokol Tier 3; nilainya ikut
# tersimpan di tiap rekaman penilaian (sistem.max_tokens).
MAX_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 1
REPEAT_PENALTY = 1.0

# Replika `no_repeat_ngram_size=3` milik eval.py. WAJIB ADA, bukan hiasan:
# llama.cpp hanya menyediakan repeat_penalty (hukuman lunak), sementara eval.py
# MEMBLOKIR keras tiap 3-gram yang berulang. Tanpa ini model kecil terjebak
# mengulang kalimat yang sama sampai jatah token habis — teramati nyata pada
# Model A dan D di 1024 token. Jadi ini memulihkan kesetaraan dengan protokol
# naskah, bukan sekadar menambal gejala.
# 0 = matikan.
NO_REPEAT_NGRAM = 3
SEED = 42

# Harus memuat prompt + MAX_TOKENS. Prompt di sini pendek (system + satu pertanyaan,
# ~150-250 token), jadi 4096 memberi ruang lebih dari cukup dan menutup kemungkinan
# jawaban terpotong karena konteks penuh, bukan karena jatah token. Ongkosnya KV cache
# per model; dengan 6 model menetap, ~2-3 GB di atas 9,3 GB bobot — masih lega di L4 24 GB,
# tetapi di T4 16 GB ini yang pertama akan membuat kehabisan memori.
N_CTX = 4096

# Satu bundel parameter yang dikirim apa adanya ke handler RunPod. Dikumpulkan di sini
# supaya config.py tetap SATU-SATUNYA sumber kebenaran: handler tidak boleh memakai
# nilai bawaannya sendiri, dan nilai yang benar-benar dipakainya dikembalikan lagi
# lewat potongan "siap" untuk dicatat di tiap rekaman penilaian.
PARAMS_GENERASI = {
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repeat_penalty": REPEAT_PENALTY,
    "seed": SEED,
    "no_repeat_ngram": NO_REPEAT_NGRAM,
    "n_ctx": N_CTX,
    "system_prompt": SYSTEM_PROMPT,
}

# ====== tata letak & keenam model menjawab serempak ======
# Mode arena: satu pertanyaan -> KEENAM model menjawab, ditampilkan sebagai kartu.
# Satu giliran saja (tanpa riwayat percakapan), sehingga prompt yang diterima model
# identik dengan protokol eval.py dan Tier 2.
#
# ACAK POSISI. Label A-F sudah diacak terhadap model asli, tetapi bila posisi kartu
# tetap, kartu pertama akan selalu ditempati model yang sama — dan kecenderungan
# menilai kartu awal lebih tinggi akan menguntungkan model itu secara sistematis.
# Karena itu urutan kartu diacak ULANG tiap pertanyaan. Judul kartu tetap menyebut
# labelnya, jadi dokter selalu tahu yang mana.
ACAK_POSISI = True

# Lebar minimum satu kartu sebelum grid melipat ke baris berikutnya. Di laptop
# menghasilkan 3 kolom, di HP otomatis menumpuk satu kolom. Baris skor memakai
# grid yang SAMA, jadi kotak skor selalu sejajar di bawah kartunya.
LEBAR_MIN_KARTU_PX = 320

# Skala penilaian per model. Sengaja TANPA nilai awal di UI: kalau kotak skor sudah
# terisi 3 sejak awal, model yang belum sempat dinilai tetap tercatat 3 dan tidak ada
# jejak bahwa itu bukan penilaian sungguhan. Simpan ditolak selama masih ada yang kosong.
SKALA = (1, 2, 3, 4, 5)

# ====== pertanyaan ======
# Contoh pertanyaan (dari human_eval/gold_set.csv). Dokter DIANJURKAN memakai daftar
# ini supaya semua penilai menghadapi pertanyaan yang sama; decoding greedy + seed
# tetap membuat jawabannya reproducible antar-sesi.
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
