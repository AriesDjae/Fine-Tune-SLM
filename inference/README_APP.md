# Antarmuka Lokal — Baseline vs Fine-tuned (Streamlit)

App `inference/app.py` menampilkan **dua jawaban berdampingan** dari satu pertanyaan:
- **Baseline** = `unsloth/Qwen3.5-0.8B` murni (pre-trained, tanpa adapter)
- **Fine-tuned** = base + adapter LoRA hasil training (`outputs/checkpoints/qwen35-0.8b-train`)

Hemat VRAM: base dimuat **sekali**, baseline pakai `disable_adapter()`, fine-tuned pakai adapter aktif.
Hanya ~1.6 GB fp16 di GPU → muat di RTX 4050 6 GB. Format prompt & parameter generate identik dengan `eval.py`/`chat_format.py`.

---

## 1. Prasyarat

- Adapter lokal sudah ada di `outputs/checkpoints/qwen35-0.8b-train/` (✅ sudah).
- **Internet saat pertama jalan**: base `unsloth/Qwen3.5-0.8B` (~1.6 GB) diunduh dari HuggingFace
  dan di-cache (`~/.cache/huggingface`). Run berikutnya tidak perlu internet lagi.
- GPU opsional. Tanpa GPU otomatis jatuh ke CPU (lebih lambat, tetap jalan).

## 2. Install dependency

Pakai venv GPU yang sudah ada (`.venv-gpu`). `torch` + `transformers` sudah terpasang;
tinggal tambah `streamlit`, `peft`, `accelerate`:

```powershell
# dari root project: "Fine-Tune SLM for Medical Chatbot"
.\.venv-gpu\Scripts\Activate.ps1
pip install streamlit peft accelerate
```

> Jika `peft` sudah ada, perintah di atas tetap aman (skip yang terpasang).

## 3. Jalankan

```powershell
streamlit run inference/app.py
```

> **Browser TIDAK terbuka otomatis** (mode `headless = true`, lihat `.streamlit/config.toml`).
> Buka manual di browser: **http://localhost:8501**
>
> Mode headless ini WAJIB: pada mode non-headless, Streamlit menjalankan skrip di thread
> event-loop utama saat startup → init CUDA bentrok dengan asyncio uvicorn → **SEGFAULT**
> saat load model. Headless membuat skrip hanya jalan lewat sesi browser (thread terpisah) → aman.
Ketik pertanyaan pasien → klik **"Bandingkan jawaban"** → jawaban Baseline (kiri) & Fine-tuned (kanan) muncul.

Hentikan dengan `Ctrl+C` di terminal.

## 4. Pengaturan (sidebar)

| Opsi | Default | Fungsi |
|------|---------|--------|
| **Tampilkan** | Bandingkan | Pilih: **Bandingkan** (2 kolom), **Fine-tuned saja**, atau **Baseline saja**. |
| **Model fine-tuned (checkpoint)** | Final (best) | Pilih adapter mana sbg 'fine-tuned': `Final (best)`, `checkpoint-5628`, `checkpoint-5600`. Semua dimuat sekali ke base yg sama. |
| `max_new_tokens` | 256 | Panjang maksimum jawaban (sama dgn protokol eval). |
| Bersihkan sapaan | ✅ | Pasca-proses `clean_greeting()` (buang basa-basi pembuka), sama spt eval/deploy. |
| **Rapikan jadi poin-poin** | ✅ | Layout teks model jadi bullet (interaktif). **Hanya tampilan — isi/akurasi tidak diubah.** |
| System prompt | `SYSTEM_DEFAULT` | Bisa diubah; default = persona medical assistant dari `chat_format.py`. |

Selektor checkpoint otomatis menemukan semua folder ber-`adapter_config.json` di
`outputs/checkpoints/qwen35-0.8b-train/`. Baseline selalu = base murni (`disable_adapter()`).

Decoding = **greedy** (`do_sample=False`, `no_repeat_ngram_size=3`) agar hasil deterministik & adil
membandingkan kedua versi (identik dengan `eval.py`).

### Kenapa kerapian dibuat di tampilan, bukan di prompt
Model fine-tuned dilatih pada jawaban dokter yang **mengalir** (Alodokter-style). Memaksa format
terstruktur lewat system prompt (mis. "jawab pakai poin-poin") mendorong model 0.8B **keluar
distribusi latih** → output degenerasi/halusinasi (mis. "hindarkan… hindra… hindrn…"). Karena
**akurasi prioritas #1**, kerapian dilakukan oleh `prettify()` (post-processing deterministik) yang
hanya **melayout ulang kata-kata model** jadi bullet — nol perubahan isi. Matikan via checkbox bila
ingin teks mentah.

## 5. Catatan & troubleshooting

- **Pertama jalan lama / "Memuat model ..."** — itu proses unduh base + load bobot; tunggu sampai
  sidebar muncul "Model siap". Model di-cache (`@st.cache_resource`) jadi rerun berikutnya instan.
- **CUDA out of memory** — tutup app lain yang pakai GPU, atau turunkan `max_new_tokens`.
  Bila tetap, paksa CPU: set `CUDA_VISIBLE_DEVICES=""` sebelum `streamlit run` (lambat tapi aman).
- **Offline** — setelah base terunduh sekali, set `HF_HUB_OFFLINE=1` agar tidak mencoba konek HF.
- **Versi resmi (single-model Pivot 5)** — baseline & fine-tuned di sini memakai base id `unsloth/Qwen3.5-0.8B`
  sesuai `adapter_config.json`. Bila nanti ada `outputs/merged/` atau GGUF Q4, app bisa diarahkan ke sana.
- **`ImportError: cannot import name 'AutoModelForCausalLM'`** (transformers 5.x) — disebabkan race
  file-watcher Streamlit. Sudah ditangani: `.streamlit/config.toml` mematikan watcher
  (`fileWatcherType = "none"`). Konsekuensinya **tidak ada auto-reload** — bila mengubah `app.py`,
  hentikan (`Ctrl+C`) lalu `streamlit run` ulang.
- **Warning `fast path is not available ... flash-linear-attention / causal-conv1d`** — TIDAK masalah,
  hanya fallback ke implementasi torch biasa untuk layer linear-attention Qwen3.5. Output tetap benar.
- **Server mati / Segmentation fault / "disconnect" / `Failed to fetch dynamically imported module ... Button.js`** —
  prosesnya segfault saat load model di Windows. DUA penyebab, keduanya sudah ditangani:
  1. `device_map="auto"` (accelerate) memasang dispatch-hook CPU↔GPU → diganti `.to("cuda")` biasa.
  2. **Mode non-headless** menjalankan skrip di thread event-loop utama saat startup → CUDA bentrok asyncio →
     SEGFAULT. Diperbaiki dengan `headless = true` di `.streamlit/config.toml`.
  Error `Failed to fetch ... Button.js` di browser hanya GEJALA server mati (filenya ADA). Jika masih muncul
  setelah perbaikan, itu sisa cache: **hard refresh** (`Ctrl+Shift+R`) atau tab incognito.
