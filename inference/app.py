"""
inference/app.py — Antarmuka Streamlit lokal: Baseline vs Fine-tuned.

Satu model base `unsloth/Qwen3.5-0.8B` dimuat SEKALI; SEMUA adapter LoRA di
`outputs/checkpoints/qwen35-0.8b-train{,/checkpoint-*}` dipasang sebagai adapter
bernama pada base yang sama (hemat VRAM — hanya 1 salinan bobot base ~1.6 GB fp16,
muat di RTX 4050 6 GB). Pemilihan jawaban:
  - BASELINE  = base murni      -> `model.disable_adapter()` (semua adapter OFF)
  - FINETUNED = base + adapter  -> `model.set_adapter(<pilihan>)` lalu generate

Format prompt + parameter generate IDENTIK dengan training/eval (chat_format.py,
eval.py): ChatML, enable_thinking=False, greedy (do_sample=False, no_repeat_ngram_size=3).

Cara jalan:  lihat inference/README_APP.md
    streamlit run inference/app.py

Catatan: file-watcher Streamlit dimatikan via .streamlit/config.toml (mencegah
ImportError transien transformers 5.x). Restart manual bila mengubah kode.
"""
import os
import sys
import time

import streamlit as st
import torch
import transformers  # import eager penuh dulu (hindari race lazy-import 5.x)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- import chat_format.py dari root project -------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from chat_format import MODEL_MODES, SYSTEM_DEFAULT, build_prompt, clean_greeting  # noqa: E402
from inference.typo_postprocess import get_corrector  # noqa: E402  (koreksi typo output — Pivot 6)

BASE_MODEL = "unsloth/Qwen3.5-0.8B"
CKPT_ROOT = os.path.join(ROOT, "outputs", "checkpoints", "qwen35-0.8b-train")
MODE = MODEL_MODES["qwen"]  # Qwen3.5 = ChatML, thinking OFF

st.set_page_config(page_title="Medical Chatbot — Baseline vs Fine-tuned", layout="wide")


# --------------------------------------------------------------------------- #
# Temukan semua adapter yang tersedia (folder berisi adapter_config.json)
# --------------------------------------------------------------------------- #
def discover_adapters():
    found = {}
    # adapter final (best) di root checkpoint -> taruh paling atas
    if os.path.isfile(os.path.join(CKPT_ROOT, "adapter_config.json")):
        found["Final (best, hasil training)"] = CKPT_ROOT
    if os.path.isdir(CKPT_ROOT):
        for d in sorted(os.listdir(CKPT_ROOT)):
            sub = os.path.join(CKPT_ROOT, d)
            if os.path.isfile(os.path.join(sub, "adapter_config.json")):
                found[d] = sub  # mis. "checkpoint-5628"
    return found


ADAPTERS = discover_adapters()  # {label: path}


# --------------------------------------------------------------------------- #
# Load base + SEMUA adapter (sekali, di-cache lintas rerun)
# Key cache = tuple path adapter -> reload hanya jika daftar berubah.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat base + adapter ... (pertama kali bisa lama: unduh base ~1.6 GB)")
def load_engine(adapter_items):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # Tokenizer dari folder adapter pertama (lokal; sudah memuat chat_template hasil training)
    first_path = adapter_items[0][1]
    tok = AutoTokenizer.from_pretrained(first_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # PENTING: JANGAN pakai device_map="auto". accelerate memasang dispatch-hook
    # CPU<->GPU yang bisa crash native (access violation) saat model dimuat dari
    # thread ScriptRunner Streamlit di Windows. Untuk 0.8B cukup .to("cuda") biasa
    # -> model nn.Module murni satu device, tanpa hook, aman di worker-thread.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,                              # transformers 5.x: `dtype` (bukan torch_dtype)
        low_cpu_mem_usage=True,
    )
    if use_cuda:
        base = base.to("cuda")

    model = None
    names = []
    for label, path in adapter_items:
        name = _safe_name(label)
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        names.append(name)
    model.eval()
    return model, tok, names, use_cuda


def _safe_name(label):
    return "".join(c if c.isalnum() else "_" for c in label)


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens):
    enc = tok(text=prompt, return_tensors="pt", padding=True,
              truncation=True, max_length=2048).to(next(model.parameters()).device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        no_repeat_ngram_size=3, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    text = tok.batch_decode(gen, skip_special_tokens=True)[0]
    return text.replace("<think>", "").replace("</think>", "").strip()


def answer_baseline(model, tok, prompt, n):
    with model.disable_adapter():          # semua adapter OFF -> base murni
        return generate(model, tok, prompt, n)


def answer_finetuned(model, tok, prompt, n, adapter_name):
    model.set_adapter(adapter_name)        # aktifkan adapter terpilih
    return generate(model, tok, prompt, n)


import re  # noqa: E402


def prettify(text):
    """Rapikan tampilan SECARA DETERMINISTIK tanpa mengubah/menambah kata.

    Hanya menyusun ulang teks model: pola "<pembuka>: a, b, c" (>=2 koma) -> bullet
    di bawah pembuka tebal; kalimat lain tetap paragraf. AMAN untuk akurasi karena
    isi medis tidak disentuh -- ini kata-kata model itu sendiri, hanya dilayout.
    (Memformat lewat system prompt TIDAK dipakai: mendorong model 0.8B keluar
    distribusi latih -> degenerasi/halusinasi. Lihat README_APP.md.)
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    blocks = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if not s:
            continue
        m = re.match(r"^(.{3,90}?):\s*(.+)$", s)
        if m and m.group(2).count(",") >= 2:
            lead = m.group(1).strip().rstrip(",")
            items = []
            for it in re.split(r",\s*", m.group(2)):
                it = re.sub(r"^(?:atau|dan|serta)\s+", "", it.strip(" .")).strip()
                if it and not re.fullmatch(
                        r"(?:dan|atau)?\s*lain[- ]?(?:lain|nya|sebagainya)?", it, re.I):
                    items.append(it)
            blocks.append(f"**{lead[0].upper() + lead[1:]}:**\n"
                          + "\n".join(f"- {it}" for it in items))
        else:
            blocks.append(s)
    return "\n\n".join(blocks)


def _block(title, fn, apply_clean, apply_pretty, apply_typo, typo_aggressive):
    st.subheader(title)
    t0 = time.time()
    with st.spinner("Menghasilkan jawaban ..."):
        ans = fn()
    if apply_clean:
        ans = clean_greeting(ans)
    n_typo = 0
    if apply_typo:                              # Pivot 6: koreksi typo pada OUTPUT (bukan retrain)
        ans, n_typo = get_corrector().correct_with_count(ans, aggressive=typo_aggressive)
    if apply_pretty:
        ans = prettify(ans)
    st.markdown(ans or "_(kosong)_")
    cap = f"⏱ {time.time() - t0:.1f} s"
    if apply_typo:
        cap += f"  ·  ✍️ {n_typo} typo dikoreksi"
    st.caption(cap)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🩺 Medical Chatbot — Baseline vs Fine-tuned")
st.caption(f"Base: `{BASE_MODEL}`  ·  Pivot 5 single-model Qwen3.5-0.8B (QLoRA)")

if not ADAPTERS:
    st.error(f"Tidak ada adapter ditemukan di `{CKPT_ROOT}`. Pastikan ada `adapter_config.json`.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Pengaturan")

    view = st.radio("Tampilkan", ["Bandingkan (Baseline + Fine-tuned)",
                                   "Fine-tuned saja", "Baseline saja"])

    adapter_label = st.selectbox(
        "Model fine-tuned (checkpoint)", list(ADAPTERS.keys()),
        help="Pilih adapter mana yang dipakai sebagai 'fine-tuned'. "
             "'Final (best)' = hasil load_best_model_at_end.")
    adapter_name = _safe_name(adapter_label)

    max_new_tokens = st.slider("max_new_tokens", 32, 512, 256, 32)
    apply_clean = st.checkbox("Bersihkan sapaan (clean_greeting)", value=True,
                              help="Pasca-proses sama seperti eval.py / deployment.")
    apply_pretty = st.checkbox("Rapikan jadi poin-poin (interaktif)", value=True,
                               help="Hanya melayout ulang teks model jadi bullet — "
                                    "TIDAK mengubah isi, akurasi tetap utuh.")
    apply_typo = st.checkbox("Koreksi typo (KBBI, pasca-proses)", value=True,
                             help="Pivot 6 — perbaiki typo pada JAWABAN model saat runtime "
                                  "(hemat resource: tanpa fine-tuning ulang). Konservatif: "
                                  "istilah medis & nama diri tidak disentuh.")
    typo_aggressive = st.checkbox("↳ mode agresif (edit-1 KBBI)", value=False,
                                  disabled=not apply_typo,
                                  help="Tambah koreksi edit-jarak-1 ke kandidat KBBI tunggal. "
                                       "Recall lebih tinggi, sedikit lebih berisiko.")
    system_prompt = st.text_area("System prompt", SYSTEM_DEFAULT, height=140)

    st.divider()
    # tuple item supaya hashable utk cache_resource
    model, tok, names, use_cuda = load_engine(tuple(ADAPTERS.items()))
    st.success(f"Model siap · device: {'CUDA (GPU)' if use_cuda else 'CPU'} · "
               f"{len(names)} adapter dimuat")

question = st.text_area("Pertanyaan pasien", height=120,
                        placeholder="Contoh: Apa penyebab dan cara mengatasi sakit kepala sebelah?")
go = st.button("Jalankan", type="primary", use_container_width=True)

if go:
    if not question.strip():
        st.warning("Tulis pertanyaan dulu.")
        st.stop()

    prompt = build_prompt(tok, question.strip(), MODE, system=system_prompt.strip())

    if view.startswith("Bandingkan"):
        col_base, col_ft = st.columns(2)
        with col_base:
            _block("📦 Baseline (pre-trained)",
                   lambda: answer_baseline(model, tok, prompt, max_new_tokens),
                   apply_clean, apply_pretty, apply_typo, typo_aggressive)
        with col_ft:
            _block(f"✨ Fine-tuned · {adapter_label}",
                   lambda: answer_finetuned(model, tok, prompt, max_new_tokens, adapter_name),
                   apply_clean, apply_pretty, apply_typo, typo_aggressive)
    elif view == "Fine-tuned saja":
        _block(f"✨ Fine-tuned · {adapter_label}",
               lambda: answer_finetuned(model, tok, prompt, max_new_tokens, adapter_name),
               apply_clean, apply_pretty, apply_typo, typo_aggressive)
    else:  # Baseline saja
        _block("📦 Baseline (pre-trained)",
               lambda: answer_baseline(model, tok, prompt, max_new_tokens),
               apply_clean, apply_pretty, apply_typo, typo_aggressive)
