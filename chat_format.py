"""
chat_format.py  —  SUMBER KEBENARAN tunggal untuk format prompt.

Konsistensi prompt = *silent killer*: format chat template saat TRAINING, EVALUASI,
dan DEPLOYMENT (Termux/llama.cpp) HARUS IDENTIK. Modul ini dipakai oleh eval.py,
benchmark_ondevice.py, dan export_gguf.py. Notebook training meng-inline logika yang
SAMA (lihat sel `build_prompt()`), jadi ketiga tahap memakai format identik.

Perbedaan antar-model (ditentukan MODEL_MODES):
  - Qwen3.5-2B  : ChatML (`<|im_start|>`); role `system` native; thinking OFF (enable_thinking=False).
  - Gemma 4 E2B : turn-token `<|turn>role ... <turn|>`; role `system` NATIVE (baru di Gemma 4,
                  tidak seperti Gemma 3n) -> tak ada lagi penggabungan system->user; thinking OFF.

clean_greeting(): post-processing sapaan/basa-basi yang DIPAKAI BERSAMA oleh
preprocessing (Bagian 1.3, sbg pass akhir), eval.py (jaring pengaman output), dan
deployment (Termux). Karena train == eval == deploy memakai cleaner yang SAMA,
sapaan residual tidak pernah sampai ke user. Pola dirancang AMAN dari over-strip
(pakai \b dan lookahead terbatas -> tidak merusak kata medis: "Hippocrates",
"Hidradenitis", "Alopecia", "Halothane" TIDAK ikut terpotong).
"""

import re

# Pola sapaan/basa-basi pembuka. Urutan penting: greeting umum dulu, lalu glued.
_GREETING_PATTERNS = [
    re.compile(r"^@\w+[\s,:]*"),                                          # @tina,
    re.compile(r"^(?:hai|halo|alo|hi|hello|hallo|hey)\b[\s,!.:;-]*", re.I),  # Hai/Hello + pemisah (\b cegah "Hippocrates")
    re.compile(r"^(?:hello|halo|hallo)(?=(?:thanks|thank|terima|hai|dear))", re.I),  # HelloThanks (nempel, AMAN)
    re.compile(r"^(?:salam)\b[\s,!.:;-]*(?:sehat|sejahtera|kenal)?[\s,!.:;-]*", re.I),  # Salam, / Salam sehat
    re.compile(r"^(?:selamat\s+(?:pagi|siang|sore|malam|datang))[\s,!.:;-]*\w*[\s,!.:;-]*", re.I),
    re.compile(r"^(?:hi|hello|hey)\s+there\b[\s,!.:;-]*", re.I),          # Hi there
    re.compile(r"^(?:terima\s*kasih|terimakasih|thanks|thank\s*you)"
               r"[\s,!.]*(?:atas|telah|for)?[^.!?]*[.!?]\s*", re.I),       # "Terima kasih telah bertanya ... ."
    re.compile(r"^(?:dear)\b[\s,!.:;-]*\w*[\s,!.:;-]*", re.I),            # Dear [nama],
]


def clean_greeting(text, min_keep=15):
    """Buang sapaan/basa-basi pembuka secara AMAN (multi-pass utk sapaan bertumpuk).

    Mengembalikan teks asli (utuh) bila hasil pembersihan < `min_keep` karakter,
    sebagai jaring pengaman agar konten valid tidak dikosongkan/over-strip.
    """
    original = (text or "").strip()
    t = original
    for _ in range(4):                       # multi-pass: sapaan bisa bertumpuk
        prev = t
        for pat in _GREETING_PATTERNS:
            t = pat.sub("", t).strip()
        if t == prev:
            break
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t if len(t.strip()) >= min_keep else original


# enable_thinking=True artinya "model PUNYA switch thinking -> kirim enable_thinking=False
# saat render" (kedua model Qwen3.5 & Gemma 4 punya mode thinking yang kita MATIKAN).
# instruction_part/response_part = penanda turn untuk train_on_responses_only (loss hanya
# di jawaban) -- diverifikasi dari notebook/doc resmi Unsloth (Qwen3.5 ChatML, Gemma 4 <|turn>).
MODEL_MODES = {
    "qwen":  dict(merge_system=False, enable_thinking=True, special="<|im_start|>",
                  instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"),
    "gemma": dict(merge_system=False, enable_thinking=True, special="<|turn>",
                  instruction_part="<|turn>user\n", response_part="<|turn>model\n"),
}

SYSTEM_DEFAULT = (
    "You are a helpful medical assistant. Answer patient questions with accurate, "
    "empathetic responses based on established clinical knowledge. Always recommend "
    "consulting a healthcare professional."
)


def to_model_messages(messages, merge_system):
    """Sesuaikan daftar messages dengan kemampuan model.

    Jika merge_system=True, isi pesan `system` digabung ke AWAL pesan user pertama
    (untuk model tanpa turn `system`, mis. Gemma 3n lama). Pivot Gemma 4: model kini
    punya role `system` native -> merge_system=False untuk KEDUA model (no-op di sini).
    """
    if not merge_system:
        return messages
    sys_txt, out, injected = "", [], False
    for m in messages:
        if m["role"] == "system":
            sys_txt = m["content"]
            continue
        if m["role"] == "user" and not injected and sys_txt:
            out.append({"role": "user", "content": sys_txt.strip() + "\n\n" + m["content"]})
            injected = True
        else:
            out.append(m)
    if sys_txt and not injected:
        out.insert(0, {"role": "user", "content": sys_txt})
    return out


def render_chat(tokenizer, messages, mode, add_generation_prompt=False):
    """Render daftar messages -> string ChatML/Gemma sesuai `mode` (dict dari MODEL_MODES)."""
    msgs = to_model_messages(messages, mode["merge_system"])
    kw = {"enable_thinking": False} if mode["enable_thinking"] else {}
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)
    except TypeError:  # template tidak menerima enable_thinking
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def build_prompt(tokenizer, question, mode, system=SYSTEM_DEFAULT):
    """INFERENSI/DEPLOY: bangun prompt sampai giliran assistant (add_generation_prompt=True)."""
    return render_chat(
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        mode, add_generation_prompt=True,
    )


def split_prompt_reference(sample, mode, tokenizer):
    """Dari satu sample test (punya messages system+user+assistant), kembalikan
    (prompt_string, reference_answer). Prompt memakai system+user SAMPLE ITU SENDIRI
    (konsisten dgn cara sample itu dilatih); reference = isi assistant."""
    sys_user = [m for m in sample["messages"] if m["role"] in ("system", "user")]
    ref = next((m["content"] for m in sample["messages"] if m["role"] == "assistant"), "")
    prompt = render_chat(tokenizer, sys_user, mode, add_generation_prompt=True)
    return prompt, ref
