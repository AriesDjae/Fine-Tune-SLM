"""
serverless/train_p6.py — training headless Pivot 6 (scaling Qwen3.5 0.8B/2B/4B)
untuk RunPod SERVERLESS. Konversi 1:1 dari notebooks/build_train_pivot6.py
(urutan sel: ENV -> CONFIG -> PATHS -> LOAD -> PEFT -> DATA -> SELF-CHECK ->
TRAIN -> GATE -> SAVE -> quick-EVAL). Hyperparameter & GATE TIDAK diubah.

Dipanggil oleh serverless/handler.py. Bisa juga dijalankan manual (pod/lokal GPU):
    python serverless/train_p6.py --model 0.8b --mode pilot

Penyimpanan (auto-deteksi):
  1. /runpod-volume ADA   -> checkpoint/hasil/cache-HF persisten di volume (resume aman).
  2. TANPA volume         -> /tmp (hilang saat worker mati) + set HF_OUT_REPO + HF_TOKEN
                             agar adapter & hasil di-push ke HF Hub sebelum job selesai.

Dataset (urutan prioritas): env DATA_DIR -> /runpod-volume/processed_id_final ->
  /workspace/processed_id_final -> Data/processed_id_final -> unduh dari HF_DATA_REPO
  (dataset repo privat HF; butuh HF_TOKEN).

Beda sadar dari notebook (didokumentasikan):
  - ADAPTER_DIR pilot diberi sufiks "_pilot" agar checkpoint pilot TIDAK ke-resume
    oleh run full (di notebook keduanya satu folder — rawan resume salah).
  - GATE me-RETURN dict verdict (bukan cuma print) karena serverless tanpa layar.
"""
import argparse
import glob
import json
import math
import os
import random
import shutil


# ====================== IDENTITAS MODEL (dari build_train_pivot6.py) ======================
MODELS = {
    "0.8b": dict(
        label="Qwen3.5-0.8B",
        model_id="unsloth/Qwen3.5-0.8B",
        model_id_fallback="Qwen/Qwen3.5-0.8B",  # resmi Qwen (ungated). "-Instruct" TIDAK ADA di HF
        loader="FastModel",                   # config repo kini multimodal (vision_config, cek 2026-07-23)
        adapter_name="qwen35_0.8b_p6",        # -> FastModel + vision beku, seragam dgn 2B/4B
    ),
    "2b": dict(
        label="Qwen3.5-2B",
        model_id="unsloth/Qwen3.5-2B",
        model_id_fallback="Qwen/Qwen3.5-2B",  # fallback resmi Qwen (ungated)
        loader="FastModel",                   # MULTIMODAL -> FastModel, text-only (vision beku)
        adapter_name="qwen35_2b_p6",
    ),
    "4b": dict(
        label="Qwen3.5-4B",
        model_id="unsloth/Qwen3.5-4B",
        model_id_fallback="Qwen/Qwen3.5-4B",
        loader="FastModel",
        adapter_name="qwen35_4b_p6",
    ),
}

# ====================== FORMAT TURN (identik ketiga model — ChatML Qwen) ======================
# Template Qwen sisip <think>\n\n</think> kosong walau enable_thinking=False
# -> RESPONSE_PART memuat scaffold itu agar loss hanya di jawaban (train==infer).
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXPECTED_PROMPT_TAIL = "</think>"
MERGE_SYSTEM_INTO_USER = False               # Qwen: system native

# ====================== HYPERPARAMETER (dikunci — komparabilitas antar model) ======================
SEED = 42
MAX_SEQ_LENGTH = 1024                        # p99 antar-model ~931 -> aman
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 1e-4                         # dikunci (2e-4 terbukti terlalu agresif)
EPOCHS = 3                                   # full-mode; + EarlyStopping(patience=3)
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
PILOT_TRAIN_N, PILOT_VAL_N, PILOT_MAX_STEPS = 1500, 200, 250


def _eager_generation():
    """Context manager: paksa EAGER hanya selama generate.
    Decode GatedDeltaNet Qwen3.5 memicu recompile torch.compile per step
    (FailOnRecompileLimitHit). set_stance("force_eager") TERBUKTI TAK MEMPAN
    di worker (job pilot 2B c99173ab, 2026-07-23: tetap crash di generasi ke-4)
    -> pakai kill-switch dynamo global torch._dynamo.config.disable, dipulihkan
    setelah generate. Training sebelum/sesudahnya tetap compiled (cepat)."""
    import torch
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        prev = torch._dynamo.config.disable
        torch._dynamo.config.disable = True      # dynamo off total -> eager murni
        try:
            yield
        finally:
            torch._dynamo.config.disable = prev
    return _ctx()


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(os.path.join(p, "train.jsonl")):
            return p
    return None


def _resolve_data_dir(progress):
    candidates = [
        os.environ.get("DATA_DIR", ""),
        "/runpod-volume/processed_id_final",
        "/runpod-volume/Data/processed_id_final",
        "/workspace/processed_id_final",
        "/workspace/Data/processed_id_final",
        "../Data/processed_id_final", "Data/processed_id_final",
    ]
    d = _first_existing(candidates)
    if d:
        return d
    # Mode tanpa volume: unduh dari dataset repo privat HF (sekali per worker).
    repo = os.environ.get("HF_DATA_REPO", "")
    if repo:
        progress(f"dataset tidak ditemukan lokal -> unduh dari HF dataset {repo}")
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir="/tmp/processed_id_final")
        if _first_existing([d]):
            return d
    raise FileNotFoundError(
        "Dataset tak ketemu. Upload train/val/test.jsonl (processed_id_final) ke "
        "network volume (processed_id_final/), ATAU set env HF_DATA_REPO + HF_TOKEN "
        "(dataset repo privat HF). Lihat RUN_SERVERLESS.md.")


def _push_to_hf(local_dirs, path_prefix, progress):
    """Push folder hasil ke HF Hub (repo privat). No-op bila env tak diset."""
    repo = os.environ.get("HF_OUT_REPO", "")
    if not repo or not os.environ.get("HF_TOKEN"):
        return None
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    for local, sub in local_dirs:
        if os.path.isdir(local):
            progress(f"push {local} -> hf://{repo}/{path_prefix}/{sub}")
            api.upload_folder(folder_path=local, repo_id=repo,
                              path_in_repo=f"{path_prefix}/{sub}",
                              ignore_patterns=["checkpoint-*", "*.pt", "optimizer*"])
    return repo


def run(model_key="0.8b", run_mode="pilot", load_in_4bit=False,
        quick_eval=None, use_compile=True, progress=None):
    """Jalankan satu leg training. Return dict ringkasan (gate, loss, path)."""
    assert model_key in MODELS, f"model harus salah satu {list(MODELS)}"
    assert run_mode in ("pilot", "full"), "mode harus 'pilot' atau 'full'"
    if quick_eval is None:
        quick_eval = (run_mode == "full")
    M = MODELS[model_key]
    prog = progress or (lambda msg: print(f"[progress] {msg}", flush=True))

    # ============ ENV (sel 2) ============
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # single GPU
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Anti-fragmentasi CUDA (disarankan pesan error OOM job 16a4ecda) — gratis, aman.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not use_compile:
        # Saklar darurat {"compile": false}: matikan compile Unsloth TOTAL (eager penuh).
        # Default = compile ON utk kecepatan training di H100; fase GENERATE selalu
        # dipaksa eager via _eager_generation() (lihat bawah) karena linear attention
        # Qwen3.5 (GatedDeltaNet) recompile tiap step decode -> FailOnRecompileLimitHit
        # (job 58dbcc7d, 2026-07-23).
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    on_volume = os.path.isdir("/runpod-volume")
    if on_volume:
        os.environ.setdefault("HF_HOME", "/runpod-volume/hf_cache")  # cache model persisten
    OUT_ROOT = "/runpod-volume" if on_volume else "/tmp/p6_out"

    # FAIL-FAST storage (pelajaran biaya 2026-07-23): run full tanpa volume DAN tanpa
    # HF_OUT_REPO+HF_TOKEN = hasil PASTI hilang -> gagalkan SEBELUM detik GPU terbayar,
    # jangan cuma warning di akhir seperti dulu.
    if run_mode == "full" and not on_volume and not (
            os.environ.get("HF_OUT_REPO") and os.environ.get("HF_TOKEN")):
        raise AssertionError(
            "Storage tak aman utk mode full: /runpod-volume tidak terpasang dan "
            "HF_OUT_REPO/HF_TOKEN tidak diset -> adapter akan hilang saat worker mati. "
            "Pasang network volume di endpoint ATAU set env HF_OUT_REPO + HF_TOKEN.")

    # Import unsloth PERTAMA (mem-patch transformers/trl).
    from unsloth import FastLanguageModel, FastModel  # noqa: F401
    import numpy as np
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch._dynamo.config.recompile_limit = 64
    prog(f"start {M['label']} mode={run_mode} 4bit={load_in_4bit} | "
         f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
         f"| storage={'volume' if on_volume else 'EPHEMERAL /tmp'}")

    # Effective batch DIKUNCI 16 di kedua mode (komparabilitas antar-run/antar-model).
    if load_in_4bit:                    # GPU 24GB: batch kecil + akumulasi
        PER_DEVICE_BATCH, GRAD_ACCUM = 2, 8
    else:                               # H100 80GB bf16: chunk terbesar, tanpa akumulasi
        # (16x1 == 8x2 secara matematis: eff batch, step count, LR schedule identik —
        #  tapi GPU dapat kerja 2x lebih besar per forward -> utilisasi naik.)
        PER_DEVICE_BATCH, GRAD_ACCUM = 16, 1

    # ============ PATHS (sel 4) ============
    DATA_DIR = _resolve_data_dir(prog)
    suffix = "_pilot" if run_mode == "pilot" else ""
    ADAPTER_DIR = os.path.join(OUT_ROOT, "checkpoints", M["adapter_name"] + suffix)
    RESULTS_DIR = os.path.join(OUT_ROOT, "results", M["adapter_name"] + suffix)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prog(f"DATA_DIR={DATA_DIR} | ADAPTER_DIR={ADAPTER_DIR}")

    # ============ LOAD (sel 5) ============
    Loader = FastLanguageModel if M["loader"] == "FastLanguageModel" else FastModel
    forced_4bit_warning = False

    def _load(mid):
        nonlocal forced_4bit_warning
        # Checkpoint "-bnb-4bit" pre-quantized -> WAJIB load_in_4bit=True walau mode bf16.
        four = load_in_4bit or ("bnb-4bit" in mid)
        if four and not load_in_4bit:
            forced_4bit_warning = True
            print(f"[WARN] {mid} pre-quantized -> dipaksa QLoRA 4-bit (bukan bf16 murni). "
                  "Jangan campur run ini dgn run bf16 di tabel jurnal.", flush=True)
        return Loader.from_pretrained(
            model_name=mid,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,                 # auto: bf16 di Hopper (H100)
            load_in_4bit=four,
        )

    try:
        model, tokenizer = _load(M["model_id"])
        loaded_id = M["model_id"]
    except Exception as e_primary:
        print("Gagal:", repr(e_primary)[:300], "-> fallback", M["model_id_fallback"], flush=True)
        try:
            model, tokenizer = _load(M["model_id_fallback"])
            loaded_id = M["model_id_fallback"]
        except Exception as e_fb:
            # Laporkan KEDUA error — error primary-lah yang biasanya informatif
            # (error fallback saja pernah menyesatkan diagnosa, 2026-07-23).
            raise RuntimeError(
                "Gagal load model di KEDUA id.\n"
                f"- PRIMARY  {M['model_id']}: {repr(e_primary)[:800]}\n"
                f"- FALLBACK {M['model_id_fallback']}: {repr(e_fb)[:400]}") from e_fb

    _q = bool(getattr(model, "is_loaded_in_4bit", False))
    precision = "QLoRA 4-bit (bnb)" if _q else f"LoRA {next(model.parameters()).dtype}"
    prog(f"loaded {loaded_id} | precision: {precision}")
    assert _q == load_in_4bit or not load_in_4bit, "Mode 4-bit diminta tapi model tidak ter-quantize?"

    # VLM (Qwen3.5-2B/4B via FastModel) bisa mengembalikan PROCESSOR, bukan tokenizer murni.
    TOK = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(TOK, "pad_token", None) is None:
        TOK.pad_token = TOK.eos_token
    TOK.padding_side = "right"
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    PAD_ID = TOK.pad_token_id
    is_processor = TOK is not tokenizer
    prog(f"loader={M['loader']} | vocab={len(TOK)} | pad={TOK.pad_token} | processor?={is_processor}")

    # ============ PEFT (sel 6) ============
    if M["loader"] == "FastLanguageModel":
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R, target_modules=LORA_TARGET,
            lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            use_gradient_checkpointing="unsloth", random_state=SEED,
        )
    else:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,     # TEXT-ONLY: bekukan menara vision
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            use_gradient_checkpointing="unsloth", random_state=SEED,
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.print_trainable_parameters()        # angka utk tabel jurnal

    # ============ DATA (sel 7) ============
    from datasets import load_dataset

    def to_model_messages(messages):
        if not MERGE_SYSTEM_INTO_USER:        # Qwen: system native -> no-op
            return messages
        sys_txt, out, injected = "", [], False
        for m in messages:
            if m["role"] == "system":
                sys_txt = m["content"]; continue
            if m["role"] == "user" and not injected and sys_txt:
                out.append({"role": "user", "content": sys_txt.strip() + "\n\n" + m["content"]})
                injected = True
            else:
                out.append(m)
        if sys_txt and not injected:
            out.insert(0, {"role": "user", "content": sys_txt})
        return out

    def render_chat(messages, add_generation_prompt=False):
        msgs = to_model_messages(messages)
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False,
                add_generation_prompt=add_generation_prompt, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

    def formatting_func(ex):
        return {"text": render_chat(ex["messages"], add_generation_prompt=False)}

    train_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"), split="train")
    if run_mode == "pilot":
        train_ds = train_ds.shuffle(seed=SEED).select(range(min(PILOT_TRAIN_N, len(train_ds))))
        val_ds = val_ds.shuffle(seed=SEED).select(range(min(PILOT_VAL_N, len(val_ds))))
    train_ds = train_ds.map(formatting_func, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(formatting_func, remove_columns=val_ds.column_names)
    prog(f"data: train={len(train_ds)} val={len(val_ds)}")

    # ============ SELF-CHECK decode (sel 8 — WAJIB hijau sebelum train) ============
    def _find_sub(seq, sub):
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i + len(sub)] == sub:
                return i
        return -1

    _rec = json.loads(open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8").readline())
    _msgs = _rec["messages"]
    _full = render_chat(_msgs, add_generation_prompt=False)
    _prompt = render_chat([m for m in _msgs if m["role"] != "assistant"], add_generation_prompt=True)
    _ids = TOK(_full, add_special_tokens=False)["input_ids"]
    _marker = TOK(RESPONSE_PART, add_special_tokens=False)["input_ids"]
    _pos = _find_sub(_ids, _marker)
    if _pos < 0:
        raise AssertionError(
            "SELF-CHECK GAGAL: RESPONSE_PART tak ditemukan di teks ter-render "
            f"(template {M['label']} beda dugaan). Rendered tail: {_full[:400]!r}")
    _start = _pos + len(_marker)
    _ans = next(m["content"] for m in _msgs if m["role"] == "assistant")
    _w1 = _ans.split()[0] if _ans.split() else ""
    _reco = TOK.decode(_ids[_start:_start + 12]).lstrip()
    if not _reco.lower().startswith(_w1[:6].lower()):
        raise AssertionError(
            f"SELF-CHECK GAGAL (off-by-one): kata-1 jawaban {_w1!r} vs rekonstruksi {_reco[:40]!r}")
    if not _prompt.rstrip().endswith(EXPECTED_PROMPT_TAIL):
        raise AssertionError(
            f"SELF-CHECK GAGAL: prompt inferensi tak berakhir {EXPECTED_PROMPT_TAIL!r} "
            f"-> train != infer. Tail aktual: {_prompt.rstrip()[-60:]!r}")
    prog("SELF-CHECK decode: OK (jawaban tak ke-mask, train==infer)")

    # ============ TRAIN CFG (sel 9) ============
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    USE_BF16 = torch.cuda.is_bf16_supported()
    if run_mode == "pilot":
        steps_kw = dict(max_steps=PILOT_MAX_STEPS, num_train_epochs=1,
                        eval_steps=50, save_steps=50)
    else:
        steps_per_epoch = max(1, math.ceil(len(train_ds) / (PER_DEVICE_BATCH * GRAD_ACCUM)))
        es = max(10, steps_per_epoch // 10)   # ~10x/epoch
        steps_kw = dict(num_train_epochs=EPOCHS, eval_steps=es, save_steps=es)

    cfg = SFTConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        # Eval batch JANGAN dibesarkan: accelerate meng-upcast logits [B, seq, vocab~250k]
        # ke fp32 per forward -> batch 64 minta ~51 GiB sekaligus = OOM di H100 80GB
        # (job 16a4ecda, 2026-07-23). Batch 16 ~ 13 GiB fp32 logits, aman.
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        optim="adamw_8bit",
        fp16=not USE_BF16,
        bf16=USE_BF16,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED, data_seed=SEED,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,                        # WAJIB utk train_on_responses_only
        dataloader_num_workers=8,             # GPU jangan menunggu CPU tunggal (util 20%)
        dataloader_pin_memory=True,
        report_to="none",
        **steps_kw,
    )

    class _ProgressCB(TrainerCallback):      # jembatan ke runpod progress_update
        def on_evaluate(self, args, state, control, metrics=None, **kw):
            try:
                prog(f"step {state.global_step}: eval_loss={metrics.get('eval_loss', float('nan')):.4f}")
            except Exception:
                pass

    _cb = [_ProgressCB()] if run_mode == "pilot" else [EarlyStoppingCallback(3, 0.001), _ProgressCB()]
    _kw = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=cfg, callbacks=_cb)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **_kw)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **_kw)
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    prog(f"trainer siap | steps_kw={steps_kw} | eff_batch={PER_DEVICE_BATCH * GRAD_ACCUM}")

    # ============ TRAIN (sel 10 — resume-aware) ============
    _ckpts = sorted(glob.glob(os.path.join(ADAPTER_DIR, "checkpoint-*")),
                    key=lambda p: int(p.rsplit("-", 1)[-1]))
    prog(f"resume={bool(_ckpts)} ({len(_ckpts)} checkpoint)")
    # Resume tahan-banting: checkpoint TERBARU bisa korup (worker mati saat save)
    # -> resume crash -> submit ulang crash lagi = loop bayar-gagal. Solusi: buang
    # checkpoint korup, mundur ke sebelumnya (JANGAN dari nol). Error training asli
    # (sudah ada progress melewati step checkpoint) tetap di-raise apa adanya.
    for _ in range(len(_ckpts) + 1):
        try:
            trainer.train(resume_from_checkpoint=bool(_ckpts) or None)
            break
        except Exception as e:
            _resumed_step = int(_ckpts[-1].rsplit("-", 1)[-1]) if _ckpts else 0
            if not _ckpts or trainer.state.global_step > _resumed_step:
                raise                      # bukan masalah checkpoint -> laporkan asli
            bad = _ckpts.pop()
            prog(f"[WARN] resume gagal ({type(e).__name__}: {str(e)[:120]}) -> "
                 f"hapus {os.path.basename(bad)}, coba checkpoint sebelumnya")
            shutil.rmtree(bad, ignore_errors=True)
    prog("training selesai")

    result = {
        "model": M["label"], "loaded_id": loaded_id, "mode": run_mode,
        "unsloth_compile": use_compile,
        "precision": precision, "forced_4bit_fallback": forced_4bit_warning,
        "trainable_params": trainable, "total_params": total,
        "eff_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "adapter_dir": ADAPTER_DIR, "results_dir": RESULTS_DIR,
        "storage": "volume" if on_volume else "ephemeral+hf_push",
    }

    # ============ GATE (sel 11 — hanya pilot) ============
    if run_mode == "pilot":
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 42
        logs = trainer.state.log_history
        losses = [l["loss"] for l in logs if "loss" in l]

        def _mean(x):
            return sum(x) / len(x) if x else float("nan")

        k = max(1, len(losses) // 5)
        loss_start, loss_end = _mean(losses[:k]), _mean(losses[-k:])
        loss_max = max(losses) if losses else float("nan")
        B2_trend = (loss_end <= loss_start - 0.15)
        B2_noexplode = (loss_max <= 2 * losses[0]) if losses else False
        B1_stable = all(np.isfinite(l) for l in losses) and len(losses) > 0

        # Generate DIBUNGKUS try/except: crash di sini (mis. recompile GatedDeltaNet,
        # job 58dbcc7d) TIDAK boleh membakar hasil training — adapter tetap di-SAVE
        # + di-push di bawah; verdict jadi STOP dgn alasan tercatat.
        gen_error = None
        Loader.for_inference(model)
        test_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "test.jsonl"), split="train")
        gens, id_ok, degen = [], 0, 0
        try:
            with _eager_generation():
                for i in range(10):
                    msgs = [m for m in test_ds[i]["messages"] if m["role"] != "assistant"]
                    p = render_chat(msgs, add_generation_prompt=True)
                    inp = TOK(p, return_tensors="pt").to(model.device)
                    out = model.generate(**inp, max_new_tokens=200, do_sample=False,
                                         no_repeat_ngram_size=3, repetition_penalty=1.1,
                                         pad_token_id=PAD_ID)
                    txt = TOK.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    gens.append((msgs[-1]["content"], txt))
                    try:
                        if detect(txt[:500]) == "id":
                            id_ok += 1
                    except Exception:
                        pass
                    toks = txt.split()
                    grams = [tuple(toks[j:j + 4]) for j in range(len(toks) - 3)]
                    rep = (1 - len(set(grams)) / len(grams)) if grams else 0
                    if rep > 0.30:
                        degen += 1
        except Exception as e:
            gen_error = f"{type(e).__name__}: {str(e)[:300]}"
            prog(f"[WARN] generate GATE crash: {gen_error} -> verdict STOP, adapter tetap disimpan")
        B3a_lang = id_ok >= 9 and gen_error is None
        B3b_degen = degen == 0 and gen_error is None
        Loader.for_training(model)

        gen_file = os.path.join(RESULTS_DIR, "pilot_generations.txt")
        with open(gen_file, "w", encoding="utf-8") as f:
            for q, a in gens:
                f.write(f"Q: {q}\nA: {a}\n{'-' * 60}\n")

        PASS = B1_stable and B2_trend and B2_noexplode and B3a_lang and B3b_degen
        result["gate"] = {
            "verdict": "PASS_GREEN" if PASS else "STOP",
            "generate_error": gen_error,
            "B1_stable_no_nan": B1_stable,
            "B2_loss_start": round(loss_start, 4), "B2_loss_end": round(loss_end, 4),
            "B2_trend_drop_ge_0.15": B2_trend, "B2_no_explode": B2_noexplode,
            "B3a_indonesian_10": id_ok, "B3a_pass": B3a_lang,
            "B3b_degenerate_10": degen, "B3b_pass": B3b_degen,
            "generations_file": gen_file,
            "sample_generations": [{"q": q[:120], "a": a[:200]} for q, a in gens[:3]],
        }
        prog(f"GATE: {result['gate']['verdict']}")

    # ============ SAVE (sel 12) ============
    model.save_pretrained(ADAPTER_DIR)
    try:
        tokenizer.save_pretrained(ADAPTER_DIR)
    except Exception:
        TOK.save_pretrained(ADAPTER_DIR)
    with open(os.path.join(RESULTS_DIR, "log_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RESULTS_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "gate"}, f, ensure_ascii=False, indent=1)
    # trainer_state best checkpoint utk kurva jurnal
    if trainer.state.best_model_checkpoint:
        src = os.path.join(trainer.state.best_model_checkpoint, "trainer_state.json")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(RESULTS_DIR, "trainer_state_best.json"))

    # ============ quick-EVAL (sel 13 — token-F1 + ROUGE-L pada val, n<=100) ============
    if quick_eval:
        import collections
        from rouge_score import rouge_scorer

        def _f1(pred, ref):
            p, r = pred.split(), ref.split()
            common = collections.Counter(p) & collections.Counter(r)
            ns = sum(common.values())
            if ns == 0 or not p or not r:
                return 0.0
            prec, rec = ns / len(p), ns / len(r)
            return 2 * prec * rec / (prec + rec)

        Loader.for_inference(model)
        val_eval = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"), split="train")
        N = min(100, len(val_eval))
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        f1s, rls = [], []
        # Dibungkus try/except: quick-eval crash TIDAK boleh menggagalkan push hasil
        # training full run (pola rugi yang sama dgn GATE — lihat komentar sel 11).
        try:
            with _eager_generation():
                for i in range(N):
                    msgs = val_eval[i]["messages"]
                    ref = next(m["content"] for m in msgs if m["role"] == "assistant")
                    p = render_chat([m for m in msgs if m["role"] != "assistant"], add_generation_prompt=True)
                    inp = TOK(p, return_tensors="pt").to(model.device)
                    out = model.generate(**inp, max_new_tokens=256, do_sample=False,
                                         no_repeat_ngram_size=3, repetition_penalty=1.1,
                                         pad_token_id=PAD_ID)
                    pred = TOK.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    f1s.append(_f1(pred, ref))
                    rls.append(scorer.score(ref, pred)["rougeL"].fmeasure)
            result["quick_eval"] = {"n": N, "token_f1": round(sum(f1s) / N, 4),
                                    "rougeL": round(sum(rls) / N, 4)}
            prog(f"quick-eval n={N}: F1={result['quick_eval']['token_f1']} "
                 f"RL={result['quick_eval']['rougeL']}")
        except Exception as e:
            done = len(rls)
            result["quick_eval"] = {
                "error": f"{type(e).__name__}: {str(e)[:300]}", "n_done": done,
                "token_f1_partial": round(sum(f1s) / done, 4) if done else None,
                "rougeL_partial": round(sum(rls) / done, 4) if done else None}
            prog(f"[WARN] quick-eval crash setelah {done} sampel -> hasil training tetap di-push")

    # ============ PUSH ke HF Hub (wajib di mode tanpa volume) ============
    pushed = _push_to_hf(
        [(ADAPTER_DIR, "adapter"), (RESULTS_DIR, "results")],
        path_prefix=f"{M['adapter_name']}{suffix}", progress=prog)
    result["hf_out_repo"] = pushed
    if not on_volume and not pushed:
        result["warning"] = ("TANPA volume & TANPA HF_OUT_REPO: adapter/hasil hilang saat "
                             "worker mati! Set HF_OUT_REPO + HF_TOKEN di env endpoint.")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Training headless Pivot 6 (RunPod serverless/pod)")
    ap.add_argument("--model", default="0.8b", choices=list(MODELS))
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--quick-eval", action="store_true", default=None)
    a = ap.parse_args()
    out = run(model_key=a.model, run_mode=a.mode,
              load_in_4bit=a.load_in_4bit, quick_eval=a.quick_eval)
    print(json.dumps(out, ensure_ascii=False, indent=1))
