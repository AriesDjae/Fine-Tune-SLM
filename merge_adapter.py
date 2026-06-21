"""
merge_adapter.py  —  PIVOT 5: gabung adapter QLoRA Qwen3.5-0.8B -> model merged 16-bit.

Diperlukan SEBELUM export GGUF (export_gguf.py butuh direktori merged 16-bit).
Untuk eval 16-bit TIDAK perlu merge: eval.py bisa langsung memuat direktori adapter
(Unsloth FastLanguageModel auto-load base + adapter).

Jalur (otomatis pilih yang tersedia):
  1) Unsloth  -> SAMA dgn env training (disarankan; menangani arsitektur Qwen3.5).
  2) PEFT     -> fallback transformers+peft murni (butuh transformers yg sudah
                 mendukung arsitektur Qwen3.5).

Contoh:
  python merge_adapter.py \
      --adapter outputs/checkpoints/qwen35-0.8b-train \
      --out     outputs/merged/qwen35-0.8b-medical

Lalu:  python export_gguf.py --merged_dir outputs/merged/qwen35-0.8b-medical --verify
"""
import argparse
import os
from pathlib import Path

BASE_DEFAULT = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 1024


def merge_unsloth(adapter, base, out, max_seq_length):
    from unsloth import FastLanguageModel
    print(f"[unsloth] load adapter: {adapter}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter, max_seq_length=max_seq_length,
        load_in_4bit=False, dtype=None)
    print(f"[unsloth] save merged 16-bit -> {out}")
    model.save_pretrained_merged(out, tok, save_method="merged_16bit")


def merge_peft(adapter, base, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    # base bisa diambil dari adapter_config bila tak diberikan
    cfg = Path(adapter) / "adapter_config.json"
    if base is None and cfg.exists():
        import json
        base = json.load(open(cfg, encoding="utf-8")).get("base_model_name_or_path")
    print(f"[peft] base={base}  adapter={adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    print("[peft] merge_and_unload ...")
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(adapter)
    print(f"[peft] save merged 16-bit -> {out}")
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/checkpoints/qwen35-0.8b-train",
                    help="direktori adapter QLoRA (root atau checkpoint-XXXX)")
    ap.add_argument("--out", default="outputs/merged/qwen35-0.8b-medical")
    ap.add_argument("--base", default=None,
                    help=f"override base model (default dari adapter_config / {BASE_DEFAULT})")
    ap.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument("--backend", choices=["auto", "unsloth", "peft"], default="auto")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def has(mod):
        import importlib.util
        return importlib.util.find_spec(mod) is not None

    backend = args.backend
    if backend == "auto":
        backend = "unsloth" if has("unsloth") else "peft"
    print(f"backend = {backend}")

    if backend == "unsloth":
        merge_unsloth(args.adapter, args.base or BASE_DEFAULT, args.out, args.max_seq_length)
    else:
        merge_peft(args.adapter, args.base, args.out)

    print("\nSELESAI. Lanjut export GGUF:")
    print(f"  python export_gguf.py --merged_dir {args.out} --verify")


if __name__ == "__main__":
    main()
