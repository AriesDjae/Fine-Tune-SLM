"""
merge_and_export.py — Merge LoRA adapter ke base model, simpan ke outputs/merged/

Run setelah training selesai:
  python training/merge_and_export.py --model qwen25-1b5
  python training/merge_and_export.py --model smollm2-1b7
  python training/merge_and_export.py --model gemma3-4b

Output:
  outputs/merged/{model_name}-medical/   <- full merged model (siap konversi ke GGUF)
"""

import argparse
from pathlib import Path

MODEL_MAP = {
    "qwen35-2b": {
        "base_id":      "Qwen/Qwen3.5-2B",
        "adapter_dir":  "outputs/checkpoints/qwen35-2b-medical",
        "merged_dir":   "outputs/merged/qwen35-2b-medical",
    },
    "smollm3-3b": {
        "base_id":      "HuggingFaceTB/SmolLM3-3B",
        "adapter_dir":  "outputs/checkpoints/smollm3-3b-medical",
        "merged_dir":   "outputs/merged/smollm3-3b-medical",
    },
    "gemma3-4b": {
        "base_id":      "google/gemma-3-4b-it",
        "adapter_dir":  "outputs/checkpoints/gemma3-4b-medical",
        "merged_dir":   "outputs/merged/gemma3-4b-medical",
    },
}

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["qwen35-2b", "smollm3-3b", "gemma3-4b"],
                    help="Model key to merge")
args = parser.parse_args()

cfg = MODEL_MAP[args.model]
adapter_path = ROOT / cfg["adapter_dir"]
merged_path  = ROOT / cfg["merged_dir"]

print(f"Merging: {args.model}")
print(f"  Base   : {cfg['base_id']}")
print(f"  Adapter: {adapter_path}")
print(f"  Output : {merged_path}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("\nLoading base model (fp16, no quantization)...")
base_model = AutoModelForCausalLM.from_pretrained(
    cfg["base_id"],
    torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, str(adapter_path))

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {merged_path} ...")
merged_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(merged_path), safe_serialization=True)
tokenizer.save_pretrained(str(merged_path))

print(f"\nDone. Merged model saved to {merged_path}")
print("Next step: bash scripts/convert_to_gguf.sh")
