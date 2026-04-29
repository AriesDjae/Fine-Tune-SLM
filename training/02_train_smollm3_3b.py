"""
Training script: SmolLM3-3B — Medical Chatbot Fine-tuning
QLoRA 4-bit, LoRA r=16, 3 epochs — GPU sewa (A100/H100 class)

Run:
  python training/02_train_smollm3_3b.py

Output:
  outputs/checkpoints/smollm3-3b-medical/   <- LoRA adapter
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
MODEL_ID       = "HuggingFaceTB/SmolLM3-3B"
CHECKPOINT_DIR = str(ROOT / "outputs" / "checkpoints" / "smollm3-3b-medical")
TRAIN_FILE     = str(ROOT / "Data" / "processed" / "train.jsonl")
VAL_FILE       = str(ROOT / "Data" / "processed" / "val.jsonl")
TEST_FILE      = str(ROOT / "Data" / "processed" / "test.jsonl")

MAX_SEQ_LENGTH     = 2048
BATCH_SIZE         = 4
GRAD_ACCUM         = 4        # effective batch = 16
LEARNING_RATE      = 2e-4
EPOCHS             = 3
WARMUP_RATIO       = 0.05
LORA_R             = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
EVAL_STEPS         = 500
SAVE_STEPS         = 500
ROUGE_EVAL_SAMPLES = 100

# ──────────────────────────────────────────
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Model: {MODEL_ID}\n")

# ── Quantization ──────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Tokenizer ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Model ─────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# ── LoRA ──────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ───────────────────────────────
def apply_chat_template(examples):
    return {"text": [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]}

print("Loading datasets...")
train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
val_ds   = load_dataset("json", data_files=VAL_FILE,   split="train")
test_ds  = load_dataset("json", data_files=TEST_FILE,  split="train")

train_ds = train_ds.map(apply_chat_template, batched=True, remove_columns=["messages"])
val_ds   = val_ds.map(apply_chat_template,   batched=True, remove_columns=["messages"])
test_ds  = test_ds.map(apply_chat_template,  batched=True, remove_columns=["messages"])
print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}\n")

# ── Training ──────────────────────────────
training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    gradient_checkpointing=False,
    optim="adamw_torch_fused",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

# ── Save adapter ──────────────────────────
trainer.save_model(CHECKPOINT_DIR)
tokenizer.save_pretrained(CHECKPOINT_DIR)
print(f"\nAdapter saved to {CHECKPOINT_DIR}")

# ── ROUGE evaluation ──────────────────────
print("\nRunning ROUGE evaluation on test set...")
try:
    import evaluate
    rouge = evaluate.load("rouge")

    _ASSISTANT_MARKERS = [
        "<|im_start|>assistant\n",
        "<start_of_turn>model\n",
        "ASSISTANT:\n",
    ]
    _END_TOKENS = ["<|im_end|>", "<end_of_turn>", "</s>", "<|endoftext|>"]

    sample = test_ds.select(range(min(ROUGE_EVAL_SAMPLES, len(test_ds))))
    model.eval()
    predictions, references = [], []

    for item in sample:
        text = item["text"]
        prompt_part = reference = None
        for marker in _ASSISTANT_MARKERS:
            if marker in text:
                prompt_part = text.rsplit(marker, 1)[0] + marker
                reference   = text.rsplit(marker, 1)[1]
                for tok in _END_TOKENS:
                    reference = reference.replace(tok, "")
                reference = reference.strip()
                break
        if not prompt_part:
            continue

        inputs = tokenizer(prompt_part, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        predictions.append(generated)
        references.append(reference)

    results = rouge.compute(predictions=predictions, references=references)
    print(f"\nROUGE Results ({len(predictions)} samples):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    with open(Path(CHECKPOINT_DIR) / "eval_results.json", "w") as f:
        json.dump({"rouge": results, "n_samples": len(predictions)}, f, indent=2)
    print(f"Results saved to {CHECKPOINT_DIR}/eval_results.json")

except Exception as e:
    print(f"ROUGE eval skipped: {e}")

print("\nDone.")
