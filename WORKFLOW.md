# Alur Kerja Detail: Fine-Tune SLM untuk Medical Chatbot

> Koreksi dan update checklist sesuai progress aktual.

---

## Struktur Project

```
Fine-Tune SLM for Medical Chatbot/
├── .env/                              ← virtual environment (sudah ada)
├── Data/                              ← data sumber (sudah ada)
│   ├── SimpleTabulation-ICD-11-MMS-en.xlsx   ← 35,664 kode ICD-11 (sudah ada)
│   ├── hidr_indicators.xlsx                  ← 4,038 indikator kesehatan WHO (sudah ada)
│   ├── 9789240077263-eng.pdf                 ← referensi WHO (sudah ada)
│   ├── ICD-Classification & Behavioural.pdf  ← referensi ICD (sudah ada)
│   ├── refguide.pdf                          ← referensi (sudah ada)
│   └── data.py                               ← script download HuggingFace (sudah ada)
├── data/                              ← data terstruktur (perlu dibuat)
│   ├── bioasq/
│   │   └── BioASQ-train-factoid-4b.json      ← perlu download dari bioasq.org
│   └── ppk_kemenkes/
│       └── ppk_qa_pairs.json                 ← dibuat manual dari PPK Kemenkes
├── notebooks/
│   ├── 00_dataset_preparation.ipynb
│   ├── 01_finetune_qwen35_2b.ipynb
│   ├── 02_finetune_smollm3_3b.ipynb
│   └── 03_finetune_gemma3_4b.ipynb
├── outputs/
│   ├── checkpoints/                   ← adapter LoRA hasil training
│   └── gguf/                          ← model final siap deploy
└── scripts/
    └── convert_to_gguf.sh
```

---

## PHASE 0 — Verifikasi Environment

### 0.1 Aktifkan Virtual Environment
```bash
cd "D:/Project/Machine_Learning/Fine-Tune SLM for Medical Chatbot"
.env/Scripts/activate
```

### 0.2 Verifikasi Packages
```bash
pip install -r requirements.txt
pip list | grep -E "torch|transformers|peft|trl|bitsandbytes|accelerate|openpyxl"
```

### 0.3 Cek GPU
```python
import torch
print(torch.cuda.get_device_name(0))
print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## PHASE 1 — Inventarisasi Data yang Tersedia

### 1.1 Data yang Sudah Ada di `Data/`

| File | Isi | Jumlah Baris | Kolom Kunci |
|------|-----|-------------|-------------|
| `SimpleTabulation-ICD-11-MMS-en.xlsx` | Klasifikasi ICD-11 MMS | **35,664** (berkode) | `Code`, `Title`, `ClassKind`, `ChapterNo`, `Grouping1–5`, `CodingNote` |
| `hidr_indicators.xlsx` | Indikator kesehatan WHO | **4,038** | `topic_area`, `indicator_name`, `dataset_name`, `dimension` |

**ICD-11 — contoh baris dengan kode**:
- `Code: 1A00`, `Title: Cholera`, `ClassKind: category`, `ChapterNo: 1`
- `Code: 5A10`, `Title: Type 1 diabetes mellitus`, `ChapterNo: 5`

**HIDR — contoh topic area**:
- Burden of disease (1,166 baris)
- Disability (570 baris)
- HIV, tuberculosis and malaria (130 baris)
- Reproductive, maternal and child health (113 baris)

### 1.2 Data yang Perlu Disiapkan

| File | Cara Mendapat | Estimasi Jumlah |
|------|--------------|-----------------|
| `data/bioasq/BioASQ-train-factoid-4b.json` | Download dari [bioasq.org](http://bioasq.org) (perlu registrasi) | ~1,000 QA factoid |
| `data/ppk_kemenkes/ppk_qa_pairs.json` | Buat manual dari PDF PPK Kemenkes | ~300–500 QA |

### 1.3 Data HuggingFace via `Data/data.py`

File `Data/data.py` sudah menyiapkan download — **semua dipakai**:

| Dataset HuggingFace | ID | Estimasi QA |
|--------------------|----|------------|
| PubMedQA | `qiaojin/PubMedQA` (pqa_labeled) | ~1,000 berlabel |
| MedQuAD | `lavita/MedQuAD` | ~47,000 |
| WikiDoc Medical | `medalpaca/medical_meadow_wikidoc` | ~10,000 |
| MedMCQA | `openlifescienceai/medmcqa` (5,000 sample) | ~5,000 |
| MedFit | `mlx-community/medfit-dataset` | cek saat runtime |

**Download**:
```bash
python Data/data.py
```
Cache tersimpan di `Data/hf_cache/` (otomatis, tidak download ulang jika sudah ada).

---

## PHASE 2 — Persiapan Data Tambahan

### 2.1 BioASQ (Jika Belum Ada)

1. Registrasi di http://bioasq.org
2. Download `BioASQ-train-factoid-4b.json`
3. Simpan ke `data/bioasq/`

**Struktur JSON BioASQ**:
```json
{
  "questions": [
    {
      "id": "...",
      "type": "factoid",
      "body": "What is the...",
      "ideal_answer": ["..."],
      "exact_answer": [["..."]]
    }
  ]
}
```
Tipe yang dipakai: `factoid`, `yesno`, `list`

### 2.2 PPK Kemenkes (Dibuat Manual)

Format target `data/ppk_kemenkes/ppk_qa_pairs.json`:
```json
[
  {
    "question": "Apa tata laksana lini pertama untuk hipertensi grade 1?",
    "answer": "Modifikasi gaya hidup selama 3-6 bulan. Jika tidak ada perbaikan, berikan ACE inhibitor atau ARB."
  }
]
```
Sumber: Buku PPK Kemenkes (ekstrak manual dari PDF)
Target topik: hipertensi, diabetes, TB, ISPA, demam berdarah

---

## PHASE 3 — Persiapan Dataset (Notebook 00)

### 3.1 Buka `notebooks/00_dataset_preparation.ipynb`

**Tujuan**: Konversi semua data ke format ChatML → gabung → split → simpan JSONL.

### 3.2 Konversi ICD-11 (`Data/SimpleTabulation-ICD-11-MMS-en.xlsx`)

Dari 35,664 baris berkode, generate **3 variasi QA per baris** (hanya baris `isLeaf=True` atau yang punya `CodingNote`):

```python
import pandas as pd
df = pd.read_excel("Data/SimpleTabulation-ICD-11-MMS-en.xlsx")
df_coded = df[df["Code"].notna() & (df["ClassKind"] == "category")]

# Variasi 1: Kode → Nama
# Q: "What is the ICD-11 code for {Title}?"
# A: "The ICD-11 code for {Title} is {Code}."

# Variasi 2: Nama → Deskripsi (jika CodingNote tidak kosong)
# Q: "Describe {Title} in ICD-11 classification."
# A: "{CodingNote}"

# Variasi 3: Kode → Kategori
# Q: "What chapter does ICD-11 code {Code} belong to?"
# A: "ICD-11 code {Code} ({Title}) belongs to Chapter {ChapterNo}, group {Grouping1}."
```

**Estimasi output**: ~35,000–70,000 QA pairs dari ICD-11.

### 3.3 Konversi HIDR (`Data/hidr_indicators.xlsx`)

Dari 4,038 baris indikator, generate QA tentang indikator kesehatan:

```python
df = pd.read_excel("Data/hidr_indicators.xlsx")

# Variasi 1:
# Q: "What does the indicator {indicator_abbr} measure?"
# A: "{indicator_name} — measured across dimension: {dimension}."

# Variasi 2:
# Q: "Which dataset covers {indicator_name}?"
# A: "The {dataset_name} dataset (ID: {dataset_id}) covers {indicator_name}."

# Variasi 3:
# Q: "What topic area does {indicator_name} fall under in WHO HIDR?"
# A: "{indicator_name} falls under the '{topic_area}' topic area."
```

**Estimasi output**: ~8,000–12,000 QA pairs dari HIDR.

### 3.4 Konversi BioASQ

```python
import json
with open("data/bioasq/BioASQ-train-factoid-4b.json") as f:
    data = json.load(f)

for q in data["questions"]:
    if q["type"] in ["factoid", "yesno"]:
        question = q["body"]
        answer = q["ideal_answer"][0] if isinstance(q["ideal_answer"], list) else q["ideal_answer"]
        # → format ChatML
```

### 3.5 Konversi PPK Kemenkes

```python
with open("data/ppk_kemenkes/ppk_qa_pairs.json") as f:
    data = json.load(f)
# Langsung ke format ChatML (sudah berupa QA pairs)
```

### 3.6 Konversi HuggingFace Datasets

**PubMedQA** (`qiaojin/PubMedQA`, split `train`):
```python
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir="Data/hf_cache")
# Kolom: pubid, question, context, long_answer, final_decision (yes/no/maybe)
# Q: question
# A: final_decision + ": " + long_answer (truncate 400 token)
```

**MedQuAD** (`lavita/MedQuAD`):
```python
ds = load_dataset("lavita/MedQuAD", cache_dir="Data/hf_cache")
# Kolom: question, answer
# Skip baris jika answer == None atau answer == ""
# Q: question → A: answer
```

**WikiDoc Medical** (`medalpaca/medical_meadow_wikidoc`):
```python
ds = load_dataset("medalpaca/medical_meadow_wikidoc", cache_dir="Data/hf_cache")
# Kolom: instruction, input, output
# Q: instruction (+ input jika tidak kosong) → A: output
```

**MedMCQA** (`openlifescienceai/medmcqa`, 5,000 sample):
```python
ds = load_dataset("openlifescienceai/medmcqa", split="train[:5000]", cache_dir="Data/hf_cache")
# Kolom: question, opa, opb, opc, opd, cop (0-3), exp
# Q: "Question: {question}\nA) {opa}\nB) {opb}\nC) {opc}\nD) {opd}"
# A: "The correct answer is {['A','B','C','D'][cop]}) {[opa,opb,opc,opd][cop]}. {exp}"
```

**MedFit** (`mlx-community/medfit-dataset`):
```python
ds = load_dataset("mlx-community/medfit-dataset", cache_dir="Data/hf_cache")
# Cek struktur kolom dulu: print(ds["train"].features)
# Sesuaikan konversi setelah melihat kolom aktual
```

### 3.7 Format ChatML Output

Semua sample dikonversi ke:
```
<|im_start|>system
You are a helpful medical assistant. Answer questions accurately based on clinical guidelines and ICD-11 classification.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>
```

### 3.8 Ringkasan Dataset Gabungan (Estimasi)

| Sumber | Bahasa | Estimasi QA |
|--------|--------|------------|
| ICD-11 Excel | Inggris | ~35,000–70,000 |
| HIDR Indicators | Inggris | ~8,000–12,000 |
| PubMedQA | Inggris | ~1,000 |
| MedQuAD | Inggris | ~47,000 |
| WikiDoc Medical | Inggris | ~10,000 |
| MedMCQA | Inggris | ~5,000 |
| MedFit | Inggris | cek runtime |
| BioASQ | Inggris | ~1,000 |
| PPK Kemenkes | Indonesia | ~300–500 |
| **Total estimasi** | | **~107,000–146,000+** |

### 3.8 Split & Simpan

```
data/processed/
├── train.jsonl      ← 85%
├── val.jsonl        ← 10%
└── test.jsonl       ← 5%
```

**Validasi sebelum lanjut**:
- [ ] Tidak ada `text` kosong
- [ ] Semua ChatML token ada (`<|im_start|>`, `<|im_end|>`)
- [ ] Panjang token rata-rata < 1024
- [ ] Split benar: cek `len(train) / len(total) ≈ 0.85`

---

## PHASE 4 — Fine-Tuning Model

### 4.1 Urutan Training (Terkecil ke Terbesar)

1. `01_finetune_qwen35_2b.ipynb` → Qwen3.5-2B
2. `02_finetune_smollm3_3b.ipynb` → SmolLM3-3B
3. `03_finetune_gemma3_4b.ipynb` → Gemma3-4B

### 4.2 Konfigurasi QLoRA (Sama untuk Semua)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

### 4.3 Konfigurasi per Model

| Parameter | Qwen3.5-2B | SmolLM3-3B | Gemma3-4B |
|-----------|-----------|-----------|----------|
| `model_id` | *(konfirmasi HuggingFace ID)* | *(konfirmasi HuggingFace ID)* | `google/gemma-3-4b-it` |
| `max_seq_length` | 2048 | 2048 | 2048 |
| `batch_size` | 2 | 2 | 2 |
| `gradient_accumulation` | 8 | 8 | 8 |
| `lora_r` | 16 | 16 | 16 |
| `lora_alpha` | 32 | 32 | 32 |
| `learning_rate` | 2e-4 | 2e-4 | 2e-4 |
| `epochs` | 3 | 3 | 3 |

### 4.4 Output Checkpoint

```
outputs/checkpoints/
├── qwen35-2b-medical/      ← adapter LoRA saja
├── smollm3-3b-medical/
└── gemma3-4b-medical/
```

### 4.5 Monitoring

```bash
# Terminal terpisah
nvidia-smi -l 3
```

**Tanda training sehat**:
- Loss epoch 1: ~2.0–2.5 → epoch 3: < 1.5
- VRAM stabil (tidak naik terus)

**Jika OOM**: turunkan `batch_size=1`, `max_seq_length=512`, naikkan `gradient_accumulation=16`

---

## PHASE 5 — Evaluasi (Di Dalam Notebook Training)

| Metrik | Target |
|--------|--------|
| Validation Loss | < 1.5 |
| ROUGE-L (test.jsonl) | > 0.30 |
| Qualitative test 10 soal | Jawaban koheren & akurat |

**10 pertanyaan test wajib** (campuran ICD-11, klinis, Indonesia):
1. What is the ICD-11 code for Type 1 diabetes mellitus?
2. What chapter does cholera belong to in ICD-11?
3. What does the indicator "covid_cfr" measure in WHO HIDR?
4. What is the first-line treatment for hypertension?
5. What are the symptoms of tuberculosis?
6. Apa tanda-tanda demam berdarah dengue?
7. Apa tata laksana ISPA pada anak?
8. What topic area does "HIV prevalence" fall under in WHO HIDR?
9. What ICD-11 code is used for influenza?
10. Describe Type 2 diabetes mellitus in ICD-11 classification.

---

## PHASE 6 — Konversi ke GGUF

### 6.1 Merge LoRA → Full Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("model_id")
model = PeftModel.from_pretrained(base, "outputs/checkpoints/qwen35-2b-medical")
merged = model.merge_and_unload()
merged.save_pretrained("outputs/merged/qwen35-2b-medical")
```

### 6.2 Konversi GGUF

```bash
bash scripts/convert_to_gguf.sh qwen35-2b-medical
```

**Isi `scripts/convert_to_gguf.sh`** (perlu dibuat):
```bash
#!/bin/bash
MODEL=$1
python llama.cpp/convert_hf_to_gguf.py \
    outputs/merged/${MODEL} \
    --outfile outputs/gguf/${MODEL}.gguf \
    --outtype q4_k_m
```

**Output**:
```
outputs/gguf/
├── qwen35-2b-medical.gguf
├── smollm3-3b-medical.gguf
└── gemma3-4b-medical.gguf
```

### 6.3 Test GGUF

```bash
./llama.cpp/llama-cli -m outputs/gguf/qwen35-2b-medical.gguf \
    -p "What is the ICD-11 code for Type 2 diabetes?" -n 100
```

---

## Checklist Eksekusi

### Phase 0 — Environment
- [ ] `.env` aktif, `torch.cuda.is_available()` = True
- [ ] `pip install -r requirements.txt` selesai, termasuk `openpyxl`

### Phase 1 — Data
- [ ] `Data/SimpleTabulation-ICD-11-MMS-en.xlsx` — **sudah ada** (35,664 baris)
- [ ] `Data/hidr_indicators.xlsx` — **sudah ada** (4,038 baris)
- [ ] `data/bioasq/BioASQ-train-factoid-4b.json` — perlu download
- [ ] `data/ppk_kemenkes/ppk_qa_pairs.json` — perlu dibuat manual

### Phase 2 — Notebook 00
- [ ] Konversi ICD-11 → QA pairs selesai
- [ ] Konversi HIDR → QA pairs selesai
- [ ] Konversi BioASQ selesai (jika sudah ada)
- [ ] Konversi PPK Kemenkes selesai (jika sudah ada)
- [ ] `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` tersimpan
- [ ] Validasi format passed

### Phase 3 — Training
- [ ] Notebook 01: Qwen3.5-2B → `outputs/checkpoints/qwen35-2b-medical/`
- [ ] Notebook 02: SmolLM3-3B → `outputs/checkpoints/smollm3-3b-medical/`
- [ ] Notebook 03: Gemma3-4B → `outputs/checkpoints/gemma3-4b-medical/`

### Phase 4 — Evaluasi
- [ ] Loss < 1.5, ROUGE-L > 0.30 untuk semua model
- [ ] Qualitative test 10 soal: jawaban koheren
- [ ] Model terbaik dipilih

### Phase 5 — GGUF
- [ ] llama.cpp ter-clone dan ter-build
- [ ] `scripts/convert_to_gguf.sh` dibuat
- [ ] Semua model ter-convert ke `.gguf`
- [ ] Test llama.cpp CLI berhasil

---

## Risiko & Mitigasi

| Risiko | Kemungkinan | Mitigasi |
|--------|-------------|----------|
| OOM saat training | Sedang | Turunkan `batch_size=1`, `max_seq_length=1024`, naikkan `gradient_accumulation=16` |
| bitsandbytes error Windows | Sedang | Install wheel Windows dari GitHub releases |
| ICD-11 QA terlalu repetitif | Sedang | Filter hanya baris `isLeaf=True` dan punya `CodingNote` |
| HIDR kolom tidak cukup informatif | Rendah | Gunakan hanya `topic_area` + `indicator_name` |
| Model ID Qwen3.5/SmolLM3 salah | Sedang | Konfirmasi nama repo di HuggingFace sebelum training |
