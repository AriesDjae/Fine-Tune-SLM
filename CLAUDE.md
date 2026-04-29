# Fine-Tune SLM for Medical Chatbot

## Project Overview
Fine-tuning Small Language Model (SLM) untuk Medical Chatbot yang mampu menjelaskan keluhan pasien secara detail.
**Patokan lengkap ada di `WORKFLOW.md`** — CLAUDE.md ini adalah ringkasan referensi cepat.

---

## Dataset

### Data Lokal (`Data/`)
| File | Isi | Jumlah | Status |
|------|-----|--------|--------|
| `SimpleTabulation-ICD-11-MMS-en.xlsx` | Klasifikasi ICD-11 MMS | 35,664 berkode | sudah ada |
| `hidr_indicators.xlsx` | Indikator kesehatan WHO | 4,038 baris | sudah ada |
| `indonesia-medical-qna/qna.csv` | QnA pasien-dokter Alodokter | 681k baris | sudah ada |
| `training13b.json` + `13B[1-4]_golden.json` | BioASQ biomedical QA | 5,729 | sudah ada |
| `indonesia-bioner/` | NER biomedis Indonesia | — | **skip** (bukan QA) |

### Data yang Perlu Disiapkan
| File | Cara | Estimasi |
|------|------|---------|
| `data/bioasq/BioASQ-train-factoid-4b.json` | Download dari bioasq.org (perlu registrasi) | ~1,000 QA |
| `data/ppk_kemenkes/ppk_qa_pairs.json` | Buat manual dari PDF PPK Kemenkes | ~300–500 QA |

### HuggingFace Datasets (`Data/data.py`)
| Dataset | HF ID | Estimasi QA |
|---------|-------|------------|
| PubMedQA | `qiaojin/PubMedQA` | ~1,000 |
| MedQuAD | `lavita/MedQuAD` | ~47,000 |
| WikiDoc Medical | `medalpaca/medical_meadow_wikidoc` | ~10,000 |
| MedMCQA | `openlifescienceai/medmcqa` (5k sample) | ~5,000 |
| MedFit | `mlx-community/medfit-dataset` | cek runtime |
| ChatDoctor | `lavita/medical-qa-datasets` (chatdoctor_healthcaremagic) | cek runtime |

### Estimasi Dataset Gabungan
~107,000–146,000+ samples → split `data/processed/train.jsonl` (85%) / `val.jsonl` (10%) / `test.jsonl` (5%)

---

## Format Dataset
ChatML format:
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

---

## Models & Konfigurasi (Identik untuk Semua)

| Notebook | Model |
|----------|-------|
| `01_finetune_qwen35_2b.ipynb` | Qwen3.5-2B |
| `02_finetune_smollm3_3b.ipynb` | SmolLM3-3B |
| `03_finetune_gemma3_4b.ipynb` | Gemma3-4B (`google/gemma-3-4b-it`) |

**Konfigurasi QLoRA (sama semua)**:
- Quantization: 4-bit NF4, double quant, compute dtype bfloat16
- `max_seq_length`: 2048
- `batch_size`: 2 | `gradient_accumulation`: 8
- `lora_r`: 16 | `lora_alpha`: 32
- `learning_rate`: 2e-4 | `epochs`: 3

---

## Struktur File
```
├── Data/                              # data sumber mentah
│   ├── hf_cache/                      # cache HuggingFace datasets
│   └── data.py                        # script download HF datasets
├── data/                              # data terstruktur hasil proses
│   ├── bioasq/
│   ├── ppk_kemenkes/
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── notebooks/
│   ├── 00_dataset_preparation.ipynb   # konversi semua sumber → JSONL
│   ├── 01_finetune_qwen35_2b.ipynb
│   ├── 02_finetune_smollm3_3b.ipynb
│   └── 03_finetune_gemma3_4b.ipynb
├── outputs/
│   ├── checkpoints/                   # adapter LoRA hasil training
│   │   ├── qwen35-2b-medical/
│   │   ├── smollm3-3b-medical/
│   │   └── gemma3-4b-medical/
│   └── gguf/                          # model final siap deploy
│       ├── qwen35-2b-medical.gguf
│       ├── smollm3-3b-medical.gguf
│       └── gemma3-4b-medical.gguf
├── preprocessing/                     # script bantu (opsional, di luar notebook)
│   ├── process_bioasq.py
│   ├── process_indonesia_qna.py
│   ├── process_icd11.py
│   └── build_dataset.py
└── scripts/
    └── convert_to_gguf.sh
```

---

## Evaluasi
| Metrik | Target |
|--------|--------|
| Validation Loss | < 1.5 |
| ROUGE-L (test.jsonl) | > 0.30 |
| Qualitative test 10 soal | Jawaban koheren & akurat |

---

## Progress

### Phase 0 — Environment
- [ ] `.env` aktif, `torch.cuda.is_available()` = True
- [ ] `pip install -r requirements.txt` selesai

### Phase 1 — Data
- [x] `Data/SimpleTabulation-ICD-11-MMS-en.xlsx` — sudah ada
- [x] `Data/hidr_indicators.xlsx` — sudah ada
- [x] `Data/training13b.json` + `13B[1-4]_golden.json` — sudah ada
- [x] `Data/indonesia-medical-qna/qna.csv` — sudah ada
- [ ] `data/bioasq/BioASQ-train-factoid-4b.json` — perlu download
- [ ] `data/ppk_kemenkes/ppk_qa_pairs.json` — perlu dibuat manual
- [ ] `python Data/data.py` — download HuggingFace datasets

### Phase 2 — Dataset Preparation (Notebook 00)
- [ ] Konversi semua sumber → ChatML JSONL
- [ ] `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` tersimpan
- [ ] Validasi format passed

### Phase 3 — Training
- [ ] Notebook 01: Qwen3.5-2B → `outputs/checkpoints/qwen35-2b-medical/`
- [ ] Notebook 02: SmolLM3-3B → `outputs/checkpoints/smollm3-3b-medical/`
- [ ] Notebook 03: Gemma3-4B → `outputs/checkpoints/gemma3-4b-medical/`

### Phase 4 — Evaluasi
- [ ] Loss < 1.5, ROUGE-L > 0.30 untuk semua model
- [ ] Qualitative test 10 soal koheren
- [ ] Model terbaik dipilih

### Phase 5 — GGUF
- [ ] llama.cpp ter-clone dan ter-build
- [ ] `scripts/convert_to_gguf.sh` dibuat
- [ ] Semua model ter-convert ke `.gguf`
