"""
scripts/upload_dataset_hf.py — upload Data/processed_id_final/{train,val,test}.jsonl
ke dataset repo PRIVAT di HuggingFace Hub (Plan B: serverless TANPA network volume).

Dataset ini legal-cleared utk riset (DOI 10.57967/hf/8356 + UU Psl 44) tapi TETAP
private — worker mengunduhnya dgn HF_TOKEN yang sama.

Setup sekali (PowerShell):
    pip install huggingface_hub
    $env:HF_TOKEN = "hf_..."     # token WRITE dari huggingface.co/settings/tokens

Pakai:
    python scripts/upload_dataset_hf.py --repo <username>/processed-id-final
"""
import argparse
import os
import sys

from huggingface_hub import HfApi

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Data", "processed_id_final")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="mis. AriesDjae/processed-id-final")
    a = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set dulu $env:HF_TOKEN (token write).")
    for f in ("train.jsonl", "val.jsonl", "test.jsonl"):
        p = os.path.join(DATA_DIR, f)
        if not os.path.exists(p):
            sys.exit(f"Tidak ketemu: {p}")

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=DATA_DIR, repo_id=a.repo, repo_type="dataset",
                      allow_patterns=["*.jsonl"])
    print(f"OK -> https://huggingface.co/datasets/{a.repo} (PRIVAT)")
    print("Pastikan repo tetap private! Lalu set env endpoint RunPod:")
    print(f"  HF_DATA_REPO={a.repo}")
    print("  HF_TOKEN=<token yang sama / token read>")
