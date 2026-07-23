"""
scripts/submit_job.py — kirim & pantau job training ke endpoint RunPod Serverless.
Jalankan dari LAPTOP (Windows OK). Hanya butuh `pip install requests`.

Setup sekali (PowerShell):
    $env:RUNPOD_API_KEY     = "rpa_..."       # Settings -> API Keys
    $env:RUNPOD_ENDPOINT_ID = "abc123..."     # halaman endpoint

Pakai:
    python scripts/submit_job.py --model 0.8b --mode pilot          # kirim + pantau
    python scripts/submit_job.py --model 0.8b --mode full
    python scripts/submit_job.py --status <JOB_ID>                  # cek job lama
    python scripts/submit_job.py --cancel <JOB_ID>

Urutan sesi: 0.8b pilot -> (gate PASS_GREEN) -> 0.8b full -> 2b pilot -> ... -> 4b full.
"""
import argparse
import json
import os
import sys
import time

import requests

API = "https://api.runpod.ai/v2"


def _env(name):
    v = os.environ.get(name, "")
    if not v:
        sys.exit(f"Set dulu env {name} (lihat header file ini).")
    return v


def _headers():
    return {"Authorization": f"Bearer {_env('RUNPOD_API_KEY')}",
            "Content-Type": "application/json"}


def _url(path):
    return f"{API}/{_env('RUNPOD_ENDPOINT_ID')}/{path}"


def submit(model, mode, load_in_4bit):
    payload = {"input": {"model": model, "mode": mode, "load_in_4bit": load_in_4bit}}
    r = requests.post(_url("run"), headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    job = r.json()
    print(f"job terkirim: id={job['id']} status={job.get('status')}")
    print(f"  input: {json.dumps(payload['input'])}")
    return job["id"]


def status(job_id):
    r = requests.get(_url(f"status/{job_id}"), headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def cancel(job_id):
    r = requests.post(_url(f"cancel/{job_id}"), headers=_headers(), timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=1))


def watch(job_id, poll_s=30):
    print(f"memantau job {job_id} (poll tiap {poll_s}s; Ctrl+C aman — job tetap jalan di cloud)")
    last = ""
    while True:
        s = status(job_id)
        st = s.get("status", "?")
        line = f"[{time.strftime('%H:%M:%S')}] {st}"
        # progress_update dari worker muncul di field output selama IN_PROGRESS
        out = s.get("output")
        if st == "IN_PROGRESS" and isinstance(out, str) and out != last:
            line += f" | {out}"
            last = out
        print(line, flush=True)
        if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            print("=" * 60)
            print(json.dumps(s, ensure_ascii=False, indent=1))
            if st == "COMPLETED" and isinstance(out, dict):
                gate = (out or {}).get("gate")
                if gate:
                    print("=" * 60)
                    print(f"GATE: {gate.get('verdict')}  "
                          f"({'lanjut mode=full' if gate.get('verdict') == 'PASS_GREEN' else 'JANGAN lanjut — diagnosa dulu'})")
                if out.get("error"):
                    print("ERROR:", out["error"])
            return st
        time.sleep(poll_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit/pantau job training Pivot 6 (RunPod serverless)")
    ap.add_argument("--model", choices=["0.8b", "2b", "4b"])
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--status", metavar="JOB_ID", help="cek status job yang sudah ada")
    ap.add_argument("--cancel", metavar="JOB_ID")
    ap.add_argument("--no-watch", action="store_true", help="kirim saja, jangan pantau")
    a = ap.parse_args()

    if a.cancel:
        cancel(a.cancel)
    elif a.status:
        watch(a.status)
    elif a.model:
        jid = submit(a.model, a.mode, a.load_in_4bit)
        if not a.no_watch:
            watch(jid)
    else:
        ap.print_help()
