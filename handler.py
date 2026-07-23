"""
handler.py — entrypoint worker RunPod Serverless (Pivot 6).
DI ROOT repo karena pre-deploy check RunPod hanya memindai root utk
`runpod.serverless.start()`. Logika training tetap di serverless/train_p6.py.

Job input (JSON):
    {"input": {"model": "0.8b"|"2b"|"4b",      # wajib
               "mode": "pilot"|"full",          # default "pilot"
               "load_in_4bit": false,           # default false (bf16 H100)
               "quick_eval": null}}             # default: true saat mode=full

Alur: kirim job pilot -> baca output.gate.verdict -> PASS_GREEN -> kirim job full.
Lihat RUN_SERVERLESS.md untuk perintah lengkap.
"""
import os
import sys
import traceback

import runpod

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "serverless"))
from train_p6 import MODELS, run  # noqa: E402


def handler(job):
    inp = job.get("input") or {}
    model_key = str(inp.get("model", "0.8b")).lower()
    mode = str(inp.get("mode", "pilot")).lower()
    if model_key not in MODELS:
        return {"error": f"model harus salah satu {list(MODELS)}, dapat: {model_key!r}"}
    if mode not in ("pilot", "full"):
        return {"error": f"mode harus 'pilot' atau 'full', dapat: {mode!r}"}

    def prog(msg):
        print(f"[progress] {msg}", flush=True)
        try:
            runpod.serverless.progress_update(job, str(msg)[:256])
        except Exception:
            pass  # progress gagal tidak boleh mematikan training

    try:
        return run(
            model_key=model_key,
            run_mode=mode,
            load_in_4bit=bool(inp.get("load_in_4bit", False)),
            quick_eval=inp.get("quick_eval", None),
            use_compile=bool(inp.get("compile", True)),
            progress=prog,
        )
    except AssertionError as e:      # SELF-CHECK / gate assert -> jangan train, laporkan
        return {"error": f"ASSERT: {e}", "traceback": traceback.format_exc()[-3000:]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
