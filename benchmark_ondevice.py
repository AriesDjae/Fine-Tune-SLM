"""
benchmark_ondevice.py  —  BAGIAN 5 (MASTER NOTE FINAL): pengukuran on-device (CPU-only).

Menguji HIPOTESIS arsitektur (5.2): "Gemma 4 E2B (PLE effective-param dense, ~2.3B aktif)
butuh RAM LEBIH BESAR dari Qwen3.5-2B standard dense karena tabel Per-Layer Embeddings
(total ~5.1B) tetap dimuat ke memori — kecuali caching PLE meredamnya." Ukur:
  - PEAK RAM saat LOADING model            (apakah Gemma 4 memuat ~5.1B embedding?)
  - PEAK RAM saat INFERENSI (dengan KV cache)
  - INFERENCE TIME / throughput (token/detik), mean +- std  (5.3)
  - ukuran file GGUF + dokumentasi arsitektur (total vs active params)  (5.1)

Dijalankan pada model GGUF Q4_K_M via llama.cpp (CPU murni, n_gpu_layers=0) -> paling
mendekati deployment Termux di HP. Sebutkan spesifikasi perangkat uji di metodologi.

    python benchmark_ondevice.py --gguf outputs/gguf/qwen35-2b-medical-Q4_K_M.gguf  --model_type qwen  --label qwen_q4
    python benchmark_ondevice.py --gguf outputs/gguf/gemma4-e2b-medical-Q4_K_M.gguf --model_type gemma --label gemma_q4
    python benchmark_ondevice.py --summarize results_bench
"""
import argparse
import glob
import json
import os
import statistics
import threading
import time

import psutil

# Dokumentasi arsitektur (dari model card; "active" tak bisa diintrospeksi otomatis).
ARCH = {
    "qwen":  {"family": "Qwen3.5-2B", "type": "dense",
              "total_params_B": 2.0, "active_params_B": 2.0,
              "note": "standard dense; semua parameter aktif tiap inferensi"},
    "gemma": {"family": "Gemma 4 E2B", "type": "PLE effective-param dense",
              "total_params_B": 5.1, "active_params_B": 2.3,
              "note": "dense + Per-Layer Embeddings; total ~5.1B dimuat, ~2.3B aktif (compute)"},
}

QUERIES = [
    "Apa penyebab umum demam pada anak dan kapan harus dibawa ke dokter?",
    "Jelaskan gejala dan penanganan awal diare pada balita.",
    "Apa perbedaan antara hipertensi primer dan sekunder?",
    "Bagaimana cara mengelola gula darah pada penderita diabetes tipe 2?",
    "What are the warning signs of a stroke that require emergency care?",
    "Explain the difference between bacterial and viral pneumonia.",
    "Apa saja efek samping umum dari obat antibiotik dan cara menghindarinya?",
    "When should a persistent cough be evaluated by a physician?",
]


class PeakRSS:
    """Sampler RSS latar belakang -> menangkap PUNCAK memori (mis. saat KV cache tumbuh)."""
    def __init__(self, interval=0.05):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = self.proc.memory_info().rss
        self._run = False
        self._t = None

    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def reset(self):
        self.peak = self.proc.memory_info().rss

    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._run = False
        if self._t:
            self._t.join()
        return self.peak / 1e6  # MB


def build_prompt(model_type, q):
    _sys = "You are a helpful medical assistant."
    if model_type == "gemma":   # Gemma 4 turn-token format (system native)
        return (f"<|turn>system\n{_sys}<turn|>\n<|turn>user\n{q}<turn|>\n<|turn>model\n")
    return ("<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def benchmark(args):
    from llama_cpp import Llama

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / 1e6
    sampler = PeakRSS()

    # ---- LOAD (CPU-only) + peak RAM saat loading ----
    sampler.reset(); sampler.start()
    t0 = time.time()
    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, n_gpu_layers=0,
                n_threads=args.threads, verbose=False)
    load_time = time.time() - t0
    peak_load = sampler.stop()
    rss_after_load = proc.memory_info().rss / 1e6

    n_params = None
    try:
        n_params = int(llm.metadata.get("general.parameter_count"))
    except Exception:
        pass

    # ---- warmup ----
    llm.create_completion(build_prompt(args.model_type, QUERIES[0]),
                          max_tokens=16, temperature=0.0)

    # ---- INFERENSI + peak RAM (KV cache) + waktu ----
    sampler.reset(); sampler.start()
    lat, tps = [], []
    for i in range(args.n_queries):
        q = QUERIES[i % len(QUERIES)]
        t = time.time()
        o = llm.create_completion(build_prompt(args.model_type, q),
                                  max_tokens=args.max_new_tokens, temperature=0.0, top_k=1)
        dt = time.time() - t
        ntok = o.get("usage", {}).get("completion_tokens") or args.max_new_tokens
        lat.append(dt)
        tps.append(ntok / dt if dt > 0 else 0)
        print(f"  query {i+1}/{args.n_queries}  {dt:5.2f}s  {tps[-1]:5.1f} tok/s", end="\r")
    print()
    peak_infer = sampler.stop()

    arch = ARCH[args.model_type]
    out = {
        "label": args.label, "gguf": args.gguf, "model_type": args.model_type,
        "device": {"cpu": _cpu_name(), "threads": args.threads,
                   "total_ram_GB": round(psutil.virtual_memory().total / 1e9, 1),
                   "os": _os_name()},
        "architecture": {**arch, "gguf_n_params_meta": n_params},
        "file_size_GB": round(os.path.getsize(args.gguf) / 1e9, 3),
        "ram_MB": {
            "baseline": round(rss0, 1),
            "after_load": round(rss_after_load, 1),
            "peak_load": round(peak_load, 1),
            "peak_infer": round(peak_infer, 1),
            "load_weight_only": round(rss_after_load - rss0, 1),
        },
        "timing": {
            "load_time_s": round(load_time, 2),
            "latency_s_mean": round(statistics.mean(lat), 3),
            "latency_s_std": round(statistics.pstdev(lat), 3),
            "tok_per_s_mean": round(statistics.mean(tps), 2),
            "tok_per_s_std": round(statistics.pstdev(tps), 2),
            "n_queries": args.n_queries, "max_new_tokens": args.max_new_tokens,
            "n_ctx": args.n_ctx,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")
    _print_one(out)


def _cpu_name():
    try:
        import platform
        return platform.processor() or platform.machine()
    except Exception:
        return "unknown"


def _os_name():
    import platform
    return f"{platform.system()} {platform.release()}"


def _print_one(o):
    print(f"\n=== {o['label']} ({o['architecture']['family']}, {o['architecture']['type']}) ===")
    print(f"  file size      : {o['file_size_GB']} GB")
    print(f"  peak RAM load  : {o['ram_MB']['peak_load']} MB")
    print(f"  peak RAM infer : {o['ram_MB']['peak_infer']} MB")
    print(f"  throughput     : {o['timing']['tok_per_s_mean']} +- "
          f"{o['timing']['tok_per_s_std']} tok/s")
    print(f"  latency/query  : {o['timing']['latency_s_mean']} +- "
          f"{o['timing']['latency_s_std']} s")


def summarize(folder):
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{folder}/*.json"))]
    if not rows:
        print("Tak ada hasil di", folder); return
    hdr = ["label", "arch", "size_GB", "RAM_load", "RAM_infer", "tok/s", "lat_s"]
    print("\nTABEL TRADE-OFF ON-DEVICE (BAB IV)")
    print(f"{hdr[0]:14s} {hdr[1]:10s} {hdr[2]:>8s} {hdr[3]:>9s} {hdr[4]:>10s} {hdr[5]:>8s} {hdr[6]:>7s}")
    print("-" * 74)
    for o in rows:
        print(f"{o['label']:14s} {o['architecture']['type'][:10]:10s} "
              f"{o['file_size_GB']:8.3f} {o['ram_MB']['peak_load']:9.0f} "
              f"{o['ram_MB']['peak_infer']:10.0f} {o['timing']['tok_per_s_mean']:8.2f} "
              f"{o['timing']['latency_s_mean']:7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", metavar="FOLDER")
    ap.add_argument("--gguf")
    ap.add_argument("--model_type", choices=["qwen", "gemma"])
    ap.add_argument("--label", default="run")
    ap.add_argument("--n_queries", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n_ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize); return
    assert args.gguf and args.model_type, "--gguf dan --model_type wajib"
    if args.out is None:
        args.out = f"results_bench/bench_{args.label}.json"
    benchmark(args)


if __name__ == "__main__":
    main()
