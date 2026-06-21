"""
export_gguf.py  —  BAGIAN 4 (MASTER NOTE FINAL): export merged 16-bit -> GGUF Q4_K_M.

Alur: merged HF (16-bit)  --convert_hf_to_gguf.py-->  *-f16.gguf
                          --llama-quantize-->          *-Q4_K_M.gguf

Lalu evaluasi ULANG metrik Bagian 3 pada GGUF (lihat eval.py --gguf) dan laporkan
16-bit vs Q4_K_M (analisis trade-off kuantisasi). `--verify` menjalankan satu
generasi via llama-cli memakai chat template SAMA (cek konsistensi template 4.4).

PIVOT 5 (single-model): hanya Qwen3.5-0.8B. Merge adapter dulu (merge_adapter.py).

Contoh:
    python merge_adapter.py --adapter outputs/checkpoints/qwen35-0.8b-train \
                            --out outputs/merged/qwen35-0.8b-medical
    python export_gguf.py --merged_dir outputs/merged/qwen35-0.8b-medical --verify

Alternatif termudah (DI NOTEBOOK, model masih di memori, tanpa script ini):
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def find_bin(llama_dir, name):
    """Cari binary (llama-quantize / llama-cli) lintas layout build & OS."""
    cands = []
    for sub in ("build/bin", "build/bin/Release", "build", "build/Release"):
        for ext in ("", ".exe"):
            cands.append(Path(llama_dir) / sub / (name + ext))
    for c in cands:
        if c.exists():
            return str(c)
    return None


def ensure_llama_cpp(llama_dir, force_build=False):
    llama_dir = Path(llama_dir)
    if not llama_dir.exists():
        run(["git", "clone", "--depth", "1", LLAMA_REPO, str(llama_dir)])
    if force_build or find_bin(llama_dir, "llama-quantize") is None:
        print("Building llama.cpp (cmake Release)...")
        run(["cmake", "-B", str(llama_dir / "build"), "-S", str(llama_dir),
             "-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake", "--build", str(llama_dir / "build"), "--config", "Release", "-j"])
    qbin = find_bin(llama_dir, "llama-quantize")
    if qbin is None:
        sys.exit("Gagal menemukan/membangun llama-quantize. Build llama.cpp manual.")
    return llama_dir, qbin


def convert_to_f16(llama_dir, merged_dir, out_f16):
    conv = Path(llama_dir) / "convert_hf_to_gguf.py"
    if not conv.exists():                    # nama lama
        conv = Path(llama_dir) / "convert-hf-to-gguf.py"
    run([sys.executable, str(conv), str(merged_dir),
         "--outfile", str(out_f16), "--outtype", "f16"])


def sizeof(p):
    return f"{os.path.getsize(p) / 1e9:.2f} GB" if os.path.exists(p) else "?"


def verify(llama_dir, gguf_path):
    cli = find_bin(llama_dir, "llama-cli")
    if not cli:
        print("llama-cli tak ditemukan -> lewati verify.")
        return
    q = "Apa penyebab umum demam pada anak dan kapan harus ke dokter?"
    # Qwen3.5 ChatML (scaffold <think></think> kosong, sama spt train=eval=deploy)
    prompt = ("<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n"
              f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    print("\n[verify] generasi via llama-cli (cek template & koherensi):")
    run([cli, "-m", str(gguf_path), "-p", prompt, "-n", "96",
         "--temp", "0", "-no-cnv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="outputs/merged/qwen35-0.8b-medical",
                    help="dir model merged 16-bit (hasil merge_adapter.py)")
    ap.add_argument("--out_dir", default="outputs/gguf")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--llama_cpp", default="llama.cpp")
    ap.add_argument("--force_build", action="store_true")
    ap.add_argument("--keep_f16", action="store_true", help="jangan hapus *-f16.gguf")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    name = Path(args.merged_dir).name
    os.makedirs(args.out_dir, exist_ok=True)
    out_f16 = Path(args.out_dir) / f"{name}-f16.gguf"
    out_q = Path(args.out_dir) / f"{name}-{args.quant}.gguf"

    llama_dir, qbin = ensure_llama_cpp(args.llama_cpp, args.force_build)

    print(f"\n[1/2] convert {args.merged_dir} -> {out_f16}")
    convert_to_f16(llama_dir, args.merged_dir, out_f16)

    print(f"\n[2/2] quantize -> {out_q} ({args.quant})")
    run([qbin, str(out_f16), str(out_q), args.quant])

    if not args.keep_f16 and out_f16.exists():
        os.remove(out_f16)

    print("\n=== RINGKASAN UKURAN (untuk tabel trade-off BAB IV) ===")
    print(f"  merged 16-bit dir : {args.merged_dir}")
    print(f"  GGUF {args.quant:8s}    : {out_q}  ({sizeof(out_q)})")

    if args.verify:
        verify(llama_dir, out_q)

    print("\nLangkah berikut: eval ulang terkuantisasi ->")
    print(f"  python eval.py --gguf {out_q} --model {args.merged_dir} "
          f"--label {name}_q4 --loader gguf --n_eval 3000")


if __name__ == "__main__":
    main()
