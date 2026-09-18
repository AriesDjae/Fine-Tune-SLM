"""
space_mode.py — mengalihkan Space arena antara mode CPU (demo) dan GPU (produksi).

    python scripts/space_mode.py cpu     # demo: cpu-upgrade, $0,03/jam
    python scripts/space_mode.py gpu     # produksi: t4-medium, $0,60/jam
    python scripts/space_mode.py status  # lihat keadaan sekarang

Kenapa harus satu perintah dan bukan "tinggal ganti hardware di Settings": mode GPU
menuntut TIGA hal berubah bersamaan, dan kalau hanya sebagian, kegagalannya SENYAP —
Space tetap hidup, dokter tetap bisa memakai, hanya saja tetap selambat CPU sementara
tagihan sudah berjalan di tarif GPU:

  1. `Dockerfile` harus memakai basis CUDA + wheel llama-cpp-python dari indeks cu124;
  2. variabel `GPU_LAYERS` harus > 0 supaya llama.cpp benar-benar memindahkan lapisan;
  3. perangkat kerasnya sendiri harus t4-medium.

Nomor 1 dan 2 dikerjakan skrip ini. **Nomor 3 harus dilakukan manual** di
Settings → Hardware: token hasil `hf auth login` lewat browser ditolak 401 oleh
endpoint hardware, jadi tidak bisa diotomatiskan dengan kredensial itu.

Urutan yang benar saat menyalakan produksi:
    1) `python scripts/space_mode.py gpu`  → tunggu build selesai
    2) Settings → Hardware → t4-medium
    3) `python scripts/space_mode.py status` → pastikan hardware & GPU_LAYERS cocok
Dan saat selesai: kembalikan ke cpu, lalu turunkan hardware-nya lagi. Tarif GPU
berjalan selama Space Running, bukan hanya saat ada yang bertanya.
"""

from __future__ import annotations

import os
import shutil
import sys

from huggingface_hub import HfApi

SPACE = os.environ.get("SPACE_ARENA", "AriesDjaenuri/Medis-Chat")
AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(AKAR, "hf_space_arena")

MODE = {
    # gpu_layers 99 = seluruh lapisan ke GPU. maks_dimuat 4, bukan 6: di T4 16 GB,
    # bobot keenam GGUF 9,3 GB ditambah KV cache 4096 token per model bisa mepet.
    "gpu": {"dockerfile": "Dockerfile.gpu", "GPU_LAYERS": "99", "MAKS_DIMUAT": "4",
            "hardware": "t4-medium", "tarif": "$0,60/jam"},
    "cpu": {"dockerfile": "Dockerfile.cpu", "GPU_LAYERS": "0", "MAKS_DIMUAT": "6",
            "hardware": "cpu-upgrade", "tarif": "$0,03/jam"},
}


def status(api: HfApi) -> None:
    rt = api.get_space_runtime(SPACE)
    print(f"Space      : {SPACE}")
    print(f"stage      : {rt.stage}")
    print(f"hardware   : {rt.hardware} (diminta: {rt.requested_hardware})")
    print(f"sleep_time : {rt.sleep_time} detik")
    # Hanya baris perintah yang dibaca. Komentar di Dockerfile CPU menyebut cu124 juga,
    # dan pencarian substring polos membuat mode CPU terbaca sebagai GPU.
    perintah = [b for b in open(os.path.join(DIR, "Dockerfile"), encoding="utf-8")
                if not b.lstrip().startswith("#") and "--extra-index-url" in b]
    indeks = "cu124" if any("cu124" in b for b in perintah) else "cpu"
    print(f"Dockerfile lokal memakai indeks wheel: {indeks}")
    if indeks == "cu124" and rt.hardware != "t4-medium":
        print("  ⚠ image GPU tetapi perangkat kerasnya bukan t4-medium")
    if indeks == "cpu" and rt.hardware == "t4-medium":
        print("  ⚠ membayar tarif GPU dengan image CPU — GPU tidak akan terpakai")


def alih(api: HfApi, mode: str) -> None:
    m = MODE[mode]
    shutil.copyfile(os.path.join(DIR, m["dockerfile"]), os.path.join(DIR, "Dockerfile"))
    print(f"Dockerfile <- {m['dockerfile']}")

    for nama in ("GPU_LAYERS", "MAKS_DIMUAT"):
        api.add_space_variable(SPACE, nama, m[nama])
        print(f"variabel {nama} = {m[nama]}")

    api.upload_file(path_or_fileobj=os.path.join(DIR, "Dockerfile"),
                    path_in_repo="Dockerfile", repo_id=SPACE, repo_type="space",
                    commit_message=f"mode {mode}: image + GPU_LAYERS")
    print("Dockerfile terkirim — rebuild berjalan.")
    print(f"\nLANGKAH MANUAL: Settings → Hardware → {m['hardware']} ({m['tarif']}).")
    print("Tanpa itu, mode ini belum lengkap.")


def main() -> int:
    arg = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    api = HfApi()
    if arg == "status":
        status(api)
    elif arg in MODE:
        alih(api, arg)
        print()
        status(api)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
