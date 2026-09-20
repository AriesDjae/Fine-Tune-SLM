#!/bin/sh
# spek_hp.sh — catat spesifikasi perangkat uji RQ3, dijalankan DI HP lewat Termux.
#
# Metodologi mewajibkan perangkat uji disebut lengkap; tanpa itu angka RQ3 tidak
# bisa ditafsirkan. Skrip ini hanya membaca properti sistem dan /proc, tidak
# menyentuh data pribadi, dan keluarannya aman ditempel ke naskah atau ke chat.
#
#   sh spek_hp.sh                 # tampilkan
#   sh spek_hp.sh > spek_hp.txt   # simpan untuk disalin ke laptop

p() { printf '%-26s: %s\n' "$1" "$2"; }
g() { getprop "$1" 2>/dev/null; }

echo "=== PERANGKAT ==="
p "pabrikan"        "$(g ro.product.manufacturer)"
p "model pemasaran" "$(g ro.product.marketname)"
p "model"           "$(g ro.product.model)"
p "nama kode"       "$(g ro.product.device)"

echo
echo "=== ANDROID ==="
p "versi Android"   "$(g ro.build.version.release)"
p "API level"       "$(g ro.build.version.sdk)"
p "build"           "$(g ro.build.display.id)"
p "security patch"  "$(g ro.build.version.security_patch)"

echo
echo "=== CHIPSET ==="
p "SoC pabrikan"    "$(g ro.soc.manufacturer)"
p "SoC model"       "$(g ro.soc.model)"
p "platform"        "$(g ro.board.platform)"
p "ABI"             "$(g ro.product.cpu.abi)"
p "arsitektur"      "$(uname -m)"
p "kernel"          "$(uname -r)"

echo
echo "=== CPU ==="
p "inti terlihat"   "$(nproc 2>/dev/null)"
echo "frekuensi maksimum per inti (kHz):"
for f in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do
  [ -r "$f" ] || continue
  c=$(echo "$f" | sed 's#.*/cpu\([0-9]*\)/.*#\1#')
  printf '  cpu%-3s %s\n' "$c" "$(cat "$f")"
done
echo "model inti (dari /proc/cpuinfo):"
grep -m1 -i "hardware\|model name" /proc/cpuinfo 2>/dev/null | sed 's/^/  /'
grep -c "^processor" /proc/cpuinfo 2>/dev/null | sed 's/^/  jumlah processor: /'

echo
echo "=== MEMORI ==="
grep -E "^(MemTotal|MemAvailable|MemFree|SwapTotal)" /proc/meminfo | sed 's/^/  /'
p "ukuran halaman"  "$(getconf PAGE_SIZE 2>/dev/null || getconf PAGESIZE 2>/dev/null || python -c 'import os;print(os.sysconf("SC_PAGE_SIZE"))' 2>/dev/null) byte"

echo
echo "=== TERMUX ==="
p "python"          "$(python -V 2>&1)"
cek_llama=$(python - 2>&1 <<'PY'
import sys
# benchmark_ondevice.py melakukan hal yang sama sebelum mengimpor llama_cpp
if sys.platform == "android":
    sys.platform = "linux"
try:
    import llama_cpp
    print(llama_cpp.__version__)
except Exception as e:
    print("GAGAL: %s: %s" % (type(e).__name__, str(e)[:60]))
PY
)
p "llama_cpp"       "$cek_llama"
p "numpy"           "$(python -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo '-')"

echo
case "$cek_llama" in
  GAGAL*)
    echo
    echo "!! llama_cpp tidak bisa diimpor. Dua sebab yang lazim:"
    echo "   1. memang belum dipasang; atau"
    echo "   2. Termux baru memperbarui Python, sehingga modul yang sudah"
    echo "      dikompilasi untuk versi lama tidak terbaca lagi."
    echo "   Pasang ulang:"
    echo "   CMAKE_ARGS=\"-DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF\" pip install --force-reinstall --no-cache-dir llama-cpp-python"
    ;;
esac

echo "Catatan: RAM total yang dilaporkan Android SELALU lebih kecil daripada RAM"
echo "yang tertulis di kotak (sebagian dipakai firmware). Naskah menyebut angka"
echo "MemTotal di atas, bukan angka pemasaran."
