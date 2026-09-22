"""
uji_konsistensi.py — gerbang anti-hanyut. JALANKAN SEBELUM DEPLOY.

`hf_space_arena/app.py` memuat SALINAN konstanta dari `arena/config.py`, karena sebuah
Space hanya membundel isi direktorinya sendiri. Salinan yang menyimpang diam-diam adalah
cara paling mudah membuat dokter menilai sistem yang berbeda dari yang dilaporkan di
naskah — system prompt yang bergeser satu kata, atau max_tokens yang tidak sama, cukup
untuk itu.

    python scripts/uji_konsistensi.py

Keluar dengan kode 1 bila ada yang menyimpang.

Riwayat: versi sebelumnya juga menggerbangi jalur Vercel + RunPod
(`webapp/api/index.py`, `webapp/runpod/models.py`, `vercel.json`). Jalur itu
dipensiunkan, berkasnya ada di `arsip/pivot6_jalur-lama/`, dan pemeriksaannya ikut
dicabut dari sini supaya gerbang ini tidak menguji sesuatu yang tidak lagi di-deploy.
"""

from __future__ import annotations

import ast
import pathlib

AKAR = pathlib.Path(__file__).resolve().parent.parent


def _konstanta(jalur: pathlib.Path) -> dict:
    """Baca penetapan tingkat-modul TANPA mengimpor berkasnya.

    Sengaja lewat AST, bukan import: `hf_space_arena/app.py` mengimpor fastapi dan
    llama_cpp, dan gerbang sekecil ini tidak boleh menuntut seluruh dependensi server
    terpasang hanya untuk membandingkan beberapa konstanta.

    Ekspresi dievaluasi memakai nama-nama yang sudah terkumpul sebelumnya, jadi
    PARAMS_GENERASI yang merujuk MAX_TOKENS dkk. tetap terbaca. Yang tidak bisa
    dievaluasi (mis. panggilan fungsi) dilewati diam-diam.
    """
    pohon = ast.parse(jalur.read_text(encoding="utf-8"))
    env: dict = {}
    for simpul in pohon.body:
        if not isinstance(simpul, ast.Assign) or len(simpul.targets) != 1:
            continue
        sasaran = simpul.targets[0]
        if not isinstance(sasaran, ast.Name):
            continue
        try:
            env[sasaran.id] = eval(                     # noqa: S307 - berkas milik repo ini
                compile(ast.Expression(simpul.value), str(jalur), "eval"), {}, dict(env))
        except Exception:
            continue
    return env


def main() -> int:
    cfg = _konstanta(AKAR / "arena" / "config.py")
    beda: list[str] = []

    # --- Space arena (HF Docker): frontend + backend + penyimpanan jadi satu ---
    space_py = AKAR / "hf_space_arena" / "app.py"
    if not space_py.exists():
        print("hf_space_arena/app.py TIDAK ADA.")
        return 1
    spc = _konstanta(space_py)

    def banding(nama: str, a, b):
        if a != b:
            beda.append(f"  {nama}\n    arena/config.py   : {a!r}\n    hf_space_arena    : {b!r}")

    banding("SYSTEM_PROMPT", cfg["SYSTEM_PROMPT"], spc["SYSTEM_PROMPT"])
    banding("PARAMS_GENERASI", cfg["PARAMS_GENERASI"], spc["PARAMS_GENERASI"])
    banding("ACAK_POSISI", cfg["ACAK_POSISI"], spc["ACAK_POSISI"])
    banding("SKALA", list(cfg["SKALA"]), list(spc["SKALA"]))
    banding("HF_DATASET_REPO", cfg["HF_DATASET_REPO"], spc["HF_DATASET_REPO"])
    banding("LABELS", list(cfg["MODELS"].keys()), list(spc["LABELS"]))
    banding("MODELS", cfg["MODELS"], spc["MODELS"])
    banding("CONTOH_PERTANYAAN",
            list(cfg["CONTOH_PERTANYAAN"]), list(spc["CONTOH_PERTANYAAN"]))

    # KEY_MAP tidak boleh pernah ada di sisi Space: apa pun yang di-deploy berada satu
    # proses dengan endpoint publik, dan tirai baru dibuka saat analisis.
    if "KEY_MAP" in spc:
        beda.append("  KEY_MAP BOCOR ke hf_space_arena/app.py — hapus.")

    # chat_format.py disalin ke Space; salinan yang menyimpang berarti prompt yang
    # dirender di sana tidak lagi sama dengan yang dipakai saat training dan eval.py.
    # Dibandingkan sebagai baris, bukan byte mentah: di Windows satu `git checkout` atau
    # editor yang menulis CRLF sudah cukup membuat perbandingan byte merah tanpa ada satu
    # karakter kode pun yang berubah.
    def _baris(p: pathlib.Path) -> list[str]:
        return p.read_text(encoding="utf-8").splitlines()

    asal = _baris(AKAR / "chat_format.py")
    salinan = AKAR / "hf_space_arena" / "chat_format.py"
    if not salinan.exists():
        beda.append(f"  {salinan.relative_to(AKAR)} TIDAK ADA — salin dari akar repo.")
    elif _baris(salinan) != asal:
        beda.append(f"  {salinan.relative_to(AKAR)} menyimpang dari chat_format.py di akar.")

    # Ikon disematkan sebagai data URI di index.html. Kalau lenyap, tiap kunjungan
    # menambah satu 404 /favicon.ico dan error rate di dasbor jadi menyesatkan.
    halaman = (AKAR / "hf_space_arena" / "public" / "index.html").read_text(encoding="utf-8")
    if 'rel="icon"' not in halaman:
        beda.append("  public/index.html: <link rel=\"icon\"> hilang — tiap kunjungan "
                    "menghasilkan 404 /favicon.ico yang mengotori error rate.")

    if beda:
        print("MENYIMPANG dari arena/config.py:\n")
        print("\n".join(beda))
        print("\nSalin ulang nilainya ke hf_space_arena/app.py, lalu jalankan lagi.")
        return 1

    print("Konsisten: hf_space_arena/app.py == arena/config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
