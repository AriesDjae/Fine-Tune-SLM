"""
redflag/test_detector.py — uji detektor red-flag.

Jalankan: `python -m redflag.test_detector` (atau `pytest redflag/test_detector.py`).

Tiga kelompok uji:
  1. HARUS TERPICU  — kalimat keluhan yang memuat tanda bahaya (uji recall).
  2. TIDAK BOLEH TERPICU — kalimat yang menyangkal gejala (uji penanganan negasi).
  3. TETAP TERPICU meski ada kata ingkar di klausa LAIN (uji batas klausa).
"""
from __future__ import annotations

import sys

from redflag.detector import apply_banner, detect, strip_banner
from redflag.rules import KATEGORI, RULES, validate_rules

# (kalimat, kategori yang diharapkan)
HARUS_TERPICU: list[tuple[str, str]] = [
    # Kardiovaskular — KMK 1186/2022 PPK Angina Pektoris Stabil
    ("Dok, dada saya nyeri sekali dan menjalar ke lengan kiri", "kardiovaskular"),
    ("Dada terasa seperti ditekan benda berat sejak tadi pagi", "kardiovaskular"),
    ("Nyeri dada disertai keringat dingin dan mual", "kardiovaskular"),
    ("Sakit dada sudah 30 menit tidak hilang", "kardiovaskular"),
    ("Apakah ini gejala serangan jantung?", "kardiovaskular"),
    # Neurologis — Kemenkes SeGeRa Ke RS
    ("Bapak saya senyumnya tidak simetris dan bicara pelo", "neurologis"),
    ("Tangan dan kaki sebelah kiri tiba-tiba lemah", "neurologis"),
    ("Mulut mencong sejak bangun tidur", "neurologis"),
    ("Kebas separuh tubuh sebelah kanan", "neurologis"),
    ("Tiba-tiba pandangan mata kabur sebelah", "neurologis"),
    ("Sakit kepala hebat yang belum pernah saya rasakan", "neurologis"),
    # Perdarahan — KMK 1186/2022 PPK Perdarahan Saluran Cerna
    ("Saya muntah darah tadi malam", "perdarahan"),
    ("BAB saya hitam seperti aspal sudah dua hari", "perdarahan"),
    ("Sudah 3 hari batuk berdarah", "perdarahan"),
    # Obstetri — Kemenkes tanda bahaya kehamilan / Buku KIA
    ("Saya hamil 7 bulan tapi keluar darah dari jalan lahir", "obstetri"),
    ("Gerakan janin berkurang sejak kemarin", "obstetri"),
    ("Usia kehamilan 8 bulan, bayi tidak bergerak seharian", "obstetri"),
    ("Ketuban sudah pecah tapi belum ada kontraksi", "obstetri"),
    ("Ibu hamil bengkak kaki disertai pandangan kabur dan nyeri ulu hati", "obstetri"),
    # Anak — MTBS 2015
    ("Bayi saya tidak mau menyusu sejak semalam", "anak"),
    ("Anak saya lemas sekali dan sulit dibangunkan", "anak"),
    ("Balita saya napas cepat dan dadanya tertarik ke dalam", "anak"),
    ("Bibir anak saya membiru", "anak"),
    ("Anak diare terus menerus dan tidak kencing sejak pagi", "anak"),
    # Kesadaran / napas
    ("Adik saya kejang tadi sore", "kesadaran"),
    ("Suami saya pingsan dan tidak sadarkan diri", "kesadaran"),
    ("Napasnya megap-megap sejak tadi", "pernapasan"),
    # Trauma / keracunan / anafilaksis
    ("Anak saya jatuh dari tangga lalu muntah-muntah", "trauma_keracunan"),
    ("Adik menelan baterai jam tadi", "trauma_keracunan"),
    ("Digigit ular saat di sawah", "trauma_keracunan"),
    ("Setelah makan udang muncul biduran lalu sesak napas", "anafilaksis"),
]

# Kalimat yang MENYANGKAL gejala -> banner tidak boleh muncul.
TIDAK_BOLEH_TERPICU: list[str] = [
    "Saya tidak ada nyeri dada, hanya batuk biasa",
    "Nyeri dada tidak ada sama sekali",
    "Anak saya belum pernah kejang",
    "Tidak muntah darah, hanya mual saja",
    "Saya tidak pingsan, cuma pusing sebentar",
    "BAB tidak hitam, warnanya normal",
    "Bukan serangan jantung kata dokter",
]

# Kata ingkar ada, TAPI di klausa lain -> gejala tetap harus terpicu.
TETAP_TERPICU_LINTAS_KLAUSA: list[str] = [
    "Saya tidak merokok, tapi dada saya nyeri menjalar ke lengan kiri",
    "Demam tidak ada. Namun anak saya kejang tadi malam",
    "Tidak ada riwayat darah tinggi, tetapi mulutnya mencong sejak pagi",
    "Saya belum sempat periksa, muntah darah sudah dua kali",
]

# Keluhan biasa (bukan kegawatan) -> tidak boleh terpicu (uji false positive).
KELUHAN_BIASA: list[str] = [
    "Dok, saya batuk pilek sudah 3 hari, obatnya apa ya?",
    "Bagaimana cara menurunkan kolesterol secara alami?",
    "Anak saya umur 5 tahun berat badannya kurang, apa yang harus saya berikan?",
    "Apa efek samping obat paracetamol jika diminum setiap hari?",
    "Saya sering gatal-gatal di kulit kepala, apakah ini ketombe?",
    "Berapa lama luka jahitan biasanya sembuh?",
    "Apakah boleh olahraga saat sedang haid?",
    "Vitamin apa yang bagus untuk daya tahan tubuh?",
]


def _cek(kondisi: bool, pesan: str, gagal: list[str]) -> None:
    if not kondisi:
        gagal.append(pesan)


def main() -> int:
    gagal: list[str] = []

    # 0. Semua aturan punya sumber sitasi yang sah.
    validate_rules()
    for rule in RULES:
        _cek(bool(rule.catatan), f"aturan {rule.id} tanpa catatan/justifikasi", gagal)

    # 1. Recall
    for kalimat, kategori in HARUS_TERPICU:
        hasil = detect(kalimat)
        _cek(hasil.terpicu, f"TIDAK terpicu (harusnya ya): {kalimat!r}", gagal)
        _cek(
            kategori in hasil.kategori,
            f"kategori {kategori!r} tak muncul untuk {kalimat!r} (dapat: {hasil.kategori})",
            gagal,
        )

    # 2. Negasi
    for kalimat in TIDAK_BOLEH_TERPICU:
        hasil = detect(kalimat)
        _cek(
            not hasil.terpicu,
            f"TERPICU padahal disangkal: {kalimat!r} -> {[m.rule_id for m in hasil.aktif]}",
            gagal,
        )

    # 3. Negasi di klausa lain tidak boleh membatalkan
    for kalimat in TETAP_TERPICU_LINTAS_KLAUSA:
        hasil = detect(kalimat)
        _cek(hasil.terpicu, f"TIDAK terpicu padahal ingkar ada di klausa lain: {kalimat!r}", gagal)

    # 4. False positive pada keluhan biasa
    for kalimat in KELUHAN_BIASA:
        hasil = detect(kalimat)
        _cek(
            not hasil.terpicu,
            f"FALSE POSITIVE: {kalimat!r} -> {[m.rule_id for m in hasil.aktif]}",
            gagal,
        )

    # 5. Banner: menempel di atas, jawaban tetap utuh, bisa dilepas lagi.
    hasil = detect("dada saya nyeri menjalar ke lengan kiri")
    jawaban = "Nyeri dada dapat disebabkan banyak hal."
    dengan_banner = apply_banner(jawaban, hasil)
    _cek(dengan_banner.index("PERHATIAN") < dengan_banner.index(jawaban), "banner tidak di atas", gagal)
    _cek(jawaban in dengan_banner, "jawaban asli hilang setelah banner", gagal)
    _cek(strip_banner(dengan_banner).strip() == jawaban, "strip_banner gagal memulihkan jawaban", gagal)
    _cek(apply_banner(jawaban, detect("cara mengobati ketombe")) == jawaban, "banner muncul tanpa pemicu", gagal)

    # 6. Semua kategori punya minimal satu aturan.
    kategori_terpakai = {r.kategori for r in RULES}
    for k in KATEGORI:
        _cek(k in kategori_terpakai, f"kategori {k} tidak punya aturan", gagal)

    if gagal:
        print(f"GAGAL: {len(gagal)} uji tidak lolos\n")
        for g in gagal:
            print("  -", g)
        return 1

    total = len(HARUS_TERPICU) + len(TIDAK_BOLEH_TERPICU) + len(TETAP_TERPICU_LINTAS_KLAUSA) + len(KELUHAN_BIASA)
    print(f"OK — semua uji lolos ({total} kalimat, {len(RULES)} aturan, {len(KATEGORI)} kategori)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
