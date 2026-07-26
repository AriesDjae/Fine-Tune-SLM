"""
redflag/detector.py — deteksi red-flag pada TEKS KELUHAN PENGGUNA (sisi input).

Alur di aplikasi:

    pertanyaan --> [detect()] --> model 0.8B --> [apply_banner()] --> jawaban + banner eskalasi

Catatan penting untuk eval:
  - Banner MENGUBAH teks jawaban. Metrik mutu (token-F1 / ROUGE-L / BERTScore) HARUS
    dihitung pada keluaran TANPA banner — referensi Alodokter tidak punya banner.
    Gunakan `strip_banner()` bila terlanjur tertempel.
  - Deteksi ini dievaluasi TERPISAH (recall & false-positive rate), bukan dengan ROUGE-L.

Setelan HIGH-RECALL adalah keputusan desain yang disengaja dan dinyatakan eksplisit:
melewatkan satu kegawatan jauh lebih mahal daripada memunculkan banner yang tidak perlu.
Karena itu penanganan negasi dibuat KONSERVATIF — negasi hanya membatalkan pemicu bila
kata ingkar berada di KLAUSA YANG SAMA dan berdekatan dengan gejalanya.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rules import KATEGORI, RULES, Rule

# --------------------------------------------------------------------------------------
# Negasi
# --------------------------------------------------------------------------------------
# Kata ingkar bahasa Indonesia (baku + ragam percakapan yang lazim di keluhan online).
_NEGASI = re.compile(
    r"\b(tidak|tidaklah|tdk|tak|ga|gak|nggak|ngga|enggak|engga|kaga|bukan|bukannya|"
    r"belum|blm|tanpa|negatif|nihil)\b",
    re.IGNORECASE,
)

# Batas klausa: tanda baca akhir/pemisah + konjungsi pertentangan.
# Koma ikut dihitung sebagai batas -> negasi lebih jarang membatalkan pemicu (pro-recall).
_BATAS_KLAUSA = re.compile(
    r"[.!?;:\n]|,|\b(tapi|tetapi|namun|melainkan|sedangkan|walau|walaupun|meski|meskipun)\b",
    re.IGNORECASE,
)

# Jendela negasi (dalam kata), dihitung dari batas klausa terdekat.
_JENDELA_SEBELUM = 5
_JENDELA_SESUDAH = 3

BANNER_JUDUL = "PERHATIAN — KEMUNGKINAN TANDA BAHAYA"
BANNER_TEKS = (
    "Keluhan yang Anda sebutkan termasuk tanda yang perlu diperiksa SEGERA oleh tenaga "
    "kesehatan. Jangan menunggu. Segera ke IGD/Puskesmas terdekat, atau hubungi 119 "
    "(layanan gawat darurat). Penjelasan di bawah hanya informasi umum dan BUKAN "
    "pengganti pemeriksaan langsung."
)
_BANNER_MARKER = "[[REDFLAG]]"


@dataclass(frozen=True)
class Match:
    """Satu pemicu yang cocok pada teks."""

    rule_id: str
    kategori: str
    source: str
    awal: int
    akhir: int
    teks: str
    dinegasikan: bool = False


@dataclass
class HasilDeteksi:
    """Hasil deteksi untuk satu pertanyaan."""

    terpicu: bool
    kategori: list[str] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)

    @property
    def aktif(self) -> list[Match]:
        return [m for m in self.matches if not m.dinegasikan]

    @property
    def rule_ids(self) -> list[str]:
        return sorted({m.rule_id for m in self.aktif})

    @property
    def sumber(self) -> list[str]:
        return sorted({m.source for m in self.aktif})

    def ringkas(self) -> str:
        if not self.terpicu:
            return "-"
        return ", ".join(f"{k} ({KATEGORI[k]})" for k in self.kategori)


# --------------------------------------------------------------------------------------
def _batas_klausa(teks: str, awal: int, akhir: int) -> tuple[int, int]:
    """Kembalikan (batas_kiri, batas_kanan) klausa yang memuat rentang [awal, akhir)."""
    kiri = 0
    for m in _BATAS_KLAUSA.finditer(teks, 0, awal):
        kiri = m.end()
    kanan = len(teks)
    m = _BATAS_KLAUSA.search(teks, akhir)
    if m:
        kanan = m.start()
    return kiri, kanan


def _ada_negasi(teks: str, awal: int, akhir: int) -> bool:
    """True bila ada kata ingkar berdekatan DI KLAUSA YANG SAMA, DI LUAR frasa pemicu.

    Dua arah, karena bahasa Indonesia memakai keduanya:
      - sebelum : "tidak ada nyeri dada", "belum pernah kejang"
      - sesudah : "nyeri dada tidak ada", "muntah darah tidak"
    """
    kiri, kanan = _batas_klausa(teks, awal, akhir)

    kata_sebelum = teks[kiri:awal].split()
    if kata_sebelum and _NEGASI.search(" ".join(kata_sebelum[-_JENDELA_SEBELUM:])):
        return True

    kata_sesudah = teks[akhir:kanan].split()
    if kata_sesudah and _NEGASI.search(" ".join(kata_sesudah[:_JENDELA_SESUDAH])):
        return True

    return False


def _masker(teks: str, spans: list[tuple[int, int]]) -> str:
    """Ganti seluruh rentang pemicu dengan spasi.

    Tujuannya: kata ingkar yang merupakan BAGIAN DARI gejala lain
    (mis. "senyumnya tidak simetris") tidak boleh terbaca sebagai penyangkalan
    bagi pemicu tetangganya ("... dan bicara pelo").
    """
    buf = list(teks)
    for awal, akhir in spans:
        for i in range(awal, min(akhir, len(buf))):
            buf[i] = " "
    return "".join(buf)


def detect(teks: str, rules: tuple[Rule, ...] = RULES) -> HasilDeteksi:
    """Deteksi red-flag pada satu teks keluhan."""
    teks = teks or ""

    # Tahap 1: kumpulkan semua kecocokan mentah.
    mentah: list[tuple[Rule, int, int, str]] = []
    for rule in rules:
        for m in rule.pattern.finditer(teks):
            mentah.append((rule, m.start(), m.end(), m.group(0).strip()))

    # Tahap 2: nilai negasi pada teks yang rentang-pemicunya sudah ditutup.
    tertutup = _masker(teks, [(a, b) for _, a, b, _ in mentah])

    matches: list[Match] = []
    for rule, awal, akhir, cocok in mentah:
        dinegasikan = False
        if rule.negatable:
            # (a) ingkar di luar frasa, klausa sama
            dinegasikan = _ada_negasi(tertutup, awal, akhir)
            # (b) ingkar DI DALAM frasa ("BAB tidak hitam") — kecuali pola yang
            #     memang berbentuk ingkar ("tidak mau minum", "tidak simetris").
            if not dinegasikan and not rule.ingkar_internal and _NEGASI.search(cocok):
                dinegasikan = True
        matches.append(
            Match(
                rule_id=rule.id,
                kategori=rule.kategori,
                source=rule.source,
                awal=awal,
                akhir=akhir,
                teks=cocok,
                dinegasikan=dinegasikan,
            )
        )

    aktif = [m for m in matches if not m.dinegasikan]
    kategori = sorted({m.kategori for m in aktif})
    return HasilDeteksi(terpicu=bool(aktif), kategori=kategori, matches=matches)


# --------------------------------------------------------------------------------------
def banner(hasil: HasilDeteksi | None = None) -> str:
    """Teks banner eskalasi (identik untuk semua kategori — sengaja tidak mendiagnosis)."""
    return f"{_BANNER_MARKER}\n**{BANNER_JUDUL}**\n{BANNER_TEKS}\n"


def apply_banner(jawaban: str, hasil: HasilDeteksi) -> str:
    """Tempelkan banner DI ATAS jawaban bila terpicu. Jawaban tetap ditampilkan utuh."""
    if not hasil.terpicu:
        return jawaban
    return f"{banner(hasil)}\n---\n\n{jawaban}"


def strip_banner(teks: str) -> str:
    """Buang banner (untuk menghitung metrik mutu pada keluaran tanpa banner)."""
    if _BANNER_MARKER not in teks:
        return teks
    _, _, sisa = teks.partition("---\n\n")
    return sisa if sisa else re.sub(rf"^{re.escape(_BANNER_MARKER)}.*?\n", "", teks, flags=re.S)


__all__ = [
    "Match",
    "HasilDeteksi",
    "detect",
    "banner",
    "apply_banner",
    "strip_banner",
    "KATEGORI",
]
