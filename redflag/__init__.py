"""Lapisan red-flag: deteksi tanda kegawatan pada keluhan pengguna + banner eskalasi.

Rule-based, nol parameter, ~0 ms — sengaja TIDAK dilatihkan ke model supaya bisa
diaudit, deterministik, dan bisa diperbarui tanpa retrain saat pedoman berubah.
Dasar sitasi tiap pola: `redflag/SUMBER_PEDOMAN.md`.
"""

from .detector import (  # noqa: F401
    HasilDeteksi,
    Match,
    apply_banner,
    banner,
    detect,
    strip_banner,
)
from .rules import KATEGORI, RULES  # noqa: F401

__all__ = [
    "detect",
    "banner",
    "apply_banner",
    "strip_banner",
    "HasilDeteksi",
    "Match",
    "RULES",
    "KATEGORI",
]
