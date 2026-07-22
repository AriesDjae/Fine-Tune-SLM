# -*- coding: utf-8 -*-
"""
inference/typo_postprocess.py — koreksi typo Bahasa Indonesia pada OUTPUT model saat RUNTIME.

Saran dosen (Pivot 6): perbaiki typo di jawaban model lewat POST-PROCESSING, BUKAN dengan
fine-tuning ulang (hemat resource). Modul ini reuse logika KBBI dari
`preprocessing/fix_typos_kbbi.py`, tapi diarahkan ke teks yang DIHASILKAN model — bukan dataset.

Beda kunci dari fix_typos_kbbi.py:
  fix_typos_kbbi bekerja pada dataset dan punya statistik frekuensi korpus untuk menyaring
  kandidat. Di runtime kita TIDAK punya itu, jadi corrector di sini dibuat KONSERVATIF —
  utamakan PRESISI, jangan pernah merusak istilah medis:
    (1) peta baku KBBI  : bentuk_tidak_baku -> bentuk baku (definitif, dari HF Lyon28).
    (2) kamus kurasi     : typo umum yang mungkin tak tercakup KBBI (CURATED).
    (3) [opsional] edit-1 -> KBBI, HANYA saat `aggressive=True` & kandidat TUNGGAL tak ambigu.
  Default (aggressive=False) hanya memakai (1)+(2): keduanya adalah pemetaan kata-nonbaku-dikenal
  -> baku, sehingga TIDAK pernah menyentuh istilah medis/asing/nama (yang bukan entri nonbaku KBBI).

Proteksi tambahan: kata < MIN_LEN, kata KAPITAL di tengah kalimat (kemungkinan nama diri),
kata ALL-CAPS (akronim), dan kata valid (ada di KBBI / berinfleksi sah) TIDAK disentuh.
Casing dipertahankan (Kapital -> hasil dikapitalkan).

Pemakaian:
    from typo_postprocess import get_corrector
    fix = get_corrector()
    bersih = fix.correct("saya mengalami diare dan resiko dehidrasi")  # -> "...risiko dehidrasi"

Offline: baca cache HF lokal (sama seperti fix_typos_kbbi). Jika KBBI tak tersedia & tak ada
jaringan, corrector fallback ke CURATED-only + WARN (tak pernah crash — aman untuk app.py/eval).
Dipakai oleh: inference/app.py (toggle) & bisa diimport eval.py untuk ablation on/off.
"""
from __future__ import annotations

import csv
import glob
import os
import re

_WORD = re.compile(r"[A-Za-z]+")
MIN_LEN = 4            # jangan koreksi kata pendek (rawan false-positive)
MAX_LEN = 18

# Prefiks/sufiks afiks ID (untuk mengenali kata berinfleksi sah -> jangan dikoreksi).
_PRE = ["meng", "meny", "mem", "men", "peng", "peny", "pem", "pen",
        "ber", "ter", "per", "di", "ke", "se", "me", "be", "pe"]
_SUF = ["kannya", "annya", "nya", "kan", "lah", "kah", "an", "i"]

# Typo umum yang mungkin tak ada di peta nonbaku KBBI. Konservatif & aman-medis.
# (nonbaku -> baku). Tambah HANYA setelah yakin bukan istilah medis/nama.
CURATED = {
    "telfon": "telepon", "hp": "hp",
    "aktifitas": "aktivitas", "apotik": "apotek", "atmosfir": "atmosfer",
    "analisa": "analisis", "diagnosa": "diagnosis", "frekwensi": "frekuensi",
    "karna": "karena", "karan": "karena", "kwalitas": "kualitas",
    "nafas": "napas", "nasehat": "nasihat", "obat2an": "obat-obatan",
    "praktek": "praktik", "resiko": "risiko", "sistim": "sistem",
    "standart": "standar", "trauma": "trauma", "propinsi": "provinsi",
    "jadwal": "jadwal", "jadual": "jadwal", "kadaluarsa": "kedaluwarsa",
    "komplen": "komplain", "silahkan": "silakan", "antri": "antre",
    "aktifis": "aktivis", "efektifitas": "efektivitas", "produktifitas": "produktivitas",
    "sembuh": "sembuh", "gejela": "gejala", "penyakip": "penyakit",
}
# hp/trauma/jadwal/sembuh dibiarkan self-map hanya sebagai penanda "sudah baku" (tak diubah).
CURATED = {k: v for k, v in CURATED.items() if k != v}

# Kata yang TIDAK BOLEH dianggap typo walau mungkin absen dari KBBI (nama umum / istilah).
NEVER = {"salma", "nova", "tante", "kakak", "adik", "bapak", "budi", "intan",
         "ratna", "sari", "dewi", "putri", "rama", "andri", "firdauziah",
         "covid", "corona", "hpv", "hsv"}

_AB = "abcdefghijklmnopqrstuvwxyz"


def _toks(t: str):
    return _WORD.findall(t or "")


_REPO_REL = ("datasets--Lyon28--kamus-besar-bahasa-indonesia/snapshots/*/data.csv")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kbbi_path():
    """OFFLINE-first: cari data.csv KBBI di cache HF (default + env) atau path lokal proyek.

    Urutan: env DATA_KBBI_CSV -> Data/kbbi/data.csv proyek -> HF cache (HF_HOME/
    HUGGINGFACE_HUB_CACHE/HF_HUB_CACHE + default ~/.cache & AppData) -> download (bila ada jaringan).
    """
    env_csv = os.environ.get("DATA_KBBI_CSV")
    if env_csv and os.path.isfile(env_csv):
        return env_csv
    local = os.path.join(_PROJECT_ROOT, "Data", "kbbi", "data.csv")
    if os.path.isfile(local):
        return local

    cache_roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "huggingface", "hub"),
    ]
    for root in cache_roots:
        if not root:
            continue
        hits = glob.glob(os.path.join(root, _REPO_REL))
        if hits:
            return hits[0]

    from huggingface_hub import hf_hub_download  # hanya dipanggil bila cache kosong & ada jaringan
    return hf_hub_download("Lyon28/kamus-besar-bahasa-indonesia", "data.csv", repo_type="dataset")


class TypoCorrector:
    """KBBI-based Indonesian typo corrector untuk teks output model (konservatif, offline)."""

    def __init__(self, load_kbbi: bool = True):
        self.valid: set[str] = set()
        self.official: dict[str, str] = {}
        self.available = False
        if load_kbbi:
            try:
                self._load_kbbi(_kbbi_path())
                self.available = True
            except Exception as e:  # tak ada cache & tak ada jaringan -> CURATED-only
                print(f"[typo_postprocess] WARN: KBBI tak dimuat ({e!r}); "
                      f"pakai CURATED-only ({len(CURATED)} entri).")

    # ---- pemuatan KBBI --------------------------------------------------- #
    def _load_kbbi(self, path: str):
        csv.field_size_limit(10 ** 7)
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                for col in ("nama", "kata_dasar", "varian", "kata_turunan", "gabungan_"):
                    for w in _toks((row.get(col) or "").lower()):
                        if len(w) >= 2:
                            self.valid.add(w)
                nb = _toks((row.get("bentuk_tidak_baku") or "").lower())
                nm = _toks((row.get("nama") or "").lower())
                if len(nb) == 1 and len(nm) == 1:
                    a, b = nb[0], nm[0]
                    if len(b) >= MIN_LEN and a[0] == b[0] and abs(len(a) - len(b)) <= 2 and a != b:
                        self.official.setdefault(a, b)

    # ---- validitas kata -------------------------------------------------- #
    def _morph_valid(self, w: str) -> bool:
        if w in self.valid:
            return True
        for p in _PRE:
            if w.startswith(p) and len(w) - len(p) >= 3:
                r = w[len(p):]
                if r in self.valid:
                    return True
                for s in _SUF:
                    if r.endswith(s) and len(r) - len(s) >= 3 and r[:-len(s)] in self.valid:
                        return True
        for s in _SUF:
            if w.endswith(s) and len(w) - len(s) >= 3 and w[:-len(s)] in self.valid:
                return True
        return False

    def _is_valid(self, w: str) -> bool:
        return w in self.valid or self._morph_valid(w)

    # ---- edit-1 (opsional, aggressive) ----------------------------------- #
    @staticmethod
    def _edits1(w: str):
        s = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        o = set()
        o.update(L + R[1:] for L, R in s if R)                       # hapus
        o.update(L + R[1] + R[0] + R[2:] for L, R in s if len(R) > 1)  # tukar
        o.update(L + c + R[1:] for L, R in s if R for c in _AB)        # ganti
        o.update(L + c + R for L, R in s for c in _AB)                 # sisip
        return o

    def _edit1_candidate(self, w: str):
        """Kandidat edit-1 yang ADA di KBBI. Kembalikan hanya bila TUNGGAL (tak ambigu)."""
        cands = {c for c in self._edits1(w)
                 if c in self.valid and len(c) >= MIN_LEN and abs(len(w) - len(c)) <= 2 and c != w}
        return next(iter(cands)) if len(cands) == 1 else None

    # ---- API utama ------------------------------------------------------- #
    def correct(self, text: str, aggressive: bool = False):
        """Kembalikan teks dengan typo dikoreksi. Casing & tanda baca dipertahankan.

        aggressive=False (default): hanya peta baku KBBI + CURATED (presisi tinggi, aman-medis).
        aggressive=True: tambah edit-1 -> KBBI bila kandidat tunggal (recall lebih tinggi, sedikit
        lebih berisiko — pakai bila mau menyapu typo panjang yang tak ada di peta nonbaku).
        """
        if not text:
            return text
        out = []
        pos = 0
        for m in _WORD.finditer(text):
            out.append(text[pos:m.start()])
            fixed = self._fix_word(m.group(0), text, m.start(), aggressive)
            out.append(fixed)
            pos = m.end()
        out.append(text[pos:])
        return "".join(out)

    def correct_with_count(self, text: str, aggressive: bool = False):
        """Seperti correct() tapi kembalikan (teks, jumlah_penggantian) — utk logging/ablation."""
        self._n = 0
        result = self.correct(text, aggressive)
        return result, self._n

    _n = 0

    def _fix_word(self, word: str, text: str, start: int, aggressive: bool) -> str:
        low = word.lower()
        if len(low) < MIN_LEN or len(low) > MAX_LEN:
            return word
        if low in NEVER:
            return word
        if word.isupper():                       # akronim (ALL CAPS) -> jangan sentuh
            return word
        # KAPITAL di tengah kalimat -> kemungkinan nama diri. Lihat char non-spasi sebelumnya.
        j = start - 1
        while j >= 0 and text[j] == " ":
            j -= 1
        mid_sentence = j >= 0 and text[j] not in ".!?\n"
        if word[0].isupper() and mid_sentence:
            return word

        repl = None
        if low in CURATED:
            repl = CURATED[low]
        elif self.available and not self._is_valid(low):
            if low in self.official:
                repl = self.official[low]
            elif aggressive:
                repl = self._edit1_candidate(low)

        if not repl or repl == low:
            return word
        self._n += 1
        return repl.capitalize() if word[0].isupper() else repl


_SINGLETON: TypoCorrector | None = None


def get_corrector() -> TypoCorrector:
    """Ambil corrector singleton (muat KBBI sekali; aman dipanggil berulang)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = TypoCorrector()
    return _SINGLETON


def correct_text(text: str, aggressive: bool = False) -> str:
    """Fungsi ringkas: koreksi typo satu teks memakai corrector singleton."""
    return get_corrector().correct(text, aggressive)


if __name__ == "__main__":
    # Self-test lokal (tanpa GPU). Jalankan: python inference/typo_postprocess.py
    fix = get_corrector()
    print("KBBI tersedia:", fix.available,
          "| |valid|=", len(fix.valid), "| |official|=", len(fix.official))
    samples = [
        "Anda memiliki resiko tinggi terkena diabetes karna pola makan.",
        "Silahkan kontrol ke apotik terdekat untuk menebus obat.",
        "Aktifitas fisik teratur menurunkan frekwensi serangan.",
        "Gejala nya berupa sesak nafas dan nyeri dada.",
        "Pasien Budi mengalami hipertensi dan gastritis kronis.",   # nama & istilah medis: jangan diubah
        "Konsultasikan ke dokter Sinta bila keluhan berlanjut.",     # nama setelah 'dokter': jangan diubah
    ]
    for s in samples:
        fixed, n = fix.correct_with_count(s)
        flag = "  <-- diubah" if n else ""
        print(f"\n[{n}] {s}\n    {fixed}{flag}")
    print("\nAGGRESSIVE (edit-1):")
    s = "Penderita mengalami muntabber dan dehidrsi ringan."
    print("  ", fix.correct(s, aggressive=True))
