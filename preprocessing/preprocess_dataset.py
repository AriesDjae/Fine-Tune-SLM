"""
preprocess_dataset.py  —  BAGIAN 1 (MASTER NOTE FINAL)

Pipeline preprocessing dataset untuk fine-tuning SLM medis (Pivot 3:
Qwen3.5-2B standard dense vs Gemma 4 E2B PLE effective-param dense).

Urutan WAJIB (catat jumlah sampel di SETIAP langkah -> tabel BAB III):
    INSPEKSI -> DEDUP/ANTI-LEAKAGE -> CLEAN GREETING -> FILTER QUALITY
    -> FILTER ICD/HIDR NOISE -> BALANCE BAHASA -> REDUKSI STRATIFIED -> VERIFIKASI

Catatan desain (penyimpangan terdokumentasi dari draft note, dengan alasan):
  * Sumber kebenaran = Data/processed/{train,val,test}.jsonl (FULL, punya field
    `source`). File `processed_shared/*_reduced.jsonl` kehilangan field `source`
    saat reduksi Notebook-00, sehingga TIDAK bisa dipakai untuk filter berbasis
    source (ICD/HIDR) maupun deteksi bahasa. Note Bagian 1.1 mengasumsikan source
    ada -> asumsi itu hanya benar di file FULL.
  * NOISY_SOURCES = {'icd11','hidr'} (nama source ASLI hasil inspeksi 1.1),
    bukan {'icd11_mms','icd_tabulation'} (nama tebakan di draft).
  * Semua file dibaca/ditulis dengan encoding='utf-8' (WAJIB di Windows; default
    cp1252 akan merusak teks Indonesia/unicode).

Output: Data/processed_final/{train_final,val_final,test_final}.jsonl
"""

import json
import re
import sys
import random
import hashlib
from collections import Counter
from pathlib import Path

# Cleaner sapaan BERSAMA (sumber kebenaran tunggal: chat_format.py di root) supaya
# pembersihan saat TRAINING identik dgn saat EVAL/DEPLOY.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_format import clean_greeting as _clean_shared  # noqa: E402

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "Data" / "processed"          # sumber FULL (punya `source`)
OUT_DIR = ROOT / "Data" / "processed_final"   # hasil akhir
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TRAIN = 20_000   # ukuran train akhir (kuota Colab L4) — stratified by source
TARGET_VAL = 1_500      # ukuran val akhir
# test_final = SELURUH test yang sudah dibersihkan (representatif; eval ambil subset fair)

NOISY_SOURCES = {"icd11", "hidr"}   # lookup ICD-11 + indikator WHO = off-task utk patient-QA
ID_TARGET_FRAC = 0.18               # target proporsi bahasa Indonesia di train
ID_UPSAMPLE_MAX = 2                 # batas keras upsample (hindari overfit)

SYSTEM_PROMPT_FALLBACK = (
    "You are a helpful medical assistant. Answer patient questions with accurate, "
    "empathetic responses based on established clinical knowledge. Always recommend "
    "consulting a healthcare professional for proper diagnosis and treatment."
)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def msgs(sample):
    return sample.get("messages", [])


def role_content(sample, role):
    return " ".join(m.get("content", "") for m in msgs(sample) if m.get("role") == role)


def src_dist(samples):
    return Counter(s.get("source", "unknown") for s in samples)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 1.1 INSPEKSI
# --------------------------------------------------------------------------- #
def inspect(split, samples):
    print(f"\n[{split}] rows={len(samples)}")
    for k, v in src_dist(samples).most_common(30):
        print(f"    {k:18s} {v:6d}  ({v/len(samples)*100:4.1f}%)")


# --------------------------------------------------------------------------- #
# 1.2 DEDUP + ANTI-LEAKAGE
# --------------------------------------------------------------------------- #
def sig(sample):
    parts = [m["content"].strip().lower()
             for m in msgs(sample) if m.get("role") in ("user", "assistant")]
    return hashlib.md5(" ".join(parts).encode("utf-8")).hexdigest()


def dedup_exact(samples):
    seen, out = set(), []
    for s in samples:
        h = sig(s)
        if h not in seen:
            seen.add(h)
            out.append(s)
    return out, len(samples) - len(out)


def remove_leak(samples, forbidden):
    out, removed = [], 0
    for s in samples:
        if sig(s) in forbidden:
            removed += 1
        else:
            out.append(s)
    return out, removed


# --------------------------------------------------------------------------- #
# 1.2b NEAR-DUPLICATE (MinHash LSH)  — ~O(n), bukan O(n^2) spt SequenceMatcher
# --------------------------------------------------------------------------- #
# Deteksi near-duplicate (parafrase/edit kecil) via Jaccard similarity pada
# shingle kata, di-bucket dengan LSH sehingga skala mendekati linear (praktis
# utk 100k+ sampel). Butuh paket `datasketch`. Jika TIDAK terpasang, langkah ini
# di-skip dengan peringatan -> pipeline tetap jalan (exact-dedup 1.2 sudah ada).
NEAR_DUP_THRESHOLD = 0.90   # ambang Jaccard; >= ini dianggap near-duplicate
NEAR_DUP_NUM_PERM = 128     # jumlah permutasi MinHash (akurasi vs memori/waktu)
NEAR_DUP_SHINGLE_K = 5      # ukuran shingle = k kata berurutan

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


def _near_text(sample):
    """Teks gabungan user+assistant (konsisten dgn sig()) untuk shingling."""
    parts = [m["content"].strip().lower()
             for m in msgs(sample) if m.get("role") in ("user", "assistant")]
    return " ".join(parts)


def _shingles(text, k=NEAR_DUP_SHINGLE_K):
    toks = text.split()
    if len(toks) < k:
        return {text} if text else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _minhash(text):
    m = MinHash(num_perm=NEAR_DUP_NUM_PERM)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def near_dedup(samples, threshold=NEAR_DUP_THRESHOLD):
    """Buang near-duplicate DALAM satu split (pertahankan kemunculan pertama)."""
    if not _HAS_DATASKETCH:
        return samples, 0
    lsh = MinHashLSH(threshold=threshold, num_perm=NEAR_DUP_NUM_PERM)
    keep, removed = [], 0
    for i, s in enumerate(samples):
        m = _minhash(_near_text(s))
        if lsh.query(m):          # sudah ada yang mirip -> near-duplicate
            removed += 1
            continue
        lsh.insert(str(i), m)
        keep.append(s)
    return keep, removed


def near_remove_leak(samples, ref_samples, threshold=NEAR_DUP_THRESHOLD):
    """Buang dari `samples` yg near-duplicate terhadap `ref_samples` (anti-leakage)."""
    if not _HAS_DATASKETCH:
        return samples, 0
    lsh = MinHashLSH(threshold=threshold, num_perm=NEAR_DUP_NUM_PERM)
    for j, r in enumerate(ref_samples):
        lsh.insert(str(j), _minhash(_near_text(r)))
    keep, removed = [], 0
    for s in samples:
        if lsh.query(_minhash(_near_text(s))):
            removed += 1
        else:
            keep.append(s)
    return keep, removed


# --------------------------------------------------------------------------- #
# 1.3 CLEAN GREETING (+ boilerplate ChatDoctor, + nama halusinasi)
# --------------------------------------------------------------------------- #
# Pendekatan clause-based: buang KLAUSA PEMBUKA basa-basi secara utuh (per
# kalimat, tidak memotong di tengah kata) dengan batas panjang agar konten medis
# yang menyatu dengan pembuka TIDAK ikut terbuang. Sapaan+nama ("Hai Sofy, ")
# dibuang via prefix-strip terpisah supaya jawaban yang menyatu tetap utuh.
_SENT_RE = re.compile(r"[^.!?…]*[.!?…]+[\s\"')]*")
_GREETING_WORDS = r"hai|halo|alo|hi|hello|hey|dear"
_GREETING_LED = re.compile(rf"^\s*(?:{_GREETING_WORDS})\b", re.IGNORECASE)
_CAP_GREETING = 40
_CAP_BOILER = 80

# Sapaan + NAMA (presisi): butuh SPASI setelah sapaan, lalu Kata-Kapital
# (case-sensitive via (?-i:)), dan diikuti koma / frasa terima-kasih. Syarat ini
# mencegah memakan kata konten ("Halo, Gangguan ..." TIDAK kena karena ada koma).
_GREET_NAME = re.compile(
    rf"^\s*(?:{_GREETING_WORDS})\s+"
    r"(?-i:[A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,2})\s*"
    r"(?=,|\s+(?:terima|terimakasih|thanks?|thank|telah|sudah|atas))",
    re.IGNORECASE,
)
# Run sapaan + sapaan-waktu Indonesia: hanya buang TOKEN sapaan, tak pernah kata lain.
_GREET_RUN = re.compile(
    rf"^\s*(?:(?:{_GREETING_WORDS})\b|selamat\s+(?:pagi|siang|sore|malam)\b)[\s,.!:;\-]*",
    re.IGNORECASE,
)
_HELLO_THERE = re.compile(r"^\s*(?:hi|hello|hey)\s+there\b[\s,.!:;\-]*", re.IGNORECASE)
# Sapaan "lengket" tanpa spasi: "HelloVolume", "HelloThanks", "HelloYour".
# Hanya hello/halo/hallo + huruf KAPITAL berikutnya (hindari "HIV","Hidradenitis").
_GLUED_GREETING = re.compile(r"^(?:hello|hallo|halo)(?=[A-Z])")

# Pleasantry TANPA tanda baca kalimat yang menyatu ke jawaban (umum di Alodokter/
# ChatDoctor): "Terima kasih ... Alodokter <jawaban>". Dibatasi {0,70} char + harus
# berakhir di nama platform -> tidak mungkin memakan jauh ke dalam konten.
_PLATFORM_PLEASANTRY = re.compile(
    r"^\s*(?:terima\s*kasih|terimakasih|thanks?|thank\s+you|welcome)"
    r"[^.!?…]{0,70}?(?:alodok\w*|chat\s*doctor|healthcaremagic)[\s,.:;\-]*",
    re.IGNORECASE,
)
# Varian Indonesia tanpa nama platform: "Terima kasih atas pertanyaannya/Anda <jawaban>".
_THANKS_QN_ID = re.compile(
    r"^\s*(?:terima\s*kasih|terimakasih)\s+"
    r"(?:atas\s+pertanyaan(?:nya|\s+anda)?|(?:sudah|telah)\s+(?:bertanya|menghubungi))"
    r"(?:\b|(?=[A-Z]))[\s,.:;\-]*",
    re.IGNORECASE,
)

# Kalimat pembuka basa-basi (TANPA sapaan telanjang -> itu ditangani di atas).
_BOILER_LEAD = re.compile(
    r"""^\s*(
        selamat\s+(pagi|siang|sore|malam)
      | salam\s+sehat
      | (terima\s*kasih|terimakasih)\b
      | welcome\b
      | thanks?\b
      | thank\s+you\b
      | for\s+(asking|posting|approaching|writing|contacting|your\s+query|your\s+question)\b
      | my\s+name\s+is\s+chat\s*doctor\b
      | i\s+(have\s+)?(carefully\s+|just\s+)?(gone|read|passed|went)\s+(through|thru)\b
      | i\s+(can\s+|could\s+)?understand\s+your\s+(concern|worry|problem|anxiety|situation)\b
      | i\s+(have\s+)?(carefully\s+)?(read|noted|reviewed|studied)\s+your\b
      | i\s+appreciate\s+your\b
      | hope\s+(this|it)\s+helps\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def clean_greeting(text):
    original = (text or "").strip()
    t = original
    for _ in range(8):
        before = t
        t = _GREET_NAME.sub("", t).lstrip(" ,.!:;-")
        t = _HELLO_THERE.sub("", t).lstrip()
        t = _GREET_RUN.sub("", t).lstrip()
        t = _GLUED_GREETING.sub("", t).lstrip()
        t = _PLATFORM_PLEASANTRY.sub("", t).lstrip()
        t = _THANKS_QN_ID.sub("", t).lstrip()
        m = _SENT_RE.match(t)
        if m:
            seg = m.group(0).strip()
            cap = _CAP_GREETING if _GREETING_LED.match(seg) else _CAP_BOILER
            if _BOILER_LEAD.match(seg) and len(seg) <= cap:
                rest = t[m.end():].strip()
                if len(rest) >= 20:    # jangan buang kalau sisa konten terlalu sedikit
                    t = rest
        if t == before:
            break
    # Pass akhir: cleaner BERSAMA chat_format (menutup celah mention @user, "Salam,",
    # "HelloThanks" glued) -> kategori sapaan ini dibersihkan identik dgn eval/deploy.
    t2 = _clean_shared(t)
    if len(t2.strip()) >= 20:
        t = t2
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t if len(t.strip()) >= 20 else original


def apply_clean(samples, audit_label=None, audit_top=8):
    """Bersihkan sapaan pada assistant turns. Jika `audit_label` diberikan, cetak
    audit OVER-STRIPPING (Task 3): jawaban yg kehilangan >50 char ditampilkan
    before/after utk verifikasi tidak ada konten medis valid yang terpotong."""
    n = 0
    removals = []  # (removed_chars, before, after)
    for s in samples:
        for m in msgs(s):
            if m.get("role") == "assistant":
                before = m["content"]
                after = clean_greeting(before)
                if after != before:
                    n += 1
                    rm = len(before) - len(after)
                    if rm > 50:
                        removals.append((rm, before[:80], after[:80]))
                m["content"] = after
    if audit_label and removals:
        removals.sort(reverse=True)
        print(f"  [over-strip audit {audit_label}] {len(removals)} jawaban -% >50 char "
              f"(top {min(audit_top, len(removals))} — cek tidak ada konten medis valid):")
        for rm, b, a in removals[:audit_top]:
            print(f"    -{rm}c | SBL: {b!r}")
            print(f"          | SSD: {a!r}")
    return n


# --------------------------------------------------------------------------- #
# 1.4 FILTER QUALITY
# --------------------------------------------------------------------------- #
BAD_SUBSTR = ["[deleted]", "lorem ipsum", "http://", "https://", "www.", "<table", "##"]


def is_low_quality(s, strict=True):
    for m in msgs(s):
        c = m.get("content", "")
        role = m.get("role")
        if role == "assistant":
            if len(c.strip()) < 20 or len(c) > 3000:
                return True
            if strict and any(x in c.lower() for x in BAD_SUBSTR):
                return True
        if role == "user" and len(c.strip()) < 5:
            return True
    return False


# --------------------------------------------------------------------------- #
# 1.5 FILTER ICD/HIDR NOISE
# --------------------------------------------------------------------------- #
def is_icd_noise(s):
    if s.get("source", "") in NOISY_SOURCES:
        return True
    for m in msgs(s):
        c = m.get("content", "")
        if "What is the" in c and "code" in c.lower() and "ICD" in c and len(c) < 200:
            return True
    return False


# --------------------------------------------------------------------------- #
# 1.6 DETEKSI BAHASA
# --------------------------------------------------------------------------- #
ID_SOURCES = ("indonesia_qna", "alodokter", "ppk", "kemenkes", "indonesia", "id_med")
ID_KEYWORDS = ["puskesmas", "dokter", "pasien", "obat", "demam", "rumah sakit",
               "kesehatan", "penyakit", "gejala", "keluhan", "saya", "yang", "dan",
               "tidak", "dengan"]


def is_id(s):
    src = s.get("source", "").lower()
    if any(x in src for x in ID_SOURCES):
        return True
    c = " ".join(m.get("content", "") for m in msgs(s)).lower()
    return sum(w in c for w in ID_KEYWORDS) >= 3


# --------------------------------------------------------------------------- #
# Reduksi stratified by source
# --------------------------------------------------------------------------- #
def stratified_reduce(samples, target):
    if len(samples) <= target:
        return samples[:]
    by_src = {}
    for s in samples:
        by_src.setdefault(s.get("source", "unknown"), []).append(s)
    out = []
    total = len(samples)
    for src, items in by_src.items():
        random.shuffle(items)
        k = max(1, round(target * len(items) / total))
        out.extend(items[:k])
    random.shuffle(out)
    return out[:target]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    stats = []  # (langkah, train, val, test)

    def log(step, tr, vl, te):
        stats.append((step, len(tr), len(vl), len(te)))
        print(f"\n>> {step}: train={len(tr)}  val={len(vl)}  test={len(te)}")

    banner("1.1 INSPEKSI DATASET")
    train = load_jsonl(IN_DIR / "train.jsonl")
    val = load_jsonl(IN_DIR / "val.jsonl")
    test = load_jsonl(IN_DIR / "test.jsonl")
    for name, sp in (("train", train), ("val", val), ("test", test)):
        inspect(name, sp)
    log("00 load", train, val, test)

    banner("1.2 DEDUP EXACT + ANTI-LEAKAGE")
    train, d1 = dedup_exact(train)
    val, d2 = dedup_exact(val)
    test, d3 = dedup_exact(test)
    print(f"Exact dedup: train -{d1}, val -{d2}, test -{d3}")
    test_h = {sig(s) for s in test}
    val_h = {sig(s) for s in val}
    train, l1 = remove_leak(train, test_h)
    train, l2 = remove_leak(train, val_h)
    val, l3 = remove_leak(val, test_h)
    print(f"Leakage removed: train -{l1}(test) -{l2}(val) | val -{l3}(test)")
    log("01 dedup+leak", train, val, test)

    banner(f"1.2b NEAR-DUPLICATE (MinHash LSH, Jaccard >= {NEAR_DUP_THRESHOLD})")
    if not _HAS_DATASKETCH:
        print("[SKIP] paket `datasketch` tidak terpasang -> near-dup dilewati.")
        print("       Install: pip install datasketch  (hanya exact-dedup 1.2 yang aktif).")
    else:
        train, n1 = near_dedup(train)
        val, n2 = near_dedup(val)
        test, n3 = near_dedup(test)
        print(f"Near-dup dalam split: train -{n1}, val -{n2}, test -{n3}")
        train, nl1 = near_remove_leak(train, test)
        train, nl2 = near_remove_leak(train, val)
        val, nl3 = near_remove_leak(val, test)
        print(f"Near-dup leakage: train -{nl1}(test) -{nl2}(val) | val -{nl3}(test)")
    log("01b near-dup", train, val, test)

    banner("1.3 CLEAN GREETING (semua split, termasuk test utk ROUGE fair)")
    ct = apply_clean(train, audit_label="train")
    cv = apply_clean(val, audit_label="val")
    cte = apply_clean(test, audit_label="test")
    print(f"Greeting/boilerplate cleaned (assistant turns): train {ct} | val {cv} | test {cte}")
    log("02 greeting", train, val, test)

    banner("1.4 FILTER QUALITY")
    train = [s for s in train if not is_low_quality(s, strict=True)]
    val = [s for s in val if not is_low_quality(s, strict=True)]
    # TEST: hanya buang yang benar-benar rusak (jaga representativitas) -> strict=False
    test = [s for s in test if not is_low_quality(s, strict=False)]
    log("03 quality", train, val, test)

    banner("1.5 FILTER ICD/HIDR NOISE (off-task lookup/indicator)")
    train = [s for s in train if not is_icd_noise(s)]
    val = [s for s in val if not is_icd_noise(s)]
    test = [s for s in test if not is_icd_noise(s)]
    log("04 icd/hidr noise", train, val, test)

    banner("1.6 BALANCE BAHASA (upsample Indonesia, MAX 2x)")
    id_s = [s for s in train if is_id(s)]
    en_s = [s for s in train if not is_id(s)]
    frac = len(id_s) / max(1, len(train))
    print(f"Sebelum balance: ID {len(id_s)} ({frac*100:.1f}%) | EN {len(en_s)}")
    if frac < ID_TARGET_FRAC:
        tgt = min(int(len(en_s) * ID_TARGET_FRAC / (1 - ID_TARGET_FRAC)),
                  len(id_s) * ID_UPSAMPLE_MAX)
        if len(id_s) < tgt:
            id_s = id_s + random.choices(id_s, k=tgt - len(id_s))
        train = id_s + en_s
        random.shuffle(train)
        print(f"Upsampled -> ID {len(id_s)} ({len(id_s)/len(train)*100:.1f}%)")
    else:
        print(f"ID% sudah >= target {ID_TARGET_FRAC*100:.0f}% -> TIDAK upsample (hindari overfit)")
    log("05 balance", train, val, test)

    banner(f"REDUKSI STRATIFIED (train->{TARGET_TRAIN}, val->{TARGET_VAL}, test=full)")
    train = stratified_reduce(train, TARGET_TRAIN)
    val = stratified_reduce(val, TARGET_VAL)
    # re-balance ID di dalam train tereduksi (proporsi bisa bergeser sedikit)
    id_after = sum(is_id(s) for s in train)
    print(f"train tereduksi: {len(train)} | ID {id_after} ({id_after/len(train)*100:.1f}%)")
    log("06 reduce", train, val, test)

    banner("1.8 SIMPAN + VERIFIKASI")
    save_jsonl(train, OUT_DIR / "train_final.jsonl")
    save_jsonl(val, OUT_DIR / "val_final.jsonl")
    save_jsonl(test, OUT_DIR / "test_final.jsonl")
    print(f"Tersimpan di {OUT_DIR}")

    print("\n--- 10 contoh assistant content (cek bebas sapaan & tidak rusak) ---")
    for s in random.sample(train, min(10, len(train))):
        for m in msgs(s):
            if m.get("role") == "assistant":
                print(f"[{s.get('source','?'):14s}] {m['content'][:110]!r}")
                break

    print("\n--- Distribusi source FINAL ---")
    for name, sp in (("train_final", train), ("val_final", val), ("test_final", test)):
        print(f"\n[{name}] {len(sp)}")
        for k, v in src_dist(sp).most_common():
            print(f"    {k:18s} {v:6d}  ({v/len(sp)*100:4.1f}%)")

    banner("RINGKASAN PER-LANGKAH (untuk tabel BAB III)")
    print(f"{'step':22s} {'train':>8s} {'val':>8s} {'test':>8s}")
    for step, tr, vl, te in stats:
        print(f"{step:22s} {tr:8d} {vl:8d} {te:8d}")

    # Simpan stats per-langkah -> dipakai grafik funnel A.5 (visualize_dataset.py)
    stats_path = ROOT / "results" / "preprocess_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump([{"step": s, "train": tr, "val": vl, "test": te}
                   for s, tr, vl, te in stats], f, indent=2)
    print(f"\nStats per-langkah -> {stats_path}")


if __name__ == "__main__":
    main()
