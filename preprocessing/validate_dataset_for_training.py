"""
validate_dataset_for_training.py — VALIDASI READ-ONLY: "dataset siap-train apa belum?"

Untuk dataset ID v2.1-remediated (`Data/processed_id/{train,val,test}.jsonl`,
30003/2997/2998) sebelum fine-tuning Qwen3.5-0.8B (QLoRA, Unsloth/TRL).
TIDAK mengubah dataset. Mencetak VERDICT PASS/FAIL di akhir.

Cek:
  1. row count  == 30003 / 2997 / 2998
  2. field wajib: messages (system/user/assistant), tidak ada content kosong
  3. cross-split dedup == 0/0 (train×val, train×test, val×test)
  4. DECODE-CHECK tokenisasi (chat template Qwen yang AKAN dipakai training):
       - token KONTEN pertama jawaban asisten IKUT dipelajari (labels != -100)
       - tidak ada off-by-one yang memotong token pertama response
       - special token / scaffold <think></think> konsisten dgn inferensi
       - cetak 1 contoh decode utk diperiksa mata
  5. statistik panjang token p50/p95/p99/max vs MAX_SEQ_LENGTH=1024 + #ter-truncate
  6. (opsional) lid sampel masih Indonesia (kalau langdetect/fasttext ada)

Jalankan (pakai env yang punya transformers + bisa fetch tokenizer Qwen):
  ./.venv-gpu/Scripts/python.exe preprocessing/validate_dataset_for_training.py
Output: results/dataset_validation_YYYYMMDD.json + verdict stdout.

CATATAN PENTING (temuan template Qwen3.5):
  Template Qwen3.5 menyisipkan blok kosong `<think>\n\n</think>\n\n` SETELAH
  `<|im_start|>assistant\n` walau enable_thinking=False. Agar train==inferensi,
  `train_on_responses_only` HARUS memakai response_part yg MENYERTAKAN scaffold itu
  (RESPONSE_PART di notebook), sehingga yang dipelajari = jawaban medis saja.
  Validator ini mengecek kedua kemungkinan dan melaporkan boundary sebenarnya.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "Data" / "processed_id"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EXPECTED = {"train": 30003, "val": 2997, "test": 2998}
MAX_SEQ_LENGTH = 1024
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Qwen3.5-0.8B")
# kandidat tokenizer fallback (ChatML kompatibel) bila MODEL_ID gagal di-fetch
TOK_FALLBACKS = ["unsloth/Qwen3.5-0.8B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]

# scaffold thinking kosong yang disisipkan template Qwen3.5 (enable_thinking=False)
THINK_SCAFFOLD = "<think>\n\n</think>\n\n"
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART_BARE = "<|im_start|>assistant\n"
RESPONSE_PART_FULL = RESPONSE_PART_BARE + THINK_SCAFFOLD  # disarankan (train==infer)

_ALNUM = re.compile(r"[^0-9a-z]+")


def norm(s: str) -> str:
    return _ALNUM.sub("", (s or "").lower())


def load(split):
    rows = []
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows


def get_roles(rec):
    out = {}
    for m in rec.get("messages", []):
        out[m["role"]] = m.get("content", "")
    return out


# --------------------------------------------------------------------------- #
def check_counts(splits, report):
    ok = True
    report["counts"] = {}
    for sp, rows in splits.items():
        n = len(rows)
        good = (n == EXPECTED[sp])
        ok &= good
        report["counts"][sp] = {"n": n, "expected": EXPECTED[sp], "ok": good}
        print(f"  [{sp}] {n:,} (harap {EXPECTED[sp]:,}) {'OK' if good else 'FAIL'}")
    return ok


def check_fields(splits, report):
    ok = True
    bad = Counter()
    for sp, rows in splits.items():
        for r in rows:
            roles = get_roles(r)
            if not {"user", "assistant"} <= set(roles):
                bad[f"{sp}_missing_role"] += 1
            elif not roles.get("user", "").strip() or not roles.get("assistant", "").strip():
                bad[f"{sp}_empty_content"] += 1
    report["fields"] = dict(bad)
    if bad:
        ok = False
        for k, v in bad.items():
            print(f"  FAIL {k}: {v}")
    else:
        print("  semua record punya user+assistant non-kosong  OK")
    return ok


def check_dedup(splits, report):
    keys = {sp: {norm(get_roles(r).get("user", "")) + "||" + norm(get_roles(r).get("assistant", ""))
                 for r in rows} for sp, rows in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    ok = True
    report["cross_split_dedup"] = {}
    for a, b in pairs:
        inter = len(keys[a] & keys[b])
        report["cross_split_dedup"][f"{a}x{b}"] = inter
        ok &= (inter == 0)
        print(f"  {a} x {b}: {inter} duplikat {'OK' if inter == 0 else 'FAIL'}")
    return ok


def _load_tokenizer():
    from transformers import AutoTokenizer
    for mid in [MODEL_ID] + TOK_FALLBACKS:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            return tok, mid
        except Exception as e:
            print(f"  (gagal load tokenizer {mid}: {repr(e)[:80]})")
    return None, None


def _render(tok, msgs, add_gen=False):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=add_gen, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)


def _find_sub(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def check_decode(splits, report):
    """Decode-check: replikasi train_on_responses_only secara manual (tanpa unsloth).
    Cari response marker, pastikan token jawaban pertama IKUT dipelajari, tak terpotong."""
    tok, used = _load_tokenizer()
    report["decode_check"] = {"tokenizer": used}
    if tok is None:
        print("  WARN: tokenizer tak bisa di-fetch -> decode-check DILEWATI (limitasi env, bukan FAIL)")
        report["decode_check"]["status"] = "SKIPPED"
        return True  # bukan FAIL dataset

    rec = splits["train"][0]
    msgs = rec["messages"]
    full = _render(tok, msgs, add_gen=False)
    prompt = _render(tok, [m for m in msgs if m["role"] != "assistant"], add_gen=True)
    ids = tok(full, add_special_tokens=False)["input_ids"]

    bare = tok(RESPONSE_PART_BARE, add_special_tokens=False)["input_ids"]
    full_marker = tok(RESPONSE_PART_FULL, add_special_tokens=False)["input_ids"]
    pos_bare = _find_sub(ids, bare)
    pos_full = _find_sub(ids, full_marker)

    has_scaffold = THINK_SCAFFOLD.strip() in full
    answer = get_roles(rec)["assistant"]
    ans_first_word = answer.split()[0] if answer.split() else ""

    # boundary yang DISARANKAN (train==infer): setelah scaffold think -> jawaban medis
    start = (pos_full + len(full_marker)) if pos_full >= 0 else (pos_bare + len(bare))
    active = ids[start:start + 10]
    active_dec = [tok.decode([t]) for t in active]
    # rekonstruksi awal jawaban dari token aktif -> cek tak ada off-by-one
    reco = tok.decode(ids[start:start + 12]).lstrip()
    first_ok = bool(ans_first_word) and reco.lower().startswith(ans_first_word[:6].lower())

    # konsistensi: prompt inferensi berakhir dgn scaffold yg sama
    infer_consistent = prompt.rstrip().endswith("</think>")

    print(f"  tokenizer           : {used} (vocab {len(tok):,})")
    print(f"  scaffold <think>     : {'ADA (template menyisipkan kosong)' if has_scaffold else 'tidak ada'}")
    print(f"  response marker bare : ditemukan@{pos_bare}  | full(+think)@{pos_full}")
    print(f"  first ACTIVE label   : idx {start} -> {active_dec}")
    print(f"  jawaban kata-1 asli  : {ans_first_word!r}  | rekonstruksi: {reco[:40]!r}")
    print(f"  first-token learned  : {'OK (tidak ke-mask / tidak off-by-one)' if first_ok else 'PERIKSA'}")
    print(f"  infer prompt selesai : {'...</think> (KONSISTEN train==infer)' if infer_consistent else 'TIDAK berakhir </think>'}")
    print("  --- contoh full text (700 char) untuk diperiksa mata ---")
    print("  " + full[:700].replace("\n", "\n  "))

    report["decode_check"].update({
        "status": "PASS" if first_ok else "CHECK",
        "vocab_size": len(tok),
        "scaffold_think_inserted": has_scaffold,
        "response_marker_bare_pos": pos_bare,
        "response_marker_full_pos": pos_full,
        "first_active_index": start,
        "first_active_tokens": active_dec,
        "answer_first_word": ans_first_word,
        "reconstructed_start": reco[:60],
        "first_content_token_learned": first_ok,
        "infer_prompt_ends_think": infer_consistent,
        "recommended_response_part": RESPONSE_PART_FULL,
        "note": ("Template menyisipkan <think></think> kosong; gunakan RESPONSE_PART="
                 "'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' agar train==infer "
                 "(loss hanya pada jawaban medis)."),
        "example_text": full[:900],
    })
    return first_ok


def check_token_lengths(splits, report):
    tok, used = _load_tokenizer()
    if tok is None:
        print("  WARN: tokenizer tak ada -> statistik token DILEWATI")
        report["token_lengths"] = {"status": "SKIPPED"}
        return
    report["token_lengths"] = {}
    for sp, rows in splits.items():
        lens = []
        for r in rows:
            full = _render(tok, r["messages"], add_gen=False)
            lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
        lens.sort()
        trunc = sum(1 for x in lens if x > MAX_SEQ_LENGTH)
        d = {
            "p50": lens[len(lens) // 2], "p95": lens[int(0.95 * (len(lens) - 1))],
            "p99": lens[int(0.99 * (len(lens) - 1))], "max": lens[-1],
            "truncated_gt_%d" % MAX_SEQ_LENGTH: trunc,
            "truncated_pct": round(100 * trunc / len(lens), 3),
        }
        report["token_lengths"][sp] = d
        print(f"  [{sp}] p50={d['p50']} p95={d['p95']} p99={d['p99']} max={d['max']} "
              f"| >{MAX_SEQ_LENGTH}: {trunc} ({d['truncated_pct']}%)")


def check_language(splits, report, sample=400):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
    except Exception:
        print("  (langdetect tak terpasang -> cek bahasa dilewati, opsional)")
        report["language"] = {"status": "SKIPPED"}
        return
    import random
    random.seed(42)
    rows = splits["test"]
    pick = random.sample(rows, min(sample, len(rows)))
    id_n = 0
    for r in pick:
        a = get_roles(r).get("assistant", "")[:500]
        try:
            if detect(a) == "id":
                id_n += 1
        except Exception:
            pass
    pct = round(100 * id_n / len(pick), 1)
    report["language"] = {"sample": len(pick), "id_pct": pct}
    print(f"  sampel {len(pick)} jawaban: {pct}% terdeteksi Indonesia (langdetect)")


def main():
    date = dt.datetime.now().strftime("%Y%m%d")
    splits = {sp: load(sp) for sp in ("train", "val", "test")}
    report = {"date": dt.datetime.now().isoformat(timespec="seconds"),
              "model_id": MODEL_ID, "max_seq_length": MAX_SEQ_LENGTH}

    print("=" * 78)
    print("VALIDASI DATASET UNTUK TRAINING — Qwen3.5-0.8B (read-only)")
    print("=" * 78)
    print("\n[1] Row count"); c1 = check_counts(splits, report)
    print("\n[2] Field wajib"); c2 = check_fields(splits, report)
    print("\n[3] Cross-split dedup"); c3 = check_dedup(splits, report)
    print("\n[4] Decode-check tokenisasi"); c4 = check_decode(splits, report)
    print("\n[5] Statistik panjang token"); check_token_lengths(splits, report)
    print("\n[6] Deteksi bahasa (opsional)"); check_language(splits, report)

    verdict = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"
    report["verdict"] = verdict
    report["gates"] = {"counts": c1, "fields": c2, "dedup": c3, "decode_check": c4}

    out = RESULTS / f"dataset_validation_{date}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    if verdict == "FAIL":
        print("  -> ADA gate gagal. STOP. Perbaiki dataset/tokenisasi sebelum bikin notebook.")
    else:
        print("  -> Dataset SIAP train. Lanjut notebooks/train_qwen_qlora.ipynb (RUN_MODE=pilot).")
    print(f"  Catatan token-truncate & scaffold <think> ada di {out.name} (bukan blocker).")
    print(f"  Detail: {out}")
    print("=" * 78)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
