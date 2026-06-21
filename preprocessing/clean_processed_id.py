# -*- coding: utf-8 -*-
"""
clean_processed_id.py — Pembersihan KONSERVATIF dataset Pivot 5 (fine-tune terakhir).
Hanya menyentuh konten ASSISTANT (user = input pasien, sengaja dibiarkan apa adanya).
Input : Data/processed_id/{train,val,test}.jsonl
Output: Data/processed_id_clean/{train,val,test}.jsonl  (ASLI TIDAK DIUBAH)
+ CLEANING_REPORT.txt (statistik per-aturan) + changes_sample.txt (200 diff utk audit).

Aturan (semuanya presisi-tinggi, ada log per-aturan):
  1. Normalisasi spasi.
  2. Strip tanda tangan dokter di ekor (anchored gelar BMed/Sci/dr./Sp. ATAU nama terkurasi).
  2.5 Pulihkan batas kalimat: sisip ". " sebelum kata-pembuka terkurasi (imperatif/konektor/
      penutup/subjek-daftar) yg nempel tanpa tanda baca — sumber Alodokter sering "rata".
      TIDAK menyentuh "Anda"/nama/istilah medis kapital (tak merusak kapital sah).
  3. Hapus kalimat promosi platform (alodokter/halodoc/aplikasi chat/klik link) ATAU rujukan
     artikel lain (halaman berikut/dapat Anda baca/diskusi terkait/artikel ini).
  4. Pisah preposisi nempel (whitelist lokatif; 'keluar'/'dimana' DIBIARKAN).
  5. Kolaps huruf sama 3+ -> 2 (kecuali angka romawi).
  5.5 Tanda baca: spasi setelah koma (skip desimal "37,5"); normalisasi singkatan enumerasi
      dll/dsb/dst -> sisip ", " sebelum (bila nempel) + titik akhir abbr.
  6. Map typo terkurasi (lainya->lainnya).
  7. Tambah titik akhir bila hilang (kosmetik).
Tahap-2 (typo berbasis KBBI) ada di fix_typos_kbbi.py — jalankan SETELAH skrip ini.
"""
import json, re, os, collections

SRC="Data/processed_id"; DST="Data/processed_id_clean"
os.makedirs(DST, exist_ok=True)

NAMES=["Radhianie Djan","Kresnawati Wahyu Setiono","Kresnawai Wahyu Setiono","Celleen Rei Setiawan",
 "Dian Paramitasari","Tri Uji Rahayu","Eni Yulvia Susilayanti","Mirra Mareta Supit",
 "Caecilia Haryu Aryapti","Hariyanto Wibowo Ramme","Doddy Kusumah","Andika Surya","Hernita Carolina"]
RE_NAME=re.compile(r"[\s.,;:-]*(?:dr\.?\s+|drg\.?\s+)?(?:"+"|".join(re.escape(n) for n in NAMES)+r")\b.*$", re.I)
RE_CRED=re.compile(r"[\s.,;:-]*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s*,?\s*)?"
                   r"(?:BMed|B\.?Med|Sci?\b|S\.?Ked\b|Sp\.?[A-Z().\-]*|M\.?Kes\b|MARS\b|FINASIM\b|drg?\.)\.?\s*$")
RE_PLAT=re.compile(r"(alodokter|halodoc|live\s*chat|fitur\s*chat|aplikasi\s+(?:alodokter|halodoc)|"
                   r"unduh\s+aplikasi|download\s+aplikasi|klik\s+(?:link|tautan))", re.I)
# Tanda baca: spasi setelah koma HANYA bila diikuti HURUF (desimal "37,5"/"1,5" -> koma+digit AMAN,
# tak disentuh). Singkatan enumerasi dll/dsb/dst -> sisipkan ", " sebelum (bila nempel kata) + titik.
RE_COMMA_SP=re.compile(r",(?=[A-Za-z])")
RE_ABBR_COMMA=re.compile(r"(?<=[a-z])\s+(?=(?:dll|dsb|dst)\b\.?(?:\s|$|[,.]))", re.I)
RE_ABBR_DOT=re.compile(r"\b(dll|dsb|dst)\b(?!\.)", re.I)
RE_DIPREP=re.compile(r"\bdi(atas|bawah|dalam|luar|rumah|sini|sana|samping|bagian|kamar|tempat|depan|belakang)\b", re.I)
RE_KEPREP=re.compile(r"\bke(dokter|dalam|rumah|atas|bawah|samping|bagian|sini|sana|depan|belakang)\b", re.I)
ROMAN=re.compile(r"^[ivxlcdm]+$", re.I)
TYPO={"lainya":"lainnya","Lainya":"Lainnya"}

# Pemulihan batas kalimat: sumber Alodokter sering "rata" (daftar/kalimat nyambung tanpa
# tanda baca) -> 84% jawaban. Sisip ". " HANYA sebelum kata-pembuka yang HAMPIR PASTI
# memulai kalimat/butir (imperatif, konektor, penutup, subjek-daftar). SENGAJA TIDAK memuat
# "Anda"/"Kamu"/nama/istilah medis kapital (Vitamin/Hepatitis/Herpes/Dengue) -> tak merusak
# kapital sah di tengah kalimat.
SAFE_STARTERS=[
 # imperatif & gaya hidup
 "Hindari","Hindarkan","Hindarilah","Gunakan","Pakai","Kenakan","Jaga","Jagalah","Minum",
 "Minumlah","Konsumsi","Konsumsilah","Perbanyak","Perbanyaklah","Kurangi","Batasi","Kelola",
 "Lakukan","Lakukanlah","Periksa","Periksakan","Periksakanlah","Konsultasikan","Konsultasikanlah",
 "Oleskan","Kompres","Istirahat","Istirahatlah","Rajin","Olahraga","Berolahraga","Makan","Makanlah",
 "Cuci","Bersihkan","Berhenti","Hentikan","Coba","Cobalah","Pastikan","Berikan","Tetap","Segera",
 "Tidur","Atur","Aturlah","Pilih","Pilihlah","Jangan","Janganlah","Kelolalah",
 # konektor / awal kalimat
 "Jika","Apabila","Bila","Namun","Untuk","Sebaiknya","Sebaliknya","Selain","Karena","Sehingga",
 "Setelah","Selama","Adapun","Beberapa","Berikut","Bahkan","Meski","Meskipun","Walaupun",
 "Kemudian","Selanjutnya","Pada","Dengan","Tanpa",
 # penutup
 "Semoga","Demikian","Sekian","Terima","Terimakasih",
 # subjek/daftar medis (kapital di tengah teks = butir baru)
 "Dokter","Gejala","Gejalanya","Penyakit","Infeksi","Gangguan","Kondisi","Peradangan","Radang",
 "Demam","Nyeri","Tumor","Kanker","Faktor","Komplikasi","Kelainan","Kekurangan","Penyebab",
 "Penanganan","Pengobatan","Pemeriksaan","Obat","Kondisinya",
]
RE_GLUE=re.compile(r"(?<=[a-z])\s(?=(?:"+"|".join(SAFE_STARTERS)+r")\b)")
RE_REF=re.compile(r"(diskusi (?:untuk dibaca|lebih lanjut|terkait)|halaman (?:berikut|lain)|"
                  r"baca juga|artikel (?:lanjut|berikut|ini)|berikut kami berikan diskusi|"
                  r"untuk dibaca\s*:|dapat (?:anda|Anda) baca|silakan baca)", re.I)

stat=collections.Counter(); samples=[]

def collapse_triples(text):
    def fix(m):
        w=m.group(0)
        if ROMAN.match(w): return w
        return re.sub(r"(.)\1{2,}", r"\1\1", w)
    return re.sub(r"[A-Za-z]+", fix, text)

def clean_answer(a):
    orig=a
    a=re.sub(r"[ \t]+"," ", a).strip()
    # 2. signature strip (loop: nama dulu, lalu credential)
    for _ in range(2):
        m=RE_NAME.search(a)
        if m and len(a[m.start():].split())<=8:
            a=a[:m.start()].rstrip(" .,;:-"); stat["sig_name"]+=1
        m=RE_CRED.search(a)
        if m and len(a[m.start():].split())<=8 and m.start()>0:
            a=a[:m.start()].rstrip(" .,;:-"); stat["sig_cred"]+=1
    # 2.5 pulihkan batas kalimat (sisip ". " sebelum kata-pembuka terkurasi)
    a2, ncnt = RE_GLUE.subn(". ", a)
    if ncnt: stat["sentence_split"]+=ncnt; a=a2
    # 3. buang kalimat promosi platform ATAU rujukan artikel lain
    sents=re.split(r"(?<=[.!?])\s+", a)
    kept=[s for s in sents if not RE_PLAT.search(s) and not RE_REF.search(s)]
    if len(kept)<len(sents): stat["platform_or_ref"]+=1
    a=" ".join(kept).strip()
    # 4. preposisi nempel
    a2=RE_DIPREP.sub(lambda m:"di "+m.group(1), a); 
    if a2!=a: stat["prep_di"]+=1; a=a2
    a2=RE_KEPREP.sub(lambda m:"ke "+m.group(1), a)
    if a2!=a: stat["prep_ke"]+=1; a=a2
    # 5. triple letters
    a2=collapse_triples(a)
    if a2!=a: stat["triple"]+=1; a=a2
    # 5.5 tanda baca: spasi setelah koma (skip desimal) + normalisasi dll/dsb/dst (koma+titik)
    a2,n=RE_COMMA_SP.subn(", ", a)
    if n: stat["comma_space"]+=n; a=a2
    a2,n=RE_ABBR_COMMA.subn(", ", a)
    if n: stat["abbr_comma"]+=n; a=a2
    a2,n=RE_ABBR_DOT.subn(lambda m: m.group(1)+".", a)
    if n: stat["abbr_dot"]+=n; a=a2
    # 6. typo map
    for k,v in TYPO.items():
        if re.search(r"\b"+k+r"\b", a): a=re.sub(r"\b"+k+r"\b", v, a); stat["typo_map"]+=1
    # 7. titik akhir
    a=a.strip()
    if a and a[-1] not in ".!?\"')]…": a+="."; stat["add_period"]+=1
    if a!=orig.strip(): stat["changed_total"]+=1
    if a!=orig.strip() and len(samples)<200:
        samples.append((orig.strip()[-90:], a[-90:]))
    return a

for split in ["train","val","test"]:
    src=os.path.join(SRC,f"{split}.jsonl"); dst=os.path.join(DST,f"{split}.jsonl")
    n=0
    with open(src,encoding="utf-8") as fi, open(dst,"w",encoding="utf-8") as fo:
        for ln in fi:
            ln=ln.strip()
            if not ln: continue
            rec=json.loads(ln); n+=1
            for m in rec["messages"]:
                if m["role"]=="assistant":
                    m["content"]=clean_answer(m["content"])
            fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
    stat[f"records_{split}"]=n

with open(os.path.join(DST,"CLEANING_REPORT.txt"),"w",encoding="utf-8") as f:
    f.write("LAPORAN PEMBERSIHAN processed_id -> processed_id_clean\n"+"="*50+"\n")
    for k in ["records_train","records_val","records_test","changed_total","sig_name","sig_cred",
              "sentence_split","platform_or_ref","prep_di","prep_ke","triple",
              "comma_space","abbr_comma","abbr_dot","typo_map","add_period"]:
        f.write(f"{k:18}: {stat[k]:,}\n")
with open(os.path.join(DST,"changes_sample.txt"),"w",encoding="utf-8") as f:
    for o,c in samples:
        f.write(f"- BEFORE: ...{o}\n+ AFTER : ...{c}\n\n")

tot=stat['records_train']+stat['records_val']+stat['records_test']
print(f"Selesai. {tot:,} record -> {DST}/")
for k in ["changed_total","sig_name","sig_cred","sentence_split","platform_or_ref",
          "prep_di","prep_ke","triple","comma_space","abbr_comma","abbr_dot","typo_map","add_period"]:
    print(f"  {k:18}: {stat[k]:,}")
