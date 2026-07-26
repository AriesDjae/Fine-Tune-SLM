"""
redflag/rules.py — daftar pola gejala kegawatan (RED-FLAG) + tautan sitasinya.

SATU-SATUNYA tempat pola gejala didefinisikan. Setiap aturan WAJIB punya `source`
yang merujuk ke kunci di `redflag/SUMBER_PEDOMAN.md`; aturan tanpa sumber = tidak boleh
masuk (lihat `validate_rules()`, dipanggil saat import).

Prinsip perancangan (argumen naskah):
  - HIGH-RECALL disengaja. Kelewatan satu serangan jantung jauh lebih mahal daripada
    seratus alarm palsu. Pola sengaja longgar; presisi dikorbankan secara sadar.
  - Deteksi di SISI INPUT (pertanyaan pengguna) -> tak perlu menunggu model generate,
    biaya ~0 ms, dan tetap bekerja walau model salah menjawab.
  - Setiap pola bisa diaudit: gejala pemicu bisa didaftar persis dan ditelusuri ke pedoman.

Perumusan pola memakai ISTILAH AWAM (register pasien), bukan istilah pemeriksaan klinis,
karena masukan berasal dari orang awam yang mengetik keluhan di HP.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern

# --------------------------------------------------------------------------------------
# Kategori
# --------------------------------------------------------------------------------------
KATEGORI = {
    "kardiovaskular": "Dugaan sindrom koroner akut / gangguan jantung akut",
    "neurologis": "Dugaan stroke / gangguan neurologis akut",
    "pernapasan": "Gangguan napas berat",
    "perdarahan": "Perdarahan saluran cerna / perdarahan hebat",
    "obstetri": "Tanda bahaya kehamilan",
    "anak": "Tanda bahaya pada bayi & balita",
    "kesadaran": "Penurunan kesadaran / kejang",
    "trauma_keracunan": "Trauma, luka bakar, keracunan, gigitan berbisa",
    "anafilaksis": "Reaksi alergi berat / anafilaksis",
}


@dataclass(frozen=True)
class Rule:
    """Satu pola red-flag.

    id       : identitas stabil (dipakai di tabel hasil & anotasi dokter)
    kategori : kunci di KATEGORI
    source   : kunci sumber di SUMBER_PEDOMAN.md (mis. "S1-angina")
    pattern  : regex (sudah di-compile, IGNORECASE)
    catatan  : penjelasan singkat kenapa pola ini ada / batasannya
    negatable: boleh dibatalkan oleh kata ingkar DI LUAR frasa yang cocok.
               False untuk pola yang penyangkalannya tak masuk akal.
    ingkar_internal:
               True bila frasa gejalanya SENDIRI memuat kata ingkar
               (mis. "tidak simetris", "tidak mau minum", "tidak sadar").
               Untuk pola begini, kata ingkar di dalam frasa JANGAN dibaca
               sebagai penyangkalan — itu bagian dari gejalanya.
    """

    id: str
    kategori: str
    source: str
    pattern: Pattern[str]
    catatan: str = ""
    negatable: bool = True
    ingkar_internal: bool = False
    contoh: tuple[str, ...] = field(default_factory=tuple)


def _r(expr: str) -> Pattern[str]:
    return re.compile(expr, re.IGNORECASE)


# Penghubung longgar antar-frasa: sampai ~40 karakter tanpa akhir kalimat.
_DEKAT = r"[^.!?;\n]{0,40}"

# --------------------------------------------------------------------------------------
# S1-angina — Kardiovaskular (KMK 1186/2022, PPK Angina Pektoris Stabil, hal. 269-271)
# --------------------------------------------------------------------------------------
_KARDIO = [
    Rule(
        id="kardio.nyeri_dada_menjalar",
        kategori="kardiovaskular",
        source="S1-angina",
        pattern=_r(
            r"\b(nyeri|sakit|kaku|ngilu|perih|panas)\b" + _DEKAT + r"\bdada\b" + _DEKAT
            + r"\b(menjalar|menyebar|tembus|merambat|sampai ke)\b"
            r"|\b(dada)\b" + _DEKAT + r"\b(menjalar|tembus)\b" + _DEKAT
            + r"\b(lengan|tangan|bahu|rahang|leher|punggung)\b"
        ),
        catatan="PPK: nyeri dada menjalar ke lengan kiri/punggung/rahang/leher/lengan kanan.",
        contoh=("nyeri dada kiri menjalar ke lengan",),
    ),
    Rule(
        id="kardio.nyeri_dada_tertekan",
        kategori="kardiovaskular",
        source="S1-angina",
        pattern=_r(
            r"\bdada\b" + _DEKAT
            + r"\b(seperti )?(ditekan|tertekan|ditimpa|tertimpa|diremas|diperas|dihimpit|berat sekali|terasa berat)\b"
            r"|\b(ditekan|ditimpa|diremas|diperas|dihimpit)\b" + _DEKAT + r"\bdada\b"
        ),
        catatan="PPK: kualitas nyeri 'seperti tertekan benda berat, diperas'.",
    ),
    Rule(
        id="kardio.nyeri_dada_plus_gejala_penyerta",
        kategori="kardiovaskular",
        source="S1-angina",
        pattern=_r(
            r"\b(nyeri|sakit)\b" + _DEKAT + r"\bdada\b" + _DEKAT
            + r"\b(keringat dingin|sesak|sulit bernapas|susah bernafas|mual|muntah|pucat|pingsan|lemas)\b"
            r"|\b(keringat dingin|sesak napas|sesak nafas|mual|muntah|pucat)\b" + _DEKAT
            + r"\b(nyeri|sakit)\b" + _DEKAT + r"\bdada\b"
        ),
        catatan="PPK: 'Nyeri dada bisa disertai keringat dingin, mual, muntah, sesak dan pucat'.",
    ),
    Rule(
        id="kardio.nyeri_dada_lama",
        kategori="kardiovaskular",
        source="S1-angina",
        pattern=_r(
            r"\b(nyeri|sakit)\b" + _DEKAT + r"\bdada\b" + _DEKAT
            + r"\b(\d{2,3}\s*menit|setengah jam|berjam-jam|tidak hilang|tak kunjung|tidak kunjung|terus menerus|terus-menerus)\b"
        ),
        catatan="PPK: nyeri dada >20 menit -> mungkin sindrom koroner akut.",
    ),
    Rule(
        id="kardio.nyeri_dada_polos",
        kategori="kardiovaskular",
        source="S1-angina",
        pattern=_r(r"\b(nyeri|sakit|ngilu|perih|sesak)\b\s*(di|pada|bagian)?\s*\bdada\b|\bdada\b\s*(terasa\s*)?(nyeri|sakit|perih)\b"),
        catatan=(
            "PPK: nyeri dada adalah keluhan utama angina/SKA. Pola paling longgar di modul ini; "
            "sengaja dipertahankan demi recall — biaya presisinya dilaporkan terpisah di tabel FPR."
        ),
    ),
    Rule(
        id="kardio.serangan_jantung",
        kategori="kardiovaskular",
        source="S1-angina",
        pattern=_r(r"\b(serangan jantung|henti jantung|infark|jantung berhenti)\b"),
        catatan="Pengguna menyebut dugaan serangan jantung secara eksplisit.",
    ),
]

# --------------------------------------------------------------------------------------
# S3-stroke — Neurologis (Kemenkes "SeGeRa Ke RS", 2023)
# --------------------------------------------------------------------------------------
_NEURO = [
    Rule(
        id="neuro.senyum_tidak_simetris",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(
            r"\b(senyum|mulut|wajah|muka|bibir)\w*" + _DEKAT
            + r"\b(tidak simetris|tak simetris|mencong|perot|miring|merot|turun sebelah|jatuh sebelah|"
            r"menceng|melorot sebelah)\b"
        ),
        catatan="Se = senyum tidak simetris.",
        ingkar_internal=True,  # frasa gejalanya sendiri memuat 'tidak simetris'
    ),
    Rule(
        id="neuro.lemah_separuh_badan",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(
            r"\b(lemah|melemah|lumpuh|tidak bisa digerakkan|susah digerakkan|sulit digerakkan|tak bertenaga)\b"
            + _DEKAT + r"\b(separuh|sebelah|satu sisi|sisi kiri|sisi kanan|setengah badan|tubuh kiri|tubuh kanan)\b"
            r"|\b(separuh|sebelah|satu sisi|setengah)\s+(badan|tubuh|anggota (gerak|tubuh))\b" + _DEKAT
            + r"\b(lemah|melemah|lumpuh|kaku|tidak bisa digerakkan|baal|kebas)\b"
            # "tangan/kaki sebelah kiri tiba-tiba lemah" -> sisi disebut LEBIH DULU
            r"|\b(tangan|kaki|lengan|badan|tubuh|anggota gerak)\w*" + _DEKAT
            + r"\b(sebelah|separuh|satu sisi|sisi)\s*(kiri|kanan)?\b" + _DEKAT
            + r"\b(lemah|melemah|lumpuh|kaku|kebas|baal|mati rasa|tidak bisa digerakkan|susah digerakkan)\b"
            r"|\b(kelumpuhan|lumpuh)\b" + _DEKAT + r"\b(mendadak|tiba-tiba)\b"
        ),
        catatan="Ge = gerak separuh anggota tubuh melemah tiba-tiba.",
        ingkar_internal=True,  # memuat 'tidak bisa digerakkan'
    ),
    Rule(
        id="neuro.bicara_pelo",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(
            r"\b(bicara|berbicara|ngomong|suara|lidah)\b" + _DEKAT
            + r"\b(pelo|cadel|tidak jelas|ngga jelas|nggak jelas|tidak nyambung|melantur|susah|sulit|kacau|berat)\b"
            r"|\b(tiba-tiba|mendadak)\b" + _DEKAT + r"\b(tidak (bisa|dapat) (bicara|berbicara)|hilang suara)\b"
            r"|\bbicara pelo\b"
        ),
        catatan="Ra = bicara pelo / tiba-tiba tidak dapat bicara.",
        ingkar_internal=True,  # memuat 'tidak jelas' / 'tidak nyambung'
    ),
    Rule(
        id="neuro.kebas_separuh",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(
            r"\b(kebas|baal|mati rasa|kesemutan)\b" + _DEKAT
            + r"\b(separuh|sebelah|satu sisi|setengah badan|tubuh kiri|tubuh kanan|badan kiri|badan kanan)\b"
        ),
        catatan="Ke = kebas separuh tubuh.",
    ),
    Rule(
        id="neuro.pandangan_kabur_mendadak",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(
            r"\b(pandangan|penglihatan|mata)\b" + _DEKAT + r"\b(kabur|buram|gelap|hilang|buta)\b" + _DEKAT
            + r"\b(tiba-tiba|mendadak|sekonyong|sekejap)\b"
            r"|\b(tiba-tiba|mendadak)\b" + _DEKAT
            + r"\b(pandangan|penglihatan|mata)\b" + _DEKAT + r"\b(kabur|buram|gelap|buta)\b"
        ),
        catatan="R = rabun / pandangan mata kabur tiba-tiba.",
    ),
    Rule(
        id="neuro.sakit_kepala_hebat",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(
            r"\b(sakit|nyeri|pusing)\s+kepala\b" + _DEKAT
            + r"\b(hebat|sangat berat|luar biasa|tak tertahankan|tidak tertahankan|terparah|belum pernah)\b"
            r"|\b(sakit kepala|nyeri kepala)\b" + _DEKAT + r"\b(tiba-tiba|mendadak)\b" + _DEKAT
            + r"\b(hebat|berat|parah)\b"
            # "…yang belum pernah dirasakan sebelumnya" = bagian dari rumusan Kemenkes,
            # ditelan ke dalam span supaya tidak terbaca sebagai penyangkalan.
            r"|\b(sakit|nyeri)\s+kepala\b" + _DEKAT + r"\b(belum pernah|tak pernah|tidak pernah)\b" + _DEKAT
            + r"\b(rasakan|dirasakan|merasakan|alami|dialami|begini|seperti ini)\b"
        ),
        catatan="S = sakit kepala hebat yang muncul tiba-tiba, 'belum pernah dirasakan sebelumnya'.",
        ingkar_internal=True,
    ),
    Rule(
        id="neuro.stroke_disebut",
        kategori="neurologis",
        source="S3-stroke",
        pattern=_r(r"\b(stroke|struk (ringan|otak)|serangan otak)\b"),
        catatan="Pengguna menyebut dugaan stroke secara eksplisit.",
    ),
]

# --------------------------------------------------------------------------------------
# S1-gi-bleed — Perdarahan (KMK 1186/2022, PPK Perdarahan Saluran Cerna, hal. 154-158)
# --------------------------------------------------------------------------------------
_PERDARAHAN = [
    Rule(
        id="perdarahan.muntah_darah",
        kategori="perdarahan",
        source="S1-gi-bleed",
        pattern=_r(
            r"\b(muntah|memuntahkan|munta)\b" + _DEKAT + r"\bdarah\b"
            r"|\bhematemesis\b"
            r"|\bmuntah\b" + _DEKAT + r"\b(seperti )?(bubuk|ampas) kopi\b"
        ),
        catatan="PPK: muntah darah / seperti bubuk kopi = hematemesis.",
    ),
    Rule(
        id="perdarahan.bab_hitam",
        kategori="perdarahan",
        source="S1-gi-bleed",
        pattern=_r(
            r"\b(bab|buang air besar|berak|feses|tinja|kotoran)\b" + _DEKAT
            + r"\b(hitam|kehitaman|seperti ter|seperti aspal|pekat)\b"
            r"|\bmelena\b"
            r"|\b(bab|buang air besar|berak|feses|tinja)\b" + _DEKAT + r"\bdarah\b"
            r"|\bberak darah\b"
        ),
        catatan="PPK: BAB hitam seperti ter/aspal = melena; darah dalam tinja (MTBS: disentri).",
    ),
    Rule(
        id="perdarahan.tidak_berhenti",
        kategori="perdarahan",
        source="S1-gi-bleed",
        pattern=_r(
            r"\b(darah|perdarahan|pendarahan)\b" + _DEKAT
            + r"\b(tidak berhenti|tak berhenti|terus mengalir|deras|banyak sekali|hebat)\b"
            r"|\b(perdarahan|pendarahan) (hebat|masif|banyak)\b"
            r"|\bbatuk\w*" + _DEKAT + r"\b(darah|berdarah)\b"
        ),
        catatan="Perdarahan masif; PPK menyebut komplikasi syok hipovolemia. Batuk darah = hemoptisis.",
        ingkar_internal=True,  # memuat 'tidak berhenti'
    ),
]

# --------------------------------------------------------------------------------------
# S4-hamil — Obstetri (Kemenkes, tanda bahaya kehamilan + Buku KIA 2024)
# --------------------------------------------------------------------------------------
_OBSTETRI = [
    Rule(
        id="obstetri.perdarahan_hamil",
        kategori="obstetri",
        source="S4-hamil",
        pattern=_r(
            r"\b(hamil|kehamilan|mengandung|bumil|kandungan)\b" + _DEKAT
            + r"\b(perdarahan|pendarahan|keluar darah|flek darah|berdarah)\b"
            r"|\b(perdarahan|pendarahan|keluar darah|berdarah)\b" + _DEKAT
            + r"\b(hamil|kehamilan|mengandung|kandungan)\b"
        ),
        catatan="Tanda bahaya kehamilan: pendarahan.",
    ),
    Rule(
        id="obstetri.gerak_janin_berkurang",
        kategori="obstetri",
        source="S4-hamil",
        pattern=_r(
            r"\b(gerak|gerakan|bergerak)\b" + _DEKAT + r"\b(janin|bayi|dedek|debay)\b" + _DEKAT
            + r"\b(berkurang|kurang|melemah|tidak terasa|tak terasa|jarang|hilang|berhenti)\b"
            r"|\b(janin|bayi)\b" + _DEKAT + r"\b(tidak|tak|nggak|ga|gak)\b" + _DEKAT + r"\bbergerak\b"
        ),
        catatan="Tanda bahaya kehamilan: gerakan janin berkurang.",
        ingkar_internal=True,  # frasa ini SENDIRI berisi ingkar ('tidak bergerak')
    ),
    Rule(
        id="obstetri.preeklamsia",
        kategori="obstetri",
        source="S4-hamil",
        pattern=_r(
            r"\b(hamil|kehamilan|mengandung|bumil)\b" + _DEKAT
            + r"\b(bengkak|sembab)\b" + _DEKAT
            + r"\b(pusing|sakit kepala|nyeri ulu hati|pandangan kabur|kejang)\b"
            r"|\b(hamil|kehamilan|bumil)\b" + _DEKAT + r"\bkejang\b"
            r"|\b(hamil|kehamilan|bumil)\b" + _DEKAT + r"\bnyeri ulu hati\b" + _DEKAT
            + r"\b(pandangan kabur|bengkak|sakit kepala)\b"
        ),
        catatan="Tanda bahaya: bengkak DISERTAI pusing/nyeri ulu hati/kejang/pandangan kabur.",
    ),
    Rule(
        id="obstetri.ketuban_pecah",
        kategori="obstetri",
        source="S4-hamil",
        pattern=_r(r"\b(ketuban)\b" + _DEKAT + r"\b(pecah|rembes|merembes|bocor)\b|\bpecah ketuban\b"),
        catatan="Tanda bahaya: pecah ketuban dini.",
    ),
    Rule(
        id="obstetri.demam_muntah_hamil",
        kategori="obstetri",
        source="S4-hamil",
        pattern=_r(
            r"\b(hamil|kehamilan|mengandung|bumil)\b" + _DEKAT
            + r"\b(demam tinggi|muntah terus|muntah terus-menerus|muntah berlebihan|tidak bisa makan sama sekali)\b"
        ),
        catatan="Tanda bahaya: demam tinggi; muntah berlebihan.",
    ),
]

# --------------------------------------------------------------------------------------
# S2 — Anak & bayi (MTBS 2015)
# --------------------------------------------------------------------------------------
_ANAK = [
    Rule(
        id="anak.tidak_mau_minum",
        kategori="anak",
        source="S2-balita-danger",
        pattern=_r(
            r"\b(bayi|balita|anak|batita|dedek|adik)\b" + _DEKAT
            + r"\b(tidak mau|tak mau|nggak mau|ga mau|gak mau|malas|susah|tidak bisa|tak bisa)\b" + _DEKAT
            + r"\b(minum|menyusu|nyusu|asi|nenen)\b"
        ),
        catatan=(
            "MTBS tanda bahaya umum: 'tidak bisa minum atau menyusu'. Kata 'makan' SENGAJA "
            "tidak dimasukkan — sulit makan kronis bukan tanda bahaya umum MTBS."
        ),
        ingkar_internal=True,  # 'tidak mau minum' = gejalanya, bukan penyangkalan
    ),
    Rule(
        id="anak.muntah_semua",
        kategori="anak",
        source="S2-balita-danger",
        pattern=_r(
            r"\b(bayi|balita|anak|batita)\b" + _DEKAT
            + r"\b(memuntahkan semua|muntah semua|muntah terus|muntah setiap|muntah tiap)\b"
        ),
        catatan="MTBS: memuntahkan semua makanan dan/atau minuman.",
    ),
    Rule(
        id="anak.letargis",
        kategori="anak",
        source="S2-balita-danger",
        pattern=_r(
            r"\b(bayi|balita|anak|batita)\b" + _DEKAT
            + r"\b(letargis|lemas sekali|lemas terus|tidak sadar|tak sadar|sulit dibangunkan|susah dibangunkan|tidak merespon|tidak respon|lunglai|loyo)\b"
            r"|\b(bayi|balita|anak)\b" + _DEKAT + r"\b(tidak bergerak|tak bergerak|diam saja)\b"
        ),
        catatan="MTBS: letargis atau tidak sadar; bayi muda tidak bergerak.",
        ingkar_internal=True,
    ),
    Rule(
        id="anak.napas_cepat_tarikan_dada",
        kategori="anak",
        source="S2-pneumonia-berat",
        pattern=_r(
            r"\b(bayi|balita|anak|batita)\b" + _DEKAT
            + r"\b(napas cepat|nafas cepat|napas berat|nafas berat|tarikan dinding dada|dada tertarik|napasnya tersengal|nafasnya tersengal|megap-megap|napas bunyi|nafas bunyi|ngik)\b"
            r"|\b(tarikan dinding dada)\b"
        ),
        catatan="MTBS: tarikan dinding dada ke dalam -> PNEUMONIA BERAT, rujuk segera.",
    ),
    Rule(
        id="anak.sianosis",
        kategori="anak",
        source="S2-balita-danger",
        pattern=_r(
            r"\b(bayi|balita|anak|bibir|kuku|badan)\b" + _DEKAT + r"\b(biru|kebiruan|membiru|sianosis)\b"
            r"|\b(ujung (tangan|jari|kaki))\b" + _DEKAT + r"\b(pucat|dingin|biru)\b"
        ),
        catatan="MTBS: tampak biru (sianosis); ujung tangan & kaki pucat dan dingin.",
    ),
    Rule(
        id="anak.dehidrasi_berat",
        kategori="anak",
        source="S2-diare-berat",
        pattern=_r(
            r"\b(bayi|balita|anak)\b" + _DEKAT
            + r"\b(mata cekung|tidak kencing|tak kencing|tidak pipis|tidak bak|kulit kering|ubun-ubun cekung)\b"
            r"|\b(diare|mencret|bab cair)\b" + _DEKAT
            + r"\b(terus menerus|terus-menerus|lemas|tidak kencing|mata cekung|berhari-hari)\b"
        ),
        catatan="MTBS: diare dehidrasi berat (mata cekung, malas minum, turgor lambat).",
        ingkar_internal=True,  # 'tidak kencing' = gejalanya
    ),
    Rule(
        id="anak.bayi_muda_demam_hipotermia",
        kategori="anak",
        source="S2-bayi-muda",
        pattern=_r(
            r"\b(bayi baru lahir|bayi (\d+|satu|dua|tiga|empat|lima|enam|tujuh) (hari|minggu)|neonatus|bayi < ?2 bulan)\b"
            + _DEKAT + r"\b(demam|panas|dingin|kuning|kejang|tidak mau menyusu|kembung|pusar)\b"
        ),
        catatan="MTBS bayi muda <2 bulan: demam >=37,5 C / suhu <36,5 C / kejang -> rujuk segera.",
    ),
]

# --------------------------------------------------------------------------------------
# Kesadaran & kejang (MTBS + PPK Kejang Demam) — berlaku semua umur
# --------------------------------------------------------------------------------------
_KESADARAN = [
    Rule(
        id="kesadaran.kejang",
        kategori="kesadaran",
        source="S1-kejang-demam",
        pattern=_r(
            r"\b(kejang|kejang-kejang|ayan kambuh|epilepsi kambuh)\b"
            # 'step'/'stuip' = sebutan awam kejang demam anak, TAPI 'step' juga kata Inggris
            # ("One Step HIV") -> wajib ada konteks anak/demam.
            r"|\b(anak|bayi|balita|demam|panas)\b" + _DEKAT + r"\b(step|stuip)\b"
        ),
        catatan=(
            "MTBS tanda bahaya umum & PPK Kejang Demam: kejang = kegawatan. "
            "'step' hanya dihitung bila ada konteks anak/demam (hindari 'One Step ...')."
        ),
    ),
    Rule(
        id="kesadaran.pingsan",
        kategori="kesadaran",
        source="S2-balita-danger",
        pattern=_r(r"\b(pingsan|semaput|koma|hilang kesadaran|kehilangan kesadaran)\b"),
        catatan="MTBS: letargis atau tidak sadar -> penyakit sangat berat.",
        # 'saya tidak pingsan' HARUS bisa membatalkan -> negatable=True (default)
    ),
    Rule(
        id="kesadaran.tidak_sadar",
        kategori="kesadaran",
        source="S2-balita-danger",
        pattern=_r(
            r"\btidak sadarkan diri\b"
            # "tidak sadar BAHWA/KALAU ..." = tidak tahu, bukan penurunan kesadaran
            r"|\b(tidak|tak|nggak|ga|gak) sadar\b(?!\s+(bahwa|kalau|kalo|klo|jika|akan|dengan|dgn|soal|adanya|ada))"
            r"|\b(sulit dibangunkan|susah dibangunkan|tidak bisa dibangunkan|tidak merespon|tidak respon)\b"
        ),
        catatan="MTBS: letargis atau tidak sadar -> penyakit sangat berat.",
        ingkar_internal=True,  # frasa gejalanya memang berbentuk ingkar
        negatable=False,
    ),
    Rule(
        id="kesadaran.henti_napas",
        kategori="pernapasan",
        source="S2-bayi-muda",
        pattern=_r(
            r"\b(henti napas|henti nafas|berhenti bernapas|berhenti bernafas|tidak bernapas|tidak bernafas|"
            r"sesak (berat|hebat|sekali|parah)|sulit bernapas|susah bernapas|susah bernafas|megap-megap|"
            r"tercekik|tersedak parah|napas tersengal|nafas tersengal)\b"
        ),
        catatan="MTBS bayi muda: kejang atau henti napas -> rujuk segera; sesak berat = kegawatan.",
        ingkar_internal=True,  # memuat 'tidak bernapas'
        negatable=False,
    ),
]

# --------------------------------------------------------------------------------------
# Trauma, keracunan, anafilaksis (PPK 1186/2022)
# --------------------------------------------------------------------------------------
_TRAUMA = [
    Rule(
        id="trauma.cedera_berat",
        kategori="trauma_keracunan",
        source="S1-luka-bakar",
        pattern=_r(
            r"\b(kecelakaan|tertabrak|jatuh dari|terjatuh dari|terbentur|benturan keras|patah tulang|tulang menonjol|"
            r"luka menganga|luka dalam|tertusuk|tertimpa|tersengat listrik|kesetrum)\b"
            r"|\b(luka bakar)\b" + _DEKAT + r"\b(luas|lebar|melepuh|dalam|parah)\b"
            r"|\b(kepala)\b" + _DEKAT + r"\b(terbentur|terbentur keras|benjol besar)\b" + _DEKAT
            + r"\b(muntah|pingsan|bingung)\b"
        ),
        catatan="PPK Luka Bakar & kegawatan trauma; cedera kepala dengan muntah/pingsan.",
    ),
    Rule(
        id="trauma.keracunan",
        kategori="trauma_keracunan",
        source="S1-keracunan",
        pattern=_r(
            r"\b(keracunan|overdosis|kebanyakan minum obat|minum baygon|minum obat serangga)\b"
            r"|\b(menelan|tertelan|menenggak|minum|meminum)\s+\w*\s*"
            r"(racun|baterai|deterjen|minyak tanah|bensin|pemutih|pembersih|pestisida|obat nyamuk|"
            r"obat serangga|cairan kimia|silica|sabun)\b"
            r"|\b(gigitan ular|digigit ular|dipatuk ular|gigitan anjing|digigit anjing|digigit kera|"
            r"disengat (tawon|lebah))\b"
        ),
        catatan="PPK Keracunan Makanan; keracunan/gigitan berbisa = kegawatan.",
    ),
    Rule(
        id="anafilaksis.reaksi_berat",
        kategori="anafilaksis",
        source="S1-anafilaksis",
        pattern=_r(
            r"\b(anafilaksis|anafilaktik|syok)\b"
            r"|\b(alergi|bentol|biduran|gatal|bengkak)\b" + _DEKAT
            + r"\b(sesak|sulit bernapas|susah bernapas|bengkak (bibir|lidah|tenggorokan|wajah)|pingsan|tenggorokan tercekik)\b"
            r"|\b(bengkak)\b" + _DEKAT + r"\b(bibir|lidah|tenggorokan)\b" + _DEKAT + r"\b(sesak|sulit menelan)\b"
        ),
        catatan="PPK Anafilaktik: reaksi cepat, serius, mengancam -> pertolongan cepat.",
    ),
]

RULES: tuple[Rule, ...] = tuple(_KARDIO + _NEURO + _PERDARAHAN + _OBSTETRI + _ANAK + _KESADARAN + _TRAUMA)

# Kunci sumber yang sah (harus ada di SUMBER_PEDOMAN.md)
SUMBER_SAH = {
    "S1-angina",
    "S1-gi-bleed",
    "S1-kejang-demam",
    "S1-anafilaksis",
    "S1-luka-bakar",
    "S1-keracunan",
    "S2-balita-danger",
    "S2-pneumonia-berat",
    "S2-diare-berat",
    "S2-bayi-muda",
    "S3-stroke",
    "S4-hamil",
}


def validate_rules() -> None:
    """Tolak aturan tanpa sumber sitasi / kategori tak dikenal / id duplikat."""
    seen = set()
    for rule in RULES:
        if rule.id in seen:
            raise ValueError(f"id aturan duplikat: {rule.id}")
        seen.add(rule.id)
        if rule.kategori not in KATEGORI:
            raise ValueError(f"{rule.id}: kategori tak dikenal '{rule.kategori}'")
        if rule.source not in SUMBER_SAH:
            raise ValueError(
                f"{rule.id}: sumber '{rule.source}' tidak ada di SUMBER_PEDOMAN.md. "
                "Setiap pola WAJIB bisa disitasi."
            )


validate_rules()
