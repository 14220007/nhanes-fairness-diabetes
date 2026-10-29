"""
02_bangun_dataset.py
Menggabungkan file NHANES di data/raw menjadi satu dataset analisis:
dewasa >= 20 tahun, diperiksa di MEC, tidak hamil, belum pernah didiagnosis
diabetes, dengan outcome diabetes belum terdiagnosis (HbA1c >= 6,5%).

Cara pakai (dari folder root repositori), setelah 01_unduh_nhanes.py:
    python scripts\\02_bangun_dataset.py

Keluaran:
    data/processed/nhanes_dm_belum_terdiagnosis.csv  -> dataset analisis
    hasil/alur_sampel.csv                            -> jumlah sampel tiap tahap eksklusi
    hasil/prevalensi_per_kelompok.csv                -> prevalensi berbobot per kelompok
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DATA = ROOT / "data" / "processed"
OUT_RES = ROOT / "hasil"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

HBA1C_CUTOFF = 6.5  # %, kriteria ADA

# siklus -> (fungsi nama file, nama kolom bobot MEC, lama siklus dalam tahun)
CYCLES = {
    "2011-2012": (lambda c: f"{c}_G", "WTMEC2YR", 2.0),
    "2013-2014": (lambda c: f"{c}_H", "WTMEC2YR", 2.0),
    "2015-2016": (lambda c: f"{c}_I", "WTMEC2YR", 2.0),
    "2017-2020": (lambda c: f"P_{c}", "WTMECPRP", 3.2),
}
TOTAL_YEARS = sum(v[2] for v in CYCLES.values())  # 9,2 tahun

KOLOM = {
    "DEMO": ["SEQN", "RIDSTATR", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR",
             "DMDEDUC2", "RIDEXPRG", "SDMVPSU", "SDMVSTRA"],
    "DIQ": ["SEQN", "DIQ010", "DIQ050", "DIQ070"],
    "GHB": ["SEQN", "LBXGH"],
    "BMX": ["SEQN", "BMXBMI", "BMXWAIST"],
    "BPQ": ["SEQN", "BPQ020"],
    "MCQ": ["SEQN", "MCQ300C"],
    "SMQ": ["SEQN", "SMQ020"],
    "PAQ": ["SEQN", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"],
}


def baca(nama_file):
    path = RAW / f"{nama_file}.xpt"
    if not path.exists():
        raise FileNotFoundError(f"File {path} belum ada. Jalankan 01_unduh_nhanes.py dulu.")
    return pd.read_sas(path, format="xport", encoding="utf-8")


def ambil(df, kolom):
    """Ambil kolom yang ada; kolom yang tidak ada diisi NaN (tidak semua siklus identik)."""
    out = pd.DataFrame(index=df.index)
    for k in kolom:
        out[k] = df[k] if k in df.columns else np.nan
    return out


def tekanan_darah(namer, siklus):
    if siklus == "2017-2020":
        bp = baca(namer("BPXO"))
        sys_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in bp.columns]
        dia_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in bp.columns]
    else:
        bp = baca(namer("BPX"))
        sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in bp.columns]
        dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in bp.columns]
    dia = bp[dia_cols].where(bp[dia_cols] > 0)  # diastolik 0 = tidak valid
    return pd.DataFrame({
        "SEQN": bp["SEQN"],
        "SBP_MEAN": bp[sys_cols].mean(axis=1),
        "DBP_MEAN": dia.mean(axis=1),
    })


def satu_siklus(siklus, namer, kolom_bobot, lama):
    demo = baca(namer("DEMO"))
    df = ambil(demo, KOLOM["DEMO"])
    df["WT_MEC_ASLI"] = demo[kolom_bobot]
    for komp in ["DIQ", "GHB", "BMX", "BPQ", "MCQ", "SMQ", "PAQ"]:
        df = df.merge(ambil(baca(namer(komp)), KOLOM[komp]), on="SEQN", how="left")
    df = df.merge(tekanan_darah(namer, siklus), on="SEQN", how="left")
    df["SIKLUS"] = siklus
    # bobot gabungan sesuai panduan NHANES untuk menggabung siklus 2011-2016 dan 2017-Maret 2020
    df["WT_GABUNGAN"] = df["WT_MEC_ASLI"] * lama / TOTAL_YEARS
    return df


def kode_hilang(s, kode=(7, 9, 77, 99, 777, 999)):
    return s.where(~s.isin(kode))


def ya_tidak(s):
    """1 = ya, 2 = tidak -> 1/0; kode lain -> NaN."""
    return s.map({1: 1.0, 2: 0.0})


def rekode(df):
    d = pd.DataFrame({"SEQN": df["SEQN"].astype("int64"), "SIKLUS": df["SIKLUS"]})

    # outcome
    d["HBA1C"] = df["LBXGH"]
    d["DM_BELUM_TERDIAGNOSIS"] = (df["LBXGH"] >= HBA1C_CUTOFF).astype(int)

    # atribut sensitif
    d["USIA"] = df["RIDAGEYR"]
    d["KEL_USIA"] = pd.cut(df["RIDAGEYR"], bins=[19, 39, 59, 200], labels=["20-39", "40-59", "60+"])
    d["JENIS_KELAMIN"] = df["RIAGENDR"].map({1: "Laki-laki", 2: "Perempuan"})
    d["RAS_ETNIS"] = df["RIDRETH3"].map({
        1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
        4: "NH Black", 6: "NH Asian", 7: "Other/Multiracial",
    })
    d["PIR"] = df["INDFMPIR"]
    d["KEL_PENDAPATAN"] = pd.cut(
        df["INDFMPIR"], bins=[-0.01, 1.3, 3.5, 10], right=False,
        labels=["<1,3", "1,3-<3,5", ">=3,5"],
    ).astype(object).fillna("Tidak diketahui")

    # prediktor non-laboratorium
    d["BMI"] = df["BMXBMI"]
    d["LINGKAR_PINGGANG"] = df["BMXWAIST"]
    d["SBP"] = df["SBP_MEAN"]
    d["DBP"] = df["DBP_MEAN"]
    d["RIWAYAT_HIPERTENSI"] = ya_tidak(df["BPQ020"])
    d["RIWAYAT_KELUARGA_DM"] = ya_tidak(df["MCQ300C"])
    d["PERNAH_MEROKOK"] = ya_tidak(df["SMQ020"])
    pa = pd.concat([ya_tidak(df[c]) for c in ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]], axis=1)
    d["AKTIF_FISIK"] = pa.max(axis=1)  # 1 jika ada aktivitas sedang/berat apa pun
    d["PENDIDIKAN"] = kode_hilang(df["DMDEDUC2"])
    d["PRADIABETES_DIBERITAHU"] = (df["DIQ010"] == 3).astype(int)

    # desain survei
    d["WT_GABUNGAN"] = df["WT_GABUNGAN"]
    d["SDMVPSU"] = df["SDMVPSU"]
    d["SDMVSTRA"] = df["SDMVSTRA"]
    return d


def main():
    gabung = pd.concat(
        [satu_siklus(s, n, w, l) for s, (n, w, l) in CYCLES.items()],
        ignore_index=True,
    )

    alur = [("Seluruh peserta NHANES 2011-Maret 2020", len(gabung))]

    def saring(df, mask, ket):
        df = df[mask.fillna(False)]
        alur.append((ket, len(df)))
        return df

    df = gabung
    df = saring(df, df["RIDSTATR"] == 2, "Diperiksa di MEC")
    df = saring(df, df["RIDAGEYR"] >= 20, "Usia >= 20 tahun")
    df = saring(df, df["RIDEXPRG"] != 1, "Tidak hamil")
    df = saring(df, df["DIQ010"].isin([2, 3]), "Belum pernah didiagnosis diabetes (DIQ010 = tidak/batas)")
    df = saring(df, (df["DIQ050"] != 1) & (df["DIQ070"] != 1), "Tidak memakai insulin atau obat diabetes")
    df = saring(df, df["LBXGH"].notna(), "Memiliki data HbA1c")
    df = saring(df, df["WT_GABUNGAN"] > 0, "Bobot MEC > 0")

    data = rekode(df)
    data.to_csv(OUT_DATA / "nhanes_dm_belum_terdiagnosis.csv", index=False)
    pd.DataFrame(alur, columns=["Tahap", "N"]).to_csv(OUT_RES / "alur_sampel.csv", index=False)

    # prevalensi berbobot per kelompok (cek cepat, belum dengan SE desain survei)
    baris = []
    for kol in ["KEL_USIA", "JENIS_KELAMIN", "RAS_ETNIS", "KEL_PENDAPATAN", "SIKLUS"]:
        for nilai, g in data.groupby(kol, observed=True):
            w = g["WT_GABUNGAN"]
            baris.append({
                "Atribut": kol, "Kelompok": nilai, "N": len(g),
                "Kasus": int(g["DM_BELUM_TERDIAGNOSIS"].sum()),
                "Prevalensi_berbobot_%": round(100 * np.average(g["DM_BELUM_TERDIAGNOSIS"], weights=w), 2),
            })
    prev = pd.DataFrame(baris)
    prev.to_csv(OUT_RES / "prevalensi_per_kelompok.csv", index=False)

    total = np.average(data["DM_BELUM_TERDIAGNOSIS"], weights=data["WT_GABUNGAN"])
    print("Alur sampel:")
    for ket, n in alur:
        print(f"  {n:>7,}  {ket}")
    print(f"\nKasus diabetes belum terdiagnosis: {int(data['DM_BELUM_TERDIAGNOSIS'].sum()):,} "
          f"dari {len(data):,} (prevalensi berbobot {100 * total:.2f}%)")
    print("\nPrevalensi per kelompok:")
    print(prev.to_string(index=False))
    print(f"\nDataset disimpan di {OUT_DATA / 'nhanes_dm_belum_terdiagnosis.csv'}")


if __name__ == "__main__":
    main()
