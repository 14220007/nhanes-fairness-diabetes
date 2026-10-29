"""
04_karakteristik_dan_alur.py
Membuat (1) tabel karakteristik peserta berbobot survei dan (2) diagram alur peserta (Fig. 1).

Cara pakai (dari folder root repositori):
    python scripts\\04_karakteristik_dan_alur.py

Keluaran:
    hasil/00_karakteristik_peserta.csv
    hasil/gambar/01_alur_peserta.png
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "nhanes_dm_belum_terdiagnosis.csv"
HASIL = ROOT / "hasil"
GAMBAR = HASIL / "gambar"
GAMBAR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(DATA)
d.loc[d["DBP"] < 30, "DBP"] = np.nan
d.loc[d["SBP"] < 60, "SBP"] = np.nan

KELOMPOK = {"Total": d,
            "Undiagnosed diabetes": d[d["DM_BELUM_TERDIAGNOSIS"] == 1],
            "No diabetes": d[d["DM_BELUM_TERDIAGNOSIS"] == 0]}


def rerata(g, kol):
    m = g[kol].notna()
    w = g.loc[m, "WT_GABUNGAN"]; x = g.loc[m, kol]
    mu = np.average(x, weights=w)
    sd = np.sqrt(np.average((x - mu) ** 2, weights=w))
    return f"{mu:.1f} ({sd:.1f})"


def persen(g, kol, nilai):
    m = g[kol].notna()
    w = g.loc[m, "WT_GABUNGAN"]
    return f"{100 * w[g.loc[m, kol] == nilai].sum() / w.sum():.1f}"


baris = [("Participants, n (unweighted)", {k: f"{len(g):,}" for k, g in KELOMPOK.items()})]
for label, kol in [("Age, years", "USIA"), ("BMI, kg/m²", "BMI"), ("Waist circumference, cm", "LINGKAR_PINGGANG"),
                   ("Systolic blood pressure, mmHg", "SBP"), ("Diastolic blood pressure, mmHg", "DBP"),
                   ("HbA1c, %", "HBA1C")]:
    baris.append((f"{label}, mean (SD)", {k: rerata(g, kol) for k, g in KELOMPOK.items()}))
baris.append(("Women, %", {k: persen(g, "JENIS_KELAMIN", "Perempuan") for k, g in KELOMPOK.items()}))
for a in ["20-39", "40-59", "60+"]:
    baris.append((f"Age {a.replace('-', '–').replace('60+', '≥60')} years, %",
                  {k: persen(g, "KEL_USIA", a) for k, g in KELOMPOK.items()}))
for r in ["Mexican American", "Other Hispanic", "NH White", "NH Black", "NH Asian", "Other/Multiracial"]:
    baris.append((f"{r}, %", {k: persen(g, "RAS_ETNIS", r) for k, g in KELOMPOK.items()}))
for p, lab in [("<1,3", "<1.3"), ("1,3-<3,5", "1.3 to <3.5"), (">=3,5", "≥3.5"), ("Tidak diketahui", "missing")]:
    baris.append((f"Income-to-poverty ratio {lab}, %", {k: persen(g, "KEL_PENDAPATAN", p) for k, g in KELOMPOK.items()}))
for label, kol in [("History of hypertension, %", "RIWAYAT_HIPERTENSI"), ("Family history of diabetes, %", "RIWAYAT_KELUARGA_DM"),
                   ("Ever smoked ≥100 cigarettes, %", "PERNAH_MEROKOK"), ("Any moderate or vigorous activity, %", "AKTIF_FISIK")]:
    baris.append((label, {k: persen(g, kol, 1.0) for k, g in KELOMPOK.items()}))
baris.append(("Education: college graduate or above, %", {k: persen(g, "PENDIDIKAN", 5.0) for k, g in KELOMPOK.items()}))

tabel = pd.DataFrame([{"Characteristic": lab, **v} for lab, v in baris])
tabel.to_csv(HASIL / "00_karakteristik_peserta.csv", index=False, encoding="utf-8-sig")
print(tabel.to_string(index=False))

# ---------------------------------------------------------------- Fig. 1 alur peserta
alur = pd.read_csv(HASIL / "alur_sampel.csv")
n = alur["N"].tolist()
kiri = ["NHANES participants, 2011–March 2020\n(4 survey cycles)",
        "Examined at mobile examination centre",
        "Aged ≥20 years",
        "Not pregnant",
        "No previous diabetes diagnosis",
        "Not using insulin or oral glucose-lowering drugs",
        "Analytic sample: valid HbA1c and MEC weight"]
idx = [0, 1, 2, 3, 4, 5, 6]
eksklusi = ["Not examined", "Aged <20 years", "Pregnant", "Previously diagnosed diabetes\nor unknown status",
            "Using insulin or oral glucose-lowering drugs", "Missing HbA1c"]

fig, ax = plt.subplots(figsize=(7.2, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ys = np.linspace(9.3, 1.9, len(kiri))
for i, (y, t) in enumerate(zip(ys, kiri)):
    tebal = i == len(kiri) - 1
    ax.add_patch(plt.Rectangle((0.3, y - 0.42), 5.0, 0.84, fill=True, facecolor="#e8eef8" if tebal else "white",
                               edgecolor="#2a78d6" if tebal else "#52514e", lw=1.6 if tebal else 1.0))
    ax.text(2.8, y + 0.12, t, ha="center", va="center", fontsize=8.5, fontweight="bold" if tebal else "normal")
    ax.text(2.8, y - 0.22, f"n = {n[idx[i]]:,}", ha="center", va="center", fontsize=8.5)
    if i < len(kiri) - 1:
        ax.annotate("", xy=(2.8, ys[i + 1] + 0.43), xytext=(2.8, y - 0.43),
                    arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1))
        yy = (y + ys[i + 1]) / 2
        ax.plot([2.8, 6.0], [yy, yy], color="#52514e", lw=0.8)
        ax.add_patch(plt.Rectangle((6.0, yy - 0.28), 3.7, 0.56, fill=False, edgecolor="#8a8984", lw=0.8))
        ax.text(7.85, yy, f"{eksklusi[i]}\nn = {n[idx[i]] - n[idx[i + 1]]:,}", ha="center", va="center", fontsize=7.8)
y_last = ys[-1]
for j, (t, k) in enumerate([("Undiagnosed diabetes\n(HbA1c ≥6.5%)", int(d["DM_BELUM_TERDIAGNOSIS"].sum())),
                            ("Undiagnosed dysglycaemia\n(HbA1c ≥5.7%)", int((d["HBA1C"] >= 5.7).sum()))]):
    x0 = 0.3 + j * 2.6
    ax.annotate("", xy=(x0 + 1.2, 1.05), xytext=(2.8, y_last - 0.43), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1))
    ax.add_patch(plt.Rectangle((x0, 0.25), 2.4, 0.8, fill=False, edgecolor="#52514e", lw=1))
    ax.text(x0 + 1.2, 0.65, f"{t}\nn = {k:,}", ha="center", va="center", fontsize=7.6)
fig.tight_layout()
fig.savefig(GAMBAR / "01_alur_peserta.png", dpi=300)
print("Gambar alur disimpan.")
