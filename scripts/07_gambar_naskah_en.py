"""
07_gambar_naskah_en.py
Gambar naskah dalam bahasa Inggris (dari CSV hasil notebook 02 dan 03).

Cara pakai (dari folder root repositori):
    python scripts/07_gambar_naskah_en.py
Keluaran (hasil/gambar/):
    Fig_FNR_subgroup.png       FNR per subkelompok, regresi logistik, dua outcome
    Fig_mitigation.png         selisih FNR vs spesifisitas, per strategi mitigasi
    Fig_intersectional.png     heatmap FNR ras/etnis x usia (disglikemia)
    Fig_SHAP.png               rerata kontribusi SHAP, kasus NH Asian vs NH White
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
H = ROOT / "hasil"; G = H / "gambar"; G.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#0b0b0b", "#8a8984", "#e6e5e0"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.labelcolor": INK})
OUT = {"DM": "Undiagnosed diabetes (HbA1c ≥6.5%)", "DISGLIKEMIA": "Undiagnosed dysglycaemia (HbA1c ≥5.7%)"}
ATR = {"Usia": "Age", "Jenis kelamin": "Sex", "Ras/etnis": "Race/ethnicity", "Pendapatan (PIR)": "Income (PIR)"}
GRP = {"Laki-laki": "Men", "Perempuan": "Women", "Tidak diketahui": "Missing", "<1,3": "<1.3", "1,3-<3,5": "1.3 to <3.5",
       ">=3,5": "≥3.5", "20-39": "20–39", "40-59": "40–59", "60+": "≥60"}
URUT = {"Usia": ["20-39", "40-59", "60+"], "Jenis kelamin": ["Laki-laki", "Perempuan"],
        "Ras/etnis": ["Mexican American", "Other Hispanic", "NH White", "NH Black", "NH Asian", "Other/Multiracial"],
        "Pendapatan (PIR)": ["<1,3", "1,3-<3,5", ">=3,5", "Tidak diketahui"]}

# ---------------------------------------------------------------- Fig: FNR by subgroup
aud = pd.read_csv(H / "02_audit_subkelompok.csv"); aud = aud[aud.Model == "Regresi logistik"]
perf = pd.read_csv(H / "02_performa_keseluruhan.csv"); perf = perf[perf.Model == "Regresi logistik"]
baris = [(a, g) for a in URUT for g in URUT[a]]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
for ax, out in zip(axes, OUT):
    sub = aud[aud.Outcome == out].set_index(["Atribut", "Kelompok"])
    ys = np.arange(len(baris))[::-1]
    for y, key in zip(ys, baris):
        r = sub.loc[key]
        c = BLUE if r.Cukup_kasus else "#b8b7b1"
        ax.plot([r.FNR_lo, r.FNR_hi], [y, y], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(r.FNR, y, "o", ms=5.5, color=c, mec="white", mew=1)
    ax.axvline(1 - perf[perf.Outcome == out].Sensitivitas.iloc[0], color=GREY, lw=1, ls="--")
    ax.set_xlim(0, 1); ax.set_xlabel("False negative rate"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(OUT[out], fontsize=8.8, loc="left", color=INK)
axes[0].set_yticks(np.arange(len(baris))[::-1])
axes[0].set_yticklabels([f"{ATR[a]}: {GRP.get(g, g)}" for a, g in baris])
fig.tight_layout(); fig.savefig(G / "Fig_FNR_subgroup.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig: mitigation trade-off
mit = pd.read_csv(H / "03_mitigasi_ringkasan.csv")
STR = [("Baseline", "Baseline", GREY, "o"), ("Reweighing", "Reweighing", BLUE, "s"),
       ("ExpGrad", "Exponentiated gradient", ORANGE, "^"), ("Threshold per kelompok", "Group-specific thresholds", AQUA, "D")]
PANEL = [("DM", "Ras/etnis", "Diabetes, race/ethnicity"), ("DM", "Usia", "Diabetes, age"),
         ("DISGLIKEMIA", "Ras/etnis", "Dysglycaemia, race/ethnicity"), ("DISGLIKEMIA", "Usia", "Dysglycaemia, age")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
for ax, (out, atr, judul) in zip(axes.ravel(), PANEL):
    for kode, nama, warna, mk in STR:
        r = mit[(mit.Outcome == out) & (mit.Atribut == atr) & (mit.Strategi == kode)].iloc[0]
        ax.plot([r["Selisih FNR_lo"], r["Selisih FNR_hi"]], [r.Spesifisitas] * 2, color=warna, lw=1.4, alpha=0.8)
        ax.plot([r["Selisih FNR"]] * 2, [r.Spesifisitas_lo, r.Spesifisitas_hi], color=warna, lw=1.4, alpha=0.8)
        ax.plot(r["Selisih FNR"], r.Spesifisitas, mk, ms=7, color=warna, mec="white", mew=1.2, label=nama)
    ax.set_title(judul, fontsize=8.8, loc="left", color=INK)
    ax.set_xlabel("FNR gap between groups"); ax.set_ylabel("Overall specificity")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(G / "Fig_mitigation.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig: intersectional heatmap
inter = pd.read_csv(H / "02_audit_interseksional.csv")
sub = inter[inter.Outcome == "DISGLIKEMIA"]
ras = URUT["Ras/etnis"]; usia = URUT["Usia"]
nilai = sub.pivot(index="Ras/etnis", columns="Usia", values="FNR").loc[ras, usia]
cukup = sub.pivot(index="Ras/etnis", columns="Usia", values="Cukup_kasus").loc[ras, usia]
seq = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#104281"])
fig, ax = plt.subplots(figsize=(5.6, 3.6))
im = ax.imshow(nilai.where(cukup).values.astype(float), cmap=seq, aspect="auto", vmin=0, vmax=1)
for i in range(len(ras)):
    for j in range(len(usia)):
        if cukup.iloc[i, j]:
            v = nilai.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.45 else INK)
        else:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0efec", edgecolor="#b8b7b1", hatch="///", lw=0))
            ax.text(j, i, "<30 cases", ha="center", va="center", fontsize=7, color=GREY)
ax.set_xticks(range(3)); ax.set_xticklabels([GRP[u] for u in usia]); ax.set_xlabel("Age group, years")
ax.set_yticks(range(len(ras))); ax.set_yticklabels(ras)
for s_ in ax.spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("False negative rate")
fig.tight_layout(); fig.savefig(G / "Fig_intersectional.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig: SHAP
shap = pd.read_csv(H / "03_shap_per_kelompok.csv", index_col=0)
FEAT = {"Lingkar pinggang": "Waist circumference", "BMI": "BMI", "Usia": "Age", "Riwayat hipertensi": "History of hypertension",
        "Tekanan sistolik": "Systolic blood pressure", "Pendidikan": "Education", "Aktif fisik": "Physical activity",
        "Riwayat keluarga DM": "Family history of diabetes", "Perempuan": "Female sex",
        "Tekanan diastolik": "Diastolic blood pressure", "Pernah merokok": "Ever smoked"}
shap = shap.iloc[::-1]
y = np.arange(len(shap)); hgt = 0.38
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.barh(y + hgt / 2, shap["Kasus NH Asian"], height=hgt, color=BLUE, label="NH Asian cases")
ax.barh(y - hgt / 2, shap["Kasus NH White"], height=hgt, color=ORANGE, label="NH White cases")
ax.axvline(0, color=GREY, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([FEAT.get(f, f) for f in shap.index])
ax.set_xlabel("Mean SHAP contribution (log-odds)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(G / "Fig_SHAP.png", dpi=300); plt.close(fig)
print("4 gambar disimpan di", G)
