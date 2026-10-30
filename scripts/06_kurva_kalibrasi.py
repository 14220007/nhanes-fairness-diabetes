"""
06_kurva_kalibrasi.py
Kurva kalibrasi per subkelompok (gambar suplemen S1 dan S2), regresi logistik, prediksi out-of-fold notebook 02.
Titik = rerata prediksi vs proporsi teramati (berbobot survei) per kuantil risiko di dalam kelompok.

Cara pakai (dari folder root repositori), setelah notebook 02 dan skrip 05:
    python scripts/06_kurva_kalibrasi.py
Keluaran: hasil/gambar/S1_kalibrasi_diabetes.png, hasil/gambar/S2_kalibrasi_disglikemia.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GAMBAR = ROOT / "hasil" / "gambar"; GAMBAR.mkdir(parents=True, exist_ok=True)
MIN_KASUS = 30
d = pd.read_csv(ROOT / "data" / "processed" / "nhanes_dm_belum_terdiagnosis.csv")
o = pd.read_csv(ROOT / "data" / "processed" / "prediksi_oof_02.csv")
o = o[o.MODEL == "Regresi logistik"].merge(d[["SEQN", "RAS_ETNIS", "KEL_USIA", "WT_GABUNGAN"]], on="SEQN")
kal = pd.read_csv(ROOT / "hasil" / "04_kalibrasi_per_kelompok.csv")

BIRU, ABU, TINTA, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#e6e5e0"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": ABU, "ytick.color": ABU, "axes.labelcolor": TINTA})
LABEL_USIA = {"20-39": "Age 20–39", "40-59": "Age 40–59", "60+": "Age ≥60"}


def titik_kalibrasi(g, n_bin):
    g = g.sort_values("PROB")
    kum = np.cumsum(g.WT_GABUNGAN.values) / g.WT_GABUNGAN.sum()
    b = np.minimum((kum * n_bin).astype(int), n_bin - 1)
    r = g.assign(B=b).groupby("B").apply(lambda x: pd.Series({
        "pred": np.average(x.PROB, weights=x.WT_GABUNGAN), "obs": np.average(x.Y, weights=x.WT_GABUNGAN)}))
    return r


def gambar(out, n_bin, batas, nama_file, judul):
    m = o[o.OUTCOME == out]
    panel = [("Ras/etnis", "RAS_ETNIS", g, g) for g in sorted(m.RAS_ETNIS.unique())] + \
            [("Usia", "KEL_USIA", g, LABEL_USIA[g]) for g in ["20-39", "40-59", "60+"]]
    panel = [p for p in panel if m[m[p[1]] == p[2]].Y.sum() >= MIN_KASUS]
    kol = 3; bar = int(np.ceil(len(panel) / kol))
    fig, axes = plt.subplots(bar, kol, figsize=(7.2, 2.35 * bar), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (atr, kolom, g, label) in zip(axes, panel):
        r = titik_kalibrasi(m[m[kolom] == g], n_bin)
        ax.plot([0, batas], [0, batas], color="#b8b7b1", lw=1, ls="--", zorder=1)
        ax.plot(r.pred, r.obs, color=BIRU, lw=2, zorder=2)
        ax.plot(r.pred, r.obs, "o", ms=5, color=BIRU, mec="white", mew=1.2, zorder=3)
        k = kal[(kal.Outcome == out) & (kal.Kelompok == g)].iloc[0]
        ax.text(0.97, 0.04, f"{label}\nintercept {k.Intercept:+.2f}\nslope {k.Slope:.2f}".replace("-", "−"),
                transform=ax.transAxes, va="bottom", ha="right", fontsize=7.8, color=TINTA)
        ax.set_xlim(0, batas); ax.set_ylim(0, batas)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax in axes[len(panel):]:
        ax.axis("off")
    for j in range(len(panel) - kol, len(panel)):
        if j >= 0 and j + kol >= len(panel):
            axes[j].tick_params(labelbottom=True)
    fig.supxlabel("Mean predicted risk", fontsize=9); fig.supylabel("Observed proportion", fontsize=9)
    fig.suptitle(judul, x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(GAMBAR / nama_file, dpi=300)
    plt.close(fig)


gambar("DM", 5, 0.12, "S1_kalibrasi_diabetes.png",
       "Undiagnosed diabetes: calibration by subgroup (survey-weighted quintiles of predicted risk)")
gambar("DISGLIKEMIA", 10, 0.90, "S2_kalibrasi_disglikemia.png",
       "Undiagnosed dysglycaemia: calibration by subgroup (survey-weighted deciles of predicted risk)")
print("Gambar S1 dan S2 disimpan.")
