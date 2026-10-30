"""
05_kalibrasi_per_kelompok.py
Calibration intercept (calibration-in-the-large, slope fixed at 1) and calibration slope per subgroup
for the logistic regression model, from out-of-fold predictions of notebook 02.
95% CI: Rao-Wu survey bootstrap (n_h - 1 PSUs per stratum).

Cara pakai (dari folder root repositori), setelah notebook 02:
    python scripts/05_kalibrasi_per_kelompok.py [jumlah_replikasi]

Keluaran: hasil/04_kalibrasi_per_kelompok.csv dan hasil/04_bias_selisih_fnr.csv
(cek bias ke atas statistik selisih FNR tertinggi-terendah per ras/etnis)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MIN_KASUS = 30
rng = np.random.default_rng(2027)

d = pd.read_csv(ROOT / "data" / "processed" / "nhanes_dm_belum_terdiagnosis.csv")
o = pd.read_csv(ROOT / "data" / "processed" / "prediksi_oof_02.csv")
o = o[o.MODEL == "Regresi logistik"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def kalibrasi(y, p, w):
    lp = logit(p)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y, sample_weight=w / w.mean()).coef_[0][0]
    a = 0.0
    for _ in range(50):  # Newton: offset model logit(P) = a + lp
        q = 1 / (1 + np.exp(-(lp + a)))
        a += np.sum(w * (y - q)) / np.sum(w * q * (1 - q))
    return a, slope


def bobot_bootstrap(m):
    k = np.ones(len(m))
    for st, idx in m.groupby("SDMVSTRA").groups.items():
        psu = m.loc[idx, "SDMVPSU"].values
        u = np.unique(psu)
        n = len(u)
        if n < 2:
            continue
        tarik = rng.choice(u, n - 1, replace=True)
        faktor = {x: np.sum(tarik == x) * n / (n - 1) for x in u}
        k[m.index.get_indexer(idx)] = [faktor[x] for x in psu]
    return m["WT_GABUNGAN"].values * k


baris, bias = [], []
for out in ["DM", "DISGLIKEMIA"]:
    m = o[o.OUTCOME == out].merge(d[["SEQN", "RAS_ETNIS", "KEL_USIA", "WT_GABUNGAN", "SDMVPSU", "SDMVSTRA"]],
                                  on="SEQN").reset_index(drop=True)
    kelompok = [("Keseluruhan", "Semua", np.ones(len(m), bool))]
    for atr, nama in [("RAS_ETNIS", "Ras/etnis"), ("KEL_USIA", "Usia")]:
        for g in sorted(m[atr].unique()):
            msk = (m[atr] == g).values
            if m.Y[msk].sum() >= MIN_KASUS:
                kelompok.append((nama, g, msk))
    y, p, w = m.Y.values, m.PROB.values, m.WT_GABUNGAN.values
    titik = [kalibrasi(y[k], p[k], w[k]) for _, _, k in kelompok]
    def selisih_fnr(wt):
        f = []
        for g in m.RAS_ETNIS.unique():
            k = (m.RAS_ETNIS == g).values & (y == 1)
            if k.sum() >= MIN_KASUS:
                f.append(1 - np.sum(wt[k] * m.PRED.values[k]) / np.sum(wt[k]))
        return max(f) - min(f)
    rep, gap = [], []
    for b in range(B):
        wb = bobot_bootstrap(m)
        rep.append([kalibrasi(y[k], p[k], wb[k]) for _, _, k in kelompok])
        gap.append(selisih_fnr(wb))
    bias.append(dict(Outcome=out, Selisih_titik=selisih_fnr(w), Rerata_bootstrap=np.mean(gap),
                     Median_bootstrap=np.median(gap), Bias=np.mean(gap) - selisih_fnr(w)))
    rep = np.array(rep)  # B x kelompok x 2
    for j, (atr, g, k) in enumerate(kelompok):
        lo, hi = np.percentile(rep[:, j, :], [2.5, 97.5], axis=0)
        baris.append(dict(Outcome=out, Atribut=atr, Kelompok=g, Kasus=int(y[k].sum()),
                          Intercept=titik[j][0], Intercept_lo=lo[0], Intercept_hi=hi[0],
                          Slope=titik[j][1], Slope_lo=lo[1], Slope_hi=hi[1]))
    print(out, "selesai")

hasil = pd.DataFrame(baris)
hasil.to_csv(ROOT / "hasil" / "04_kalibrasi_per_kelompok.csv", index=False)
print(hasil.round(2).to_string(index=False))
pd.DataFrame(bias).to_csv(ROOT / "hasil" / "04_bias_selisih_fnr.csv", index=False)
print(pd.DataFrame(bias).round(3).to_string(index=False))
