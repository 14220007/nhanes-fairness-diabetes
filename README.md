# Fairness audit of machine learning screening for undiagnosed diabetes (NHANES 2011–2020)

Code and aggregate results for the manuscript:

> Rumintarsih A. *Subgroup miscalibration drives racial and age disparities in machine learning screening for undiagnosed diabetes: a fairness audit and mitigation study using NHANES 2011–2020.* (under review)

The study audits four models (logistic regression, random forest, XGBoost, LightGBM) that predict undiagnosed diabetes (HbA1c ≥6.5%) and undiagnosed dysglycaemia (HbA1c ≥5.7%) from eleven non-laboratory predictors. All metrics use NHANES survey weights; 95% confidence intervals come from a Rao–Wu survey bootstrap. Three mitigation strategies are compared: reweighing, exponentiated gradient (Fairlearn) and group-specific thresholds.

## Repository structure

| Path | Content |
| --- | --- |
| `notebooks/01_data_nhanes.ipynb` | Downloads NHANES files (2011–2012, 2013–2014, 2015–2016, 2017–March 2020) and builds the analytic dataset |
| `notebooks/02_baseline_audit_fairness.ipynb` | Nested cross-validation, survey-weighted recalibration, subgroup and intersectional fairness audit |
| `notebooks/03_mitigation_shap.ipynb` | Bias mitigation (reweighing, exponentiated gradient, group thresholds) and linear SHAP |
| `notebooks/04_sensitivitas_glukosa_puasa.ipynb` | Sensitivity analysis in the fasting subsample: HbA1c ≥6.5% or fasting plasma glucose ≥126 mg/dL |
| `scripts/` | Script versions of the data steps, plus `04_karakteristik_dan_alur.py` (Table 1 and participant flow figure) |
| `scripts/05_kalibrasi_per_kelompok.py` | Calibration intercept and slope by subgroup with survey-bootstrap CIs (Table 5), and the bias check for the FNR gap statistic |
| `scripts/06_kurva_kalibrasi.py` | Calibration curves by subgroup (Supplementary Figs. S1–S2) |
| `hasil/` | Aggregate result tables (CSV) reported in the manuscript |
| `hasil/gambar/` | Figures 1–5 |
| `data/` | Empty; filled when notebook 01 is run (NHANES data are not redistributed here) |
| `scopus_search_queries.txt` | Literature search queries |

Code comments and variable names are in Indonesian (e.g. `hasil` = results, `gambar` = figures, `USIA` = age, `RAS_ETNIS` = race/ethnicity).

## How to reproduce

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run the notebooks in order (01 → 02 → 03) from the `notebooks/` folder. Notebook 01 needs internet access to the CDC NHANES website. In notebooks 02 and 03, set `CEPAT = True` for a quick test run; `CEPAT = False` (default) reproduces the manuscript (500 bootstrap replicates; about one hour on a laptop). Then run:

```bash
python scripts/04_karakteristik_dan_alur.py
python scripts/05_kalibrasi_per_kelompok.py
python scripts/06_kurva_kalibrasi.py
```

Random seeds are fixed, so results should match the CSV files in `hasil/` up to minor differences between library versions.

## Data

NHANES data are public and de-identified, provided by the National Center for Health Statistics (https://wwwn.cdc.gov/nchs/nhanes/). NHANES protocols were approved by the NCHS Research Ethics Review Board.

## Citation

See `CITATION.cff`, or cite the Zenodo DOI of the release.

## License

MIT (see `LICENSE`).
