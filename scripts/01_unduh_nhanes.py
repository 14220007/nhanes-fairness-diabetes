"""
01_unduh_nhanes.py
Mengunduh file NHANES (.xpt) yang dibutuhkan penelitian fairness diabetes
dari situs CDC ke folder data/raw.

Cara pakai (dari folder root repositori):
    python scripts\\01_unduh_nhanes.py

Siklus: 2011-2012 (G), 2013-2014 (H), 2015-2016 (I), 2017-Maret 2020 (P, pra-pandemi).
File yang sudah ada tidak diunduh ulang.
"""

from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"

# siklus -> (tahun awal untuk URL, cara memberi nama file)
CYCLES = {
    "2011-2012": ("2011", lambda c: f"{c}_G"),
    "2013-2014": ("2013", lambda c: f"{c}_H"),
    "2015-2016": ("2015", lambda c: f"{c}_I"),
    "2017-2020": ("2017", lambda c: f"P_{c}"),
}

# komponen yang dipakai
COMPONENTS = ["DEMO", "DIQ", "GHB", "BMX", "BPQ", "MCQ", "SMQ", "PAQ"]


def bp_component(cycle):
    # 2017-2020 memakai tensimeter osilometrik (BPXO)
    return "BPXO" if cycle == "2017-2020" else "BPX"


def download(url, dest, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            # file XPT yang benar diawali 'HEADER RECORD'
            if not data.startswith(b"HEADER RECORD"):
                raise ValueError("isi bukan file XPT (kemungkinan URL salah atau halaman error)")
            dest.write_bytes(data)
            return True
        except Exception as e:
            print(f"   percobaan {attempt} gagal: {e}")
            time.sleep(3 * attempt)
    return False


def main():
    failed = []
    for cycle, (year, namer) in CYCLES.items():
        for comp in COMPONENTS + [bp_component(cycle)]:
            name = namer(comp)
            dest = RAW / f"{name}.xpt"
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[ada]    {cycle} {name}")
                continue
            url = BASE.format(year=year, name=name)
            print(f"[unduh]  {cycle} {name}  <- {url}")
            if not download(url, dest):
                failed.append(url)

    print()
    if failed:
        print("Gagal diunduh (unduh manual dari situs NHANES lalu simpan di data\\raw):")
        for u in failed:
            print("  ", u)
    else:
        print(f"Selesai. Semua file tersimpan di {RAW}")


if __name__ == "__main__":
    main()
