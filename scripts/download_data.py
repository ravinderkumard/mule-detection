"""Download IBM AMLworld (HI-Small) and Elliptic from Kaggle into data/raw/.

Prereqs:
  1. Kaggle account + API token (Kaggle -> Settings -> API -> "Generate New Token").
  2. Provide the token via any of (kaggle >= 2.x):
       - KAGGLE_API_TOKEN in .env (or exported in the shell)
       - ~/.kaggle/access_token
       - `kaggle auth login` (OAuth)
     Legacy KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json also still work.
  3. Accept the dataset terms once on each Kaggle page if prompted (403 otherwise).

Usage:
  python scripts/download_data.py            # both datasets
  python scripts/download_data.py --only aml
  python scripts/download_data.py --only elliptic
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # must run before kaggle reads its credentials
RAW = ROOT / "data" / "raw"

AML_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
AML_FILES = ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]  # ~475 MB + small

ELLIPTIC_SLUG = "ellipticco/elliptic-data-set"
ELLIPTIC_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]


def get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle package missing -> pip install -r requirements.txt")
    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as e:  # noqa: BLE001 - kaggle calls exit(1) on auth failure
        sys.exit(
            f"Kaggle auth failed: {e}\n"
            "Set KAGGLE_API_TOKEN in .env, save it to ~/.kaggle/access_token, "
            "or run `kaggle auth login`."
        )
    return api


def unzip_all(folder: Path) -> None:
    for z in folder.rglob("*.zip"):
        print(f"  unzipping {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
        z.unlink()


def find(folder: Path, name: str) -> Path | None:
    hits = list(folder.rglob(name))
    return hits[0] if hits else None


def download_aml(api) -> None:
    out = RAW / "ibm_aml"
    out.mkdir(parents=True, exist_ok=True)
    for f in AML_FILES:
        if find(out, f):
            print(f"[aml] {f} already present, skipping")
            continue
        print(f"[aml] downloading {f} ...")
        api.dataset_download_file(AML_SLUG, f, path=str(out), quiet=False)
    unzip_all(out)


def download_elliptic(api) -> None:
    out = RAW / "elliptic"
    out.mkdir(parents=True, exist_ok=True)
    if all(find(out, f) for f in ELLIPTIC_FILES):
        print("[elliptic] all files present, skipping")
        return
    print("[elliptic] downloading full dataset (~700 MB unzipped) ...")
    api.dataset_download_files(ELLIPTIC_SLUG, path=str(out), unzip=True, quiet=False)
    unzip_all(out)


def report() -> None:
    print("\nFiles in data/raw:")
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.name != ".gitkeep":
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:,.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["aml", "elliptic"])
    args = ap.parse_args()
    api = get_api()
    if args.only in (None, "aml"):
        download_aml(api)
    if args.only in (None, "elliptic"):
        download_elliptic(api)
    report()
    print("\nNext: python scripts/verify_data.py")


if __name__ == "__main__":
    main()
