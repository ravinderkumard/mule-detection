# Mule Account Detection

Detecting money-mule accounts from transaction graphs, using two public benchmarks:

| Dataset | Type | Size | Task |
|---|---|---|---|
| **IBM AMLworld – HI-Small** | Synthetic bank transfers (IBM Research / ETH Zurich) | ~5.08M txns, ~515K accounts, ~0.1% laundering | Transaction → account-level mule scoring |
| **Elliptic** | Real Bitcoin transaction graph | 203,769 nodes, 234,355 edges, 49 time steps | Illicit node classification |

## Project layout

```
mule-detection/
├── configs/datasets.yaml     # dataset slugs, expected stats, split rules
├── data/
│   ├── raw/                  # Kaggle downloads (git-ignored)
│   ├── interim/              # parquet caches from verify_data.py
│   └── processed/            # model-ready features
├── notebooks/                # EDA + experiments
├── reports/                  # figures, metrics
├── scripts/
│   ├── download_data.py      # pulls both datasets via Kaggle API
│   └── verify_data.py        # sanity checks + parquet conversion
├── src/muledetect/           # reusable package code
├── requirements.txt          # core stack
└── requirements-graph.txt    # optional PyTorch + PyG (GNN stage)
```

## Day-1 setup

### 1. Repo
```bash
cd mule-detection
git init -b main
git add . && git commit -m "Project scaffold"
# Create an empty repo on GitHub named mule-detection, then:
git remote add origin https://github.com/<your-username>/mule-detection.git
git push -u origin main
```

### 2. Python environment (Python 3.11)
**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name mule-detection
```
**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name mule-detection
```
(Conda alternative: `conda create -n mule python=3.11 -y && conda activate mule`, then the same pip lines.)

### 3. Kaggle account + API token
1. Sign up at kaggle.com and verify your phone number (needed for some downloads).
2. Go to **Settings → API → Create New Token**. This downloads `kaggle.json`.
3. Move it into place:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json` then `chmod 600 ~/.kaggle/kaggle.json`
4. Open both dataset pages once in your browser and accept any terms:
   - https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
   - https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
5. Test: `kaggle datasets list -s elliptic`

### 4. Download + verify
```bash
python scripts/download_data.py     # HI-Small only (not the full 16 GB AMLworld bundle) + Elliptic
python scripts/verify_data.py       # prints stats, writes data/interim/*.parquet
```
Expect roughly 1.2 GB of disk for raw files plus parquet caches.

**Manual fallback:** if the API misbehaves, download from the two pages above and drop
`HI-Small_Trans.csv` / `HI-Small_Patterns.txt` into `data/raw/ibm_aml/` and the three
`elliptic_txs_*.csv` files into `data/raw/elliptic/`, then run `verify_data.py`.

## Notes on the data
- **AMLworld account identity = bank + account.** The raw CSV has two columns both named
  `Account`; `verify_data.py` renames them to `from_account` / `to_account` and builds `src`/`dst` keys.
- `HI-Small_Patterns.txt` lists laundering attempts by typology (fan-in, fan-out, cycle,
  scatter-gather, gather-scatter, bipartite, stack, random) — useful for explaining *why* an account is flagged.
- **Elliptic labels:** `1` = illicit, `2` = licit, `unknown` ≈ 77% of nodes. Features file has no header:
  column 0 = txId, column 1 = time step, then 165 anonymised features (94 local + 72 aggregated in the paper's description).
- Use a **temporal split** on Elliptic (train 1–34, test 35–49); random splits leak and inflate scores.

## References
- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 Datasets & Benchmarks.
- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD 2019 workshop.
