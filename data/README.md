# ShopSense — Data Directory

## Dataset

**Name:** Amazon Reviews 2023 — Musical Instruments  
**Source:** McAuley Lab, UC San Diego  
**HuggingFace:** [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)  
**Official site:** [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)  
**Paper:** Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024

## Files

| File | Description | Committed? |
|---|---|---|
| `samples/reviews_sample.csv` | Small cleaned sample (~5K rows) for dev/testing | ✅ Yes |
| `samples/metadata_sample.csv` | Small metadata sample for dev/testing | ✅ Yes |
| `raw/` | Full downloaded JSONL/Parquet files | ❌ No (too large) |
| `processed/` | Cleaned/filtered intermediate files | ❌ No |

## Why the raw dataset is NOT committed

- The raw Musical Instruments reviews JSONL is **~1.56 GB**.
- The metadata Parquet files total **~230 MB**.
- These files are too large for Git and can be freely re-downloaded.
- License: the dataset is for research purposes (see the HuggingFace page).

## How to reproduce the download

Run the data loader from the project root:

```bash
python -m src.data_loader
```

This will:
1. Stream the reviews JSONL from HuggingFace (no full download needed)
2. Download the metadata Parquet files
3. Clean and merge the data
4. Save processed files locally under `data/processed/`
5. Save small sample files under `data/samples/`

### Manual download (alternative)

```python
# Reviews (JSONL, ~1.56 GB)
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Musical_Instruments.jsonl

# Metadata (Parquet, 2 files, ~230 MB total)
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw_meta_Musical_Instruments
```
