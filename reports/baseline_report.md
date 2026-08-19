# 📈 ShopSense Baseline Evaluation Report: Popularity Recommender

> Non-personalized popularity baseline fitted strictly on `data/processed/train.parquet` (136,814 interactions).

---

## 1. Baseline Methodology
The Popularity Baseline ranks products across the training catalog using a **Bayesian Weighted Rating & Volume Scoring** formulation:

$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$

$$\text{Score} = v \times \text{WR}$$

Where:
- $v$ = interaction count of the item in the training set
- $m = 5$ = smoothing threshold parameter
- $R$ = sample mean rating of the item
- $C = 4.48$ = global average rating across the entire training set

For any evaluation user $u$, `recommend(user_id=u, k=10)` generates the top-$k$ products from the sorted popularity list, **strictly filtering out any products already rated by $u$ in the training set**.

---

## 2. Top 10 Most Popular Products (Training Set)

| Rank | ASIN | Product Title | Brand / Store | Price | Interactions ($v$) | Avg Rating ($R$) | Score |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `B09857JRP2` | GLS Audio Instrument Cable - Amp Cord for Bass & Electric Guitar - Straight to Right Angle 1/4 Inch Instrument Cable - Black/Grey Braided Tweed, 10ft | GLS Audio | $14.30 | 914 | 4.78 ★ | 4364.5 |
| 2 | `B0BCK6L7S5` | Donner Guitar Delay Pedal, Yellow Fall Analog Delay Guitar Effect Pedal Vintage Delay True Bypass | Donner | $39.99 | 971 | 4.43 ★ | 4304.3 |
| 3 | `B0BPJ4Q6FJ` | D'Addario Guitar Strings - XL Nickel Electric Guitar Strings - EXL150H - Perfect Intonation, Consistent Feel, Reliable Durability - For 6 String Guitars - 10-26 High Strung/Nashville Tuning | D'Addario | $5.99 | 649 | 4.69 ★ | 3041.0 |
| 4 | `B0BSGM6CQ9` | Ernie Ball Mondo Slinky Nickelwound Electric Guitar Strings 10.5-52 Gauge | Ernie Ball | $6.99 | 642 | 4.72 ★ | 3030.8 |
| 5 | `B0BTC9YJ2W` | D'Addario Guitar Strings - Phosphor Bronze Acoustic Guitar Strings - EJ41 - Rich, Full Tonal Spectrum - For 12 String Guitars - 09-45 Extra Light 12-String | D'Addario | $10.99 | 484 | 4.69 ★ | 2268.0 |
| 6 | `B08R5GM6YB` | D'Addario Pro-Winder - Guitar String Winder, Guitar String Cutter, Guitar Bridge Pin Puller - Ultimate Ergonomic All in One Guitar Tool for Restringing - White | D'Addario Accessories | $10.99 | 441 | 4.41 ★ | 1947.3 |
| 7 | `B08SJY4T7K` | Behringer TM300 Tube Amp Modeler Pedal | Behringer | N/A | 429 | 4.25 ★ | 1825.1 |
| 8 | `B004XNK7AI` | Snark SN-5 Tuner for Guitar, Bass and Violin (Black) | SNARK | N/A | 366 | 4.71 ★ | 1723.9 |
| 9 | `B0BPLFP5P6` | D'Addario Accessories Guitar Tuner - Micro Headstock Tuner - Tuner for Acoustic Guitar, Electric Guitar, Bass Guitar, Mandolin, Banjo, Ukelele, Cello, Double Bass & More - Universal Clip On - 1pk | D'Addario Accessories | $18.99 | 390 | 4.20 ★ | 1639.4 |
| 10 | `B0B95V41NR` | D'Addario Guitar Cable - Guitar Patch Cable - Instrument Cable - Guitar Accessories - 1/4 Inch Right Angle Cable - 1 Foot/0.30 Meters - Right Angle - 1 Pack | D'Addario Accessories | $4.99 | 328 | 4.63 ★ | 1519.3 |

---

## 3. Evaluation Results on Chronological Test Set (K = 10)

| Metric | Overall Test Set (All Items) | Warm-Item Test Set (Known Catalog Items) |
|:---|:---:|:---:|
| **Precision@10** | **0.010067** (1.007%) | **0.010067** (1.007%) |
| **Recall@10** | **0.029725** (2.972%) | **0.029773** (2.977%) |
| **NDCG@10** | **0.023195** | **0.023225** |

---

## 4. Catalog Coverage & Recommendation Diversity

| Metric | Value |
|:---|:---:|
| **Catalog Coverage@10** | **0.17%** |
| **Unique Products Recommended** | **19 / 11,005** |
| **Average Recommendation List Length** | **10.0** |
| **Total Evaluation Users** | **10,678** |

---

## 5. Key Analytical Takeaways
1. **Low Catalog Coverage is Expected:** As a global popularity baseline without personalization, the model recommends the same top items (shifted only by user seen-item exclusions). Only **19** distinct items appear in the top-10 across all 10,678 users, representing **0.17%** catalog coverage.
2. **Benchmark for Collaborative Filtering:** The baseline establishes our non-personalized benchmark:
   - **Recall@10 = 2.972%**
   - **NDCG@10 = 0.0232**
   - Personalized collaborative filtering models (User-Based, Item-Based, and Hybrid) in subsequent steps must demonstrate measurable improvements over this benchmark.
3. **Cold-Start / Unseen Item Impact:** Unseen test items represent 0.57% of test interactions, leading to a slight difference between overall and warm-item Recall/NDCG.
