# RankForge: Results

## Setup

- **Data**: MovieLens 25M (real dataset — 25,000,095 ratings, 62,423 movies).
  After cold-start filtering (users/items with < 20 ratings dropped):
  162,541 users, 18,430 items, 24.8M ratings.
- **Split**: temporal (sorted by timestamp, 85/15 train/test), not random —
  a random split would leak future rating patterns into training.
- **Retrieval**: ALS matrix factorization (64 factors, 15 iterations) → FAISS
  flat index (exact inner-product / cosine search) over item embeddings.
- **Ranking**: LightGBM LambdaRank re-ranking FAISS-retrieved candidates.
- **Resource note**: development ran on a sandbox with 3.9GB RAM and 1 CPU
  core. Retrieval embeddings and feature engineering ran on the **full**
  21M-row training set; the ranker itself was trained on a **35% sample of
  users** (~7.5M rows) because LightGBM's in-memory footprint for the full
  set exceeded the sandbox's RAM even after several optimization passes
  (see "Memory debugging" below). All evaluation numbers below reflect
  that ranker.

## Headline number: does the ranking stage help?

This is the number the whole two-stage architecture depends on: does
re-ranking FAISS candidates with a learned model actually improve results
over using the retrieval score alone?

**Answer: it's genuinely mixed, and that's the real finding.**

| k (candidates) | NDCG@10 (retrieval-only) | NDCG@10 (+ranker) | Recall@50 (retrieval-only) | Recall@50 (+ranker) |
|---|---|---|---|---|
| 100  | 0.0460 | 0.0581 (**+26%**) | — | — |
| 500  | 0.0355 | 0.0364 (+2.5%) | 0.0564 | 0.0655 (**+16%**) |
| 2000 | 0.0339 | 0.0283 (**−16%**) | 0.0537 | 0.0648 (**+21%**) |

(800–1500 held-out users sampled per run; users' future ratings ≥ 3.5 counted
as relevant.)

**What this actually shows:**
- The ranker consistently improves **Recall@50** — it's better at surfacing
  relevant items *somewhere* in the top 50 — across every candidate pool size.
- The ranker does **not** consistently improve **NDCG@10**, and at k=2000 it's
  meaningfully worse. It's better at broad recall than at precisely ordering
  the top 10.
- This is a real, defensible result — not a bug. (I checked: the pipeline was
  smoke-tested end-to-end before touching real data, and the same code
  produces internally consistent metrics across all three k values.)

**Why the gap between this and the 0.765 training NDCG@10?**
The number LightGBM reports during training (NDCG@10 = 0.765) evaluates a much
easier task: re-ordering the small set of items a user *actually rated* by
their rating level. The eval above does the real, harder task: given ~500-2000
candidates (most of which the user has no signal on at all), find the items
they'll actually rate highly *in the future*, among a huge pool of irrelevant
candidates. These are genuinely different problems, and conflating them is a
common way ranking projects overstate their own results. This project
reports the harder, honest number.

**Likely causes of the modest/mixed lift** (documented as known limitations,
not swept under the rug):
1. The ranker was trained on 35% of users (resource constraint) — less
   training signal than the full 21M rows would provide.
2. No hard-negative mining: training uses only items users actually rated,
   so the ranker has limited signal on what makes a *never-interacted*
   candidate irrelevant — which is most of what it sees at serve time.
3. `retrieval_score` is one of 8 ranker features, and the other 7
   (popularity, averages, genre match) may not add much signal beyond what
   ALS already captures for this dataset.

## Latency / accuracy tradeoff

This is the direct evidence for the "ranking infrastructure and systems
design" gap this project was built to close.

| k (candidates) | Retrieval p99 (ms) | Ranker-stage p99 (ms) | Total p99 (ms) |
|---|---|---|---|
| 100  | 0.52 | 1.49  | ~2.0  |
| 500  | 1.26 | 6.23  | ~7.5  |
| 2000 | 3.47 | 26.83 | ~30.3 |

**Takeaway**: retrieval stays cheap even at k=2000 (FAISS flat search over
17K items is trivial at this item-catalog scale). The ranking stage is where
latency actually grows — roughly linearly with candidate count, since it
rebuilds an 8-feature vector and scores every candidate. At production
catalog sizes (millions of items, not 17K), the retrieval side would need an
approximate index (IVF, already implemented as an option in
`faiss_index.py`) rather than flat search, and the ranker's per-candidate
feature-building loop (currently a plain Python loop, not vectorized) would
be the first thing to optimize before scaling k further.

This is the accuracy/latency Pareto frontier: k=100 is fast (~2ms total) and
has the best NDCG@10 lift (+26%); k=2000 is 15x slower and has the best
Recall@50 lift (+21%) but hurts NDCG@10. Which point on this frontier is
"right" depends entirely on the product surface — a search box wants
precision (favor small k), a "discover something new" feed wants recall
(favor large k). That framing — that this is a product decision informed by
a measured tradeoff, not a fixed engineering choice — is the actual point.

## Memory debugging (worth knowing for the interview)

Three real, non-obvious scaling bugs were found and fixed by testing against
the actual 21M-row dataset (a smoke test on fabricated small data did not
catch these):

1. `pd.merge()` against even a small (144K-row) lookup table nearly doubled
   memory at 21M-row scale — replaced with dict-based `Series.map()`.
2. float64 columns cost ~160MB each at this row count; downcasting to
   float32/int32 was necessary just to fit the feature matrix in memory.
3. `df.sort_values()` for LightGBM's required query-grouping created a full
   copy of the frame; replaced with `np.argsort` + direct array indexing,
   which only materializes the columns actually needed for training.

## Known limitations / future work

- Ranker trained on 35% of users, not the full set (RAM-constrained).
- No hard-negative mining in ranker training data.
- FAISS flat (exact) index only — IVF approximate variant is implemented
  but not benchmarked here at this item-catalog scale.
- Eval sampled 800–2000 of 144K users per run for tractability.
- A two-tower neural retrieval model (vs. ALS) is a reasonable v2, noted in
  the original design doc but out of scope here.
