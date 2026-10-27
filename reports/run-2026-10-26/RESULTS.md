# MovieLens benchmark — October 26, 2026

## Scope and provenance

The full MovieLens 25M input was read: **25,000,095 ratings and 62,423 movies**.
The official download returned HTTP 503. A pinned GitHub mirror supplied the
files; ZIP CRC, expected record counts and SHA-256 hashes were verified and
recorded in `../data_provenance.json`. There was no independent comparison to
an official checksum. Raw data is not redistributed in this repository.

The benchmark uses the repaired frozen-history protocol, not the old experiment:

- Retrieval/feature history: **12,653,346 ratings**, 90,760 users, 9,309 items;
  history ends February 20, 2008. ALS uses 64 factors and 15 iterations.
- Ranker training: **241,581 ratings**, 4,051 users; all eligible users included.
- Ranker validation: **241,582 ratings**, 3,180 users; early stopping selected
  iteration **161**. Validation NDCG over rated-item groups is not the retrieval
  benchmark metric and must not be used as a headline recommendation result.
- Final test: **33,048 ratings**, 938 known users, from March 2017 onward.
  **896** have at least one rating >=3.5 and are all evaluated; 42 without positive
  test labels are excluded. The same users and trained model are used at each k.

Exact window boundaries are in `splits.json`.
Cold-user/item filtering greatly narrows the final test population; this is
an evaluation of long-lived, known users/items, not all MovieLens users.

## Corrected results

| Candidates | Baseline NDCG@10 | Reranked NDCG@10 | Relative lift | Baseline Recall@50 | Reranked Recall@50 | Relative lift | Total p99 ms |
|---|---|---|---|---|---|---|---|
| 100 | 0.01122 | 0.01593 | +41.9% | 0.03960 | 0.04334 | +9.4% | 2.70 |
| 500 | 0.01122 | 0.01726 | +53.8% | 0.03960 | 0.05442 | +37.4% | 4.92 |
| 2000 | 0.01122 | 0.01672 | +48.9% | 0.03960 | 0.05908 | +49.2% | 18.74 |

At k=500 the absolute NDCG gain is about 0.00603, and Recall@50 rises from
3.96% to 5.44% (about 1.48 percentage points). Relative improvements are sizable,
but absolute relevance metrics remain low. k=500 has the best NDCG among these
three diagnostic settings; k=2000 has higher recall at a higher latency cost.
These are test-set comparisons, not a validated production configuration choice.

The evaluator previously computed ideal DCG only from retrieved relevant items.
It now computes ideal DCG from the full ground-truth relevant set, correctly
penalizing retrieval misses. A regression test ensures adding relevant items
below the cutoff cannot change NDCG@k. Earlier NDCG values are superseded.
Total p99 is measured from whole-request compute timings, not summed stage p99s.

## Limits

- Frozen history is old relative to the test window; there is no rolling retrain.
- Initial minimum-count eligibility filtering examines the complete dataset.
- Ranker learning uses rated-item groups without hard-negative mining. This is
  not yet the same candidate distribution encountered at serving time.
- MovieLens ratings are not advertising clicks or conversions. No online lift,
  ad-revenue impact, significance claim or production throughput is established.
- Timing covers warmed offline retrieval, feature creation and model scoring
  in this environment (4 OMP threads, 1 BLAS thread). It excludes HTTP/network,
  startup, concurrent load and external deployment. The API smoke test is local.

## Reproduce

From the repository root, install `requirements-dev.txt` in an isolated Python
3.12 environment and place MovieLens files in `data/raw/ml-25m/`:

```bash
export MLFLOW_DISABLE_TELEMETRY=true
export MLFLOW_DISABLE_AGENT_HINT=true
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python -m src.data.load_movielens
python -m src.retrieval.train_als
python -m src.features.prepare_ranker --user-fraction 1
python -m src.ranking.train_ranker
python -m src.eval.benchmark --output reports/current_benchmark.json
python -m pytest tests -q
```

The benchmark JSON contains evaluation dependency versions and raw metrics;
`model_manifest.json` records artifact hashes. ALS used four BLAS threads in
this run; the reproduction command uses one to avoid oversubscription. Exact
floating-point outputs and timings can vary by environment. No model was tuned
after observing these test results.
