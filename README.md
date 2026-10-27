# RankForge — Two-stage recommendation ranking

Select relevant content from a large catalog while measuring the cost of
retrieval and reranking. The pipeline combines ALS user/item embeddings,
FAISS retrieval, LightGBM LambdaRank and a FastAPI recommendation endpoint.

## Measured on MovieLens 25M

The repaired pipeline has now been trained and evaluated on real data.
It reads the full **25,000,095-rating** input; the chronological protocol uses
**12.65M historical ratings** for retrieval and **241,581 later ratings** for
ranker training, with a separate validation window. Evaluation includes all
**896 eligible test users** with positive labels. It does not train the ranker
on all 25M ratings or evaluate all MovieLens users.

At **500 candidates**, reranking improves:

- **NDCG@10: 0.01122 -> 0.01726 (+53.8% relative)**.
- **Recall@50: 0.03960 -> 0.05442 (+37.4% relative)**.
- Measured offline end-to-end compute **p99: 4.92 ms**.

[Current results, protocol, provenance and limitations](reports/run-2026-10-26/RESULTS.md)
include all three candidate settings. These are offline research results,
not online advertising lift or production latency guarantees. Absolute scores
remain low, and the long frozen-history window excludes most cold users/items.

## Pipeline

1. Chronological data preparation creates history, ranker training labels,
   ranker validation labels, and a final test period.
2. ALS learns embeddings only from history; FAISS indexes the item embeddings.
3. Ranker feature generation uses frozen historical lookups and cosine scores.
   The feature builder is shared with evaluation and the API.
4. LambdaRank trains on the later labels and early-stops on validation.
5. Evaluation compares retrieval alone with reranking on identical test users,
   measuring NDCG@10, Recall@50, MRR and per-stage/total latency.
6. FastAPI loads artifacts once at startup and validates recommendation requests.

## Run

Use Python 3.11 or 3.12, from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export MLFLOW_DISABLE_TELEMETRY=true
export MLFLOW_DISABLE_AGENT_HINT=true
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python -m pytest tests -q

# Obtain MovieLens 25M from GroupLens; extract into data/raw/ml-25m/.
python -m src.data.load_movielens
python -m src.retrieval.train_als
python -m src.features.prepare_ranker --user-fraction 1
python -m src.ranking.train_ranker
python -m src.eval.benchmark --output reports/current_benchmark.json
python -m uvicorn src.serving.app:app --reload
```

`--user-fraction 1` includes all eligible ranker users; the CLI's 0.35 default
is available for memory-constrained experiments. Source data and trained model
artifacts are not bundled. The complete benchmark must precede API startup.

## Verification and interpretation

Nine tests cover package imports, feature/label isolation, temporal splitting,
ranking metrics and small-catalog retrieval. The current real-data run also
includes a local API smoke check; see the run directory for evidence.

Old results in `reports/RESULTS.md` and the older JSON reports in `reports/`
are retained for traceability. Their NDCG normalization was incorrect and their
training protocol differs; **use the current results linked above**. The legacy
`assemble_training_features` helper is retained but unused by the current CLI.

The initial full-dataset count filter, frozen historical catalog, lack of
hard-negative training and absence of production load tests remain limitations.
MovieLens is a research dataset; acknowledge GroupLens and follow its license.
