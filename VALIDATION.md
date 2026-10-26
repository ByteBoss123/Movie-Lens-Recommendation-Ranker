# Repair validation — 2026-10-26

- Restored the `src/`, `tests/`, and `reports/` layout expected by imports and documentation.
- Added the missing `src.features.prepare_ranker` CLI. Frozen history supplies
  aggregate features and ALS embeddings; later disjoint time windows supply
  training and validation labels. Shared serving features avoid duplicate logic.
- Corrected FAISS score/ID alignment when requested candidates exceed catalog size.
- Added developer dependencies and pytest configuration.
- Preserved historical result files; no claim is made that their metrics apply
  to the new temporal training protocol.

Verification uses small synthetic fixtures, not the 25M-row benchmark. It checks
the pipeline mechanics and import paths, not recommendation quality or production
load capacity. The original data/model artifacts must be obtained or generated to
run the full benchmark. See README for commands and remaining limitations.
