"""Run an identical-user candidate sweep; save measured results separately."""
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from src.eval.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/current_benchmark.json')
    parser.add_argument('--users', type=int, default=1000000,
                        help='Cap; default includes all eligible test users.')
    args = parser.parse_args()
    result = {
        'protocol': 'frozen history; later ranker train/validation; final chronological test',
        'seed': 42,
        'ranker_user_fraction': 1.0,
        'python': platform.python_version(),
        'threads': {key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
        'versions': {key: importlib.metadata.version(key) for key in
                     ['numpy', 'pandas', 'lightgbm', 'faiss-cpu', 'implicit']},
        'runs': [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for k in [100, 500, 2000]:
        metrics = run_evaluation(n_eval_users=args.users, k_candidates=k, seed=42)
        result['runs'].append(metrics)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(metrics), flush=True)

if __name__ == '__main__':
    main()
