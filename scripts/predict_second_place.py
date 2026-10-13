#!/usr/bin/env python3
"""Run the released second-place ensemble on local or Kaggle test data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rogii.second_place import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=Path("models/second_place"))
    parser.add_argument("--output", type=Path, default=Path("outputs/second_place/submission.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--checkpoint-limit", type=int, help="Diagnostic only: use fewer than 15 checkpoints per family")
    parser.add_argument("--phase-limit", type=int, help="Diagnostic only: use fewer than 8 MD phase shifts")
    args = parser.parse_args()
    report = predict(args.data_dir, args.weights_dir, args.output, device=args.device,
                     batch_size=args.batch_size, threads=args.threads,
                     checkpoint_limit=args.checkpoint_limit, phase_limit=args.phase_limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
