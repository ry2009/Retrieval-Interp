"""Dataset preparation placeholder script."""
from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for retrieval experiments")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[TODO] Prepare dataset {args.dataset} split={args.split} sample={args.sample}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
