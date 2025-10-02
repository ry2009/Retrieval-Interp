"""Baseline retrieval runner.

Loads configuration, prepares retriever and LLM, then executes evaluation sweeps.
"""
from __future__ import annotations

import argparse
import pathlib

from ..core import config as config_mod
from ..core.pipeline import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline LLM-augmented retrieval")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to experiment configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_mod.load_config(args.config)
    runner = ExperimentRunner(config)
    payload = runner.run()
    print(
        f"Finished experiment {config['experiment_name']} on {payload['num_examples']} examples."
    )
    print(
        "Metrics: EM={:.3f}±{:.3f}, F1={:.3f}±{:.3f}, Hit@K={:.3f}±{:.3f}, MRR={:.3f}±{:.3f}".format(
            payload["metrics"]["em"]["mean"],
            payload["metrics"]["em"]["std"],
            payload["metrics"]["f1"]["mean"],
            payload["metrics"]["f1"]["std"],
            payload["metrics"]["hit_at_k"]["mean"],
            payload["metrics"]["hit_at_k"]["std"],
            payload["metrics"]["mrr"]["mean"],
            payload["metrics"]["mrr"]["std"],
        )
    )

    failures = [res for res in payload["results"] if res.get("f1", 0.0) < 0.5]
    if failures:
        tagged = sum(1 for res in failures if res.get("failure_tags"))
        print(f"Failures tagged: {tagged}/{len(failures)}")
        if tagged:
            tag_counts = {}
            for res in failures:
                for tag in res.get("failure_tags", []):
                    tag_counts[tag["name"]] = tag_counts.get(tag["name"], 0) + 1
            top_tags = ", ".join(f"{k}:{v}" for k, v in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:3])
            print(f"Top failure tags: {top_tags}")


if __name__ == "__main__":  # pragma: no cover
    main()
