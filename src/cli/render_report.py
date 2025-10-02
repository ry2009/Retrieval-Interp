"""Interpretability report renderer placeholder."""
from __future__ import annotations

import argparse
import pathlib

from ..core import config as config_mod
from ..core.report import ReportBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render interpretability report")
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
    output_dir = pathlib.Path(config["evaluation"]["output_dir"])
    payload_path = output_dir / "results.json"
    builder = ReportBuilder(payload_path)
    builder.print_console()
    out_path = output_dir / "report.md"
    builder.export_markdown(out_path)
    print(f"Report written to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
