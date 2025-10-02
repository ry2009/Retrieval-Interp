"""Reporting utilities for experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

from rich.console import Console
from rich.table import Table


class ReportBuilder:
    """Render textual summaries from experiment payloads."""

    def __init__(self, payload_path: Path) -> None:
        self.payload_path = payload_path
        with payload_path.open("r", encoding="utf-8") as handle:
            self.payload = json.load(handle)

    def summary_table(self) -> Table:
        table = Table(title=f"Experiment: {self.payload['config']['experiment_name']}")
        table.add_column("Metric")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        for metric_name, stats in self.payload["metrics"].items():
            table.add_row(metric_name, f"{stats['mean']:.3f}", f"{stats['std']:.3f}")
        return table

    def case_studies(self, num_examples: int = 3) -> List[Dict]:
        results = self.payload["results"]
        # Sort by F1 descending for successes
        successes = [r for r in results if r["f1"] >= 0.5]
        failures = [r for r in results if r["f1"] < 0.5]
        successes.sort(key=lambda r: r["f1"], reverse=True)
        failures.sort(key=lambda r: r["f1"])
        case_entries = []
        if successes:
            case_entries.append({"kind": "success", "example": successes[0]})
        if failures:
            case_entries.append({"kind": "failure", "example": failures[0]})
        return case_entries[:num_examples]

    def export_markdown(self, output_path: Path) -> None:
        cases = self.case_studies()
        lines = [f"# Experiment Report: {self.payload['config']['experiment_name']}"]
        metrics = self.payload["metrics"]
        lines.append("\n## Aggregate Metrics")
        for name, stats in metrics.items():
            lines.append(f"- **{name.upper()}**: {stats['mean']:.3f} ± {stats['std']:.3f}")

        failures = [res for res in self.payload["results"] if res.get("f1", 0.0) < 0.5]
        if failures:
            lines.append("\n## Failure Taxonomy")
            fail_counter = Counter()
            for res in failures:
                for tag in res.get("failure_tags", []):
                    fail_counter[tag["name"]] += 1
            if fail_counter:
                total_failures = sum(fail_counter.values())
                for name, count in fail_counter.most_common():
                    lines.append(f"- **{name}**: {count} cases ({count / len(failures):.0%} of failures)")
            else:
                lines.append("- Failure reasons: taxonomy not available (no tags recorded).")

        lines.append("\n## Case Studies")
        for entry in cases:
            example = entry["example"]
            if not example:
                continue
            label = "Success" if entry["kind"] == "success" else "Failure"
            lines.append(f"### {label}: Sample {example['sample_id']}")
            lines.append(f"**Question:** {example['question']}")
            lines.append(f"**Gold Answer:** {example['answer']}")
            lines.append(f"**Model Answer:** {example['llm_answer']}")
            lines.append(
                f"**Retrieval Hit@K:** {example['hit_at_k']:.2f} | **MRR:** {example['mrr']:.2f}"
            )
            if example.get("verifier_score") is not None:
                lines.append(
                    f"**Verifier Score:** {example['verifier_score']:.2f}"
                    f" (threshold {example.get('verifier_threshold', 0.0):.2f})"
                )
            if example.get("failure_tags"):
                tag_text = ", ".join(tag["name"] for tag in example["failure_tags"])
                lines.append(f"**Failure Tags:** {tag_text}")
            lines.append("Top Documents:")
            for doc in example["top_docs"]:
                support = "✅" if doc["is_supporting"] else "⬜"
                lines.append(f"  - {support} {doc['title']} (score={doc['score']:.2f})")
                if "token_matches" in doc:
                    tm_str = ", ".join(
                        f"{pair['query_token']}→{pair['doc_token']} ({pair['similarity']:.2f})"
                        for pair in doc["token_matches"]
                    )
                    lines.append(f"    - Token matches: {tm_str}")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def print_console(self) -> None:
        console = Console()
        console.print(self.summary_table())
