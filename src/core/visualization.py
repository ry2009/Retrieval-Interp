"""Visualization helpers for interpretability artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go


class HeatmapRenderer:
    """Create token similarity heatmaps from token alignment logs."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_example(self, example: dict, max_docs: int = 3) -> Path:
        question_tokens = example.get("question_tokens", [])
        docs = example.get("top_docs", [])[:max_docs]
        html_fragments: List[str] = []

        for doc in docs:
            matches: List[dict] = doc.get("token_matches", [])
            if not matches:
                continue
            heatmap = self._make_heatmap(question_tokens, matches, doc.get("title", ""))
            html_fragments.append(heatmap.to_html(full_html=False, include_plotlyjs='cdn'))

        out_path = self.output_dir / f"{example['sample_id']}_heatmaps.html"
        out_path.write_text("\n".join(html_fragments), encoding="utf-8")
        return out_path

    def _make_heatmap(self, question_tokens: List[str], matches: List[dict], title: str) -> go.Figure:
        doc_tokens = [item["doc_token"] for item in matches]
        similarities = [item["similarity"] for item in matches]
        fig = go.Figure(
            data=go.Heatmap(
                z=[similarities],
                x=doc_tokens,
                y=["query"],
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                hovertemplate="Doc token: %{x}<br>Similarity: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"{title}",
            xaxis_title="Document tokens",
            yaxis_title="Query",
            height=300,
        )
        return fig


def export_heatmaps(payload: dict, output_dir: Path, max_examples: int = 10) -> List[Path]:
    renderer = HeatmapRenderer(output_dir)
    generated: List[Path] = []
    for example in payload.get("results", [])[:max_examples]:
        generated.append(renderer.render_example(example))
    return generated
