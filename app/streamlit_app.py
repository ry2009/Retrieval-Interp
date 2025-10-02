"""Streamlit dashboard for inspecting LLM-augmented retrieval runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import streamlit as st


def load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_examples(payload: Dict) -> List[str]:
    examples = payload.get("results", [])
    return [f"{idx}: {ex['question'][:80]}" for idx, ex in enumerate(examples)]


def render_example(example: Dict) -> None:
    st.subheader("Question")
    st.write(example["question"])
    st.subheader("Answers")
    st.markdown(f"**Gold:** {example['answer']}")
    st.markdown(f"**Final:** {example['llm_answer']}")
    if initial := example.get("initial_answer"):
        if initial != example["llm_answer"]:
            st.markdown(f"**Initial:** {initial}")
    st.subheader("Verifier")
    st.write(
        {
            "initial": example.get("initial_verifier_score"),
            "final": example.get("verifier_score"),
            "threshold": example.get("verifier_threshold"),
            "supported_docs": example.get("verifier_supported_docs"),
        }
    )
    if example.get("failure_tags"):
        st.subheader("Failure Tags")
        for tag in example["failure_tags"]:
            st.write(f"- {tag['name']}: {tag['reason']}")
    if example.get("formatting"):
        st.subheader("Formatting Meta")
        st.write(example["formatting"])
    if example.get("refinement"):
        st.subheader("Refinement Meta")
        st.write(example["refinement"])

    st.subheader("Top Documents")
    for doc in example.get("top_docs", [])[:3]:
        st.markdown(
            f"**{'✅' if doc['is_supporting'] else '⬜'} {doc['title']}** (score={doc['score']:.2f})"
        )
        st.write(doc.get("text", "")[:500] + ("..." if len(doc.get("text", "")) > 500 else ""))
        if doc.get("token_matches"):
            st.write("Token matches:")
            st.write(doc["token_matches"])


st.set_page_config(page_title="LLM-Augmented Retrieval Inspector", layout="wide")
st.title("LLM-Augmented Retrieval Inspector")

results_dir = Path("results")
options = [p for p in results_dir.glob("*_late_interaction/results.json")]
if not options:
    st.warning("No results found. Run experiments first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select results file",
    options,
    format_func=lambda p: p.parent.name,
)

payload = load_payload(selected_file)
metrics = payload.get("metrics", {})
st.sidebar.markdown("### Metrics")
for name, stats in metrics.items():
    st.sidebar.write(f"{name.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")

examples = payload.get("results", [])
if not examples:
    st.write("No examples in payload.")
    st.stop()

example_labels = list_examples(payload)
selected_idx = st.selectbox("Choose example", range(len(example_labels)), format_func=lambda i: example_labels[i])
render_example(examples[selected_idx])
