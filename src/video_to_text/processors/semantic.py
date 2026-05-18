from __future__ import annotations

import re

from ..types import ChunkResult


TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "definition": ("is defined as", "means", "is a", "refers to"),
    "formula": ("equation", "formula", "=", "z1", "a1", "softmax", "log"),
    "example": ("for example", "concrete example", "suppose", "let's say"),
    "intuition": ("intuition", "interpret", "think of", "in other words"),
    "quiz": ("quiz", "question", "what do you think"),
    "transition": ("let's now", "next", "recall", "moving on", "generalize"),
}

TRANSITIONS: tuple[tuple[str, str, str], ...] = (
    ("logistic regression", "softmax regression", "generalization"),
    ("softmax regression", "cost function", "objective_shift"),
    ("cost function", "cross entropy", "loss_refinement"),
    ("softmax regression", "neural network", "model_extension"),
)


def _combined_text(chunk_result: ChunkResult) -> str:
    transcript = chunk_result.trusted_transcript or chunk_result.cleaned_transcript or " ".join(
        s.text for s in chunk_result.transcript_segments if not s.low_trust
    )
    visuals = " ".join(n.text for n in chunk_result.visual_notes)
    return f"{transcript} {visuals}".strip().lower()


def annotate_semantics(chunk_results: list[ChunkResult]) -> None:
    for chunk_result in chunk_results:
        text = _combined_text(chunk_result)
        tags: list[str] = []
        transitions: list[str] = []

        for tag, patterns in TAG_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                tags.append(tag)

        for src, dst, kind in TRANSITIONS:
            if src in text and dst in text:
                transitions.append(f"{src}->{dst}:{kind}")

        if re.search(r"\b(e\^|sigma|sum|probability|p\(y\))\b", text):
            if "formula" not in tags:
                tags.append("formula")

        chunk_result.semantic_tags = sorted(set(tags))
        chunk_result.concept_transitions = sorted(set(transitions))
