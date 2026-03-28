#!/usr/bin/env python3
"""
scripts/demo.py
End-to-end demo using mock retriever/generator and real diagnosis/metrics.

Runs without GPU or model downloads — zero setup required.

Usage:
    python scripts/demo.py
    python scripts/demo.py --verbose
    python scripts/demo.py --queries "Who built the Eiffel Tower?" --save out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Mock corpus ───────────────────────────────────────────────────────────────
MOCK_CORPUS = {
    "Alexander Graham Bell": (
        "Alexander Graham Bell patented a practical telephone in 1876. "
        "Bell conducted his first successful telephone call on March 10, 1876."
    ),
    "DNA Structure": (
        "The DNA double-helix structure was described by Watson and Crick in 1953. "
        "DNA carries the genetic information of all known living organisms."
    ),
    "2008 Financial Crisis": (
        "The 2008 financial crisis followed the housing bubble collapse and Lehman bankruptcy. "
        "Subprime mortgage lending and securitisation were key drivers."
    ),
    "Eiffel Tower": (
        "The Eiffel Tower was built by Gustave Eiffel's company between 1887 and 1889. "
        "It served as the entrance arch for the 1889 World's Fair in Paris."
    ),
    "Climate Change": (
        "Global average temperatures have risen approximately 1.2 degrees Celsius since "
        "pre-industrial times. The primary driver is the increased concentration of "
        "greenhouse gases from human activity."
    ),
}

# ── Mock answers (intentional hallucination in the Eiffel Tower answer) ───────
MOCK_ANSWERS = {
    "who invented the telephone": (
        "The telephone was invented by Alexander Graham Bell in 1876."
    ),
    "what is dna": (
        "DNA is a molecule carrying genetic information, and its double helix was "
        "described by Watson and Crick in 1953."
    ),
    "what caused the 2008 financial crisis": (
        "The crisis was driven by subprime mortgages, securitization risk, "
        "and Lehman Brothers' collapse."
    ),
    "who built the eiffel tower": (
        # Intentional hallucination: Napoleon did not commission the Eiffel Tower.
        # The KeywordVerifier catches this as a CONTRADICTION.
        "The Eiffel Tower was built by Gustave Eiffel. "
        "Napoleon Bonaparte commissioned the tower in 1800."
    ),
    "what is climate change": (
        "Climate change refers to long-term shifts in temperatures and weather patterns, "
        "primarily driven by human emissions of greenhouse gases since the industrial era."
    ),
}


# ── Mock components ───────────────────────────────────────────────────────────
class MockRetriever:
    """Keyword-overlap retriever over MOCK_CORPUS (no FAISS needed)."""

    def retrieve(self, query: str, top_k: int = 3):
        q = query.lower()
        scored = []
        for title, text in MOCK_CORPUS.items():
            score = sum(1 for word in q.split() if word in text.lower())
            scored.append(
                {
                    "doc_id": title.lower().replace(" ", "_"),
                    "text": text,
                    "score": float(score),
                    "metadata": {"title": title},
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class MockGenerator:
    """Returns a canned answer keyed on the query (with one intentional hallucination)."""

    def generate(self, query: str, documents):
        q = query.lower().rstrip("?").strip()
        for key, ans in MOCK_ANSWERS.items():
            if key in q:
                return ans
        return documents[0]["text"] if documents else "No answer."


# ── Runtime stubs (loguru + spacy) ───────────────────────────────────────────
def _install_runtime_stubs():
    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _Logger:
            def info(self, *_, **__): return None
            def debug(self, *_, **__): return None
            def warning(self, *_, **__): return None
            def error(self, *_, **__): return None

        loguru.logger = _Logger()
        sys.modules["loguru"] = loguru

    if "spacy" not in sys.modules:
        sys.modules["spacy"] = types.ModuleType("spacy")


# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_mock_pipeline():
    """Wire real claim extraction + metric engine with mock retrieval/generation/NLI."""
    _install_runtime_stubs()
    from src.diagnosis.claim_extractor import SpacyClaimExtractor
    from src.diagnosis.diagnose import HallucinationDiagnoser
    from src.diagnosis.metric_engine import MetricEngine
    from src.diagnosis.verification_engine import (
        ClaimVerificationResult,
        NLILabel,
        _is_temporal_claim,
    )

    # spaCy regex fallback (nlp=None)
    extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
    extractor.min_length = 5
    extractor.nlp = None

    class KeywordVerifier:
        """
        Lightweight NLI substitute using word overlap.
        Detects the Napoleon hallucination as CONTRADICTION because
        'napoleon' appears in the claim but not in any corpus document.
        """

        def verify_claims(self, claims, evidence_lists):
            results = []
            for claim, evidences in zip(claims, evidence_lists):
                claim_l = claim.lower()
                ent, con, label = 0.1, 0.1, NLILabel.NEUTRAL
                for ev in evidences:
                    ev_l = ev.lower()
                    overlap = len(set(claim_l.split()) & set(ev_l.split()))
                    if "napoleon" in claim_l and "napoleon" not in ev_l:
                        # Napoleon not mentioned in corpus -> contradiction signal
                        label, ent, con = NLILabel.CONTRADICTION, 0.05, 0.85
                    elif overlap >= 3 and label != NLILabel.CONTRADICTION:
                        label, ent, con = NLILabel.ENTAILMENT, 0.82, 0.08
                is_t = _is_temporal_claim(claim)
                results.append(
                    ClaimVerificationResult(
                        claim=claim,
                        label=label,
                        entailment_score=ent,
                        contradiction_score=con,
                        neutral_score=max(0.0, 1.0 - ent - con),
                        is_temporal=is_t,
                        is_outdated=is_t and label != NLILabel.ENTAILMENT,
                    )
                )
            return results

    retriever = MockRetriever()
    generator = MockGenerator()
    diagnoser = HallucinationDiagnoser(
        retriever=retriever,
        claim_extractor=extractor,
        verifier=KeywordVerifier(),
        metric_engine=MetricEngine(),
        evidence_top_k=3,
        max_claims=15,
    )
    return retriever, generator, diagnoser


# ── Display helpers ───────────────────────────────────────────────────────────
def _bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for a metric value in [0, 1]."""
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "[" + "\u2588" * filled + "\u2591" * (width - filled) + f"] {value:.3f}"


def print_result(query: str, answer: str, diag, idx: int, verbose: bool = False):
    print("\n" + "=" * 72)
    print(f"Q{idx}. {query}")
    print("-" * 72)
    print(textwrap.fill(answer, width=72))
    print("-" * 72)
    print(f"Claims extracted: {len(diag.claims)} | {diag.metrics.summary()}")

    if verbose:
        # Per-claim label table
        label_icons = {"entailment": "\u2713", "contradiction": "\u2717", "neutral": "~"}
        print("\n  Claim verification:")
        for i, (claim, vr) in enumerate(zip(diag.claims, diag.verification_results), 1):
            icon = label_icons.get(vr.label.value, "?")
            print(f"    {i}. [{icon} {vr.label.value.upper():<13}] {claim}")

        # ASCII metric bars (with direction hints)
        m = diag.metrics
        print("\n  Metric bars:")
        print(f"    SCR  (up=better)  {_bar(m.scr)}")
        print(f"    CR   (dn=better)  {_bar(m.cr)}")
        print(f"    TVE  (dn=better)  {_bar(m.tve)}")
        print(f"    CDEE (dn=better)  {_bar(m.cdee)}")
        print(f"    CHS  (dn=better)  {_bar(m.chs)}")
    else:
        for i, (claim, vr) in enumerate(zip(diag.claims, diag.verification_results), 1):
            print(f"  {i}. [{vr.label.value}] {claim}")


def print_training_signal_summary(rows: list):
    """Print CHS distribution and available DPO pairs."""
    print("\n" + "=" * 72)
    print("TRAINING SIGNAL SUMMARY")
    print("-" * 72)
    chs_values = [r["metrics"]["chs"] for r in rows]
    low = sum(1 for c in chs_values if c <= 0.3)
    mid = sum(1 for c in chs_values if 0.3 < c <= 0.6)
    high = sum(1 for c in chs_values if c > 0.6)
    mean_chs = sum(chs_values) / max(len(chs_values), 1)

    print(f"  Total samples    : {len(rows)}")
    print(f"  Mean CHS         : {mean_chs:.3f}")
    print(f"  Low   (\u22640.3)     : {low:>3}  " + "\u2588" * low)
    print(f"  Mid   (0.3-0.6)  : {mid:>3}  " + "\u2588" * mid)
    print(f"  High  (>0.6)     : {high:>3}  " + "\u2588" * high)

    # Estimate available DPO pairs (gap >= 0.2)
    from itertools import combinations
    pairs = [
        (a, b) for a, b in combinations(chs_values, 2)
        if abs(b - a) >= 0.2
    ]
    print(f"\n  DPO pairs available (gap \u2265 0.2): {len(pairs)}")
    print()


def print_comparison_table(rows: list):
    """Print a simulated baseline vs fine-tuned comparison."""
    chs_vals = [r["metrics"]["chs"] for r in rows]
    baseline_chs = sum(chs_vals) / max(len(chs_vals), 1)
    tuned_chs = baseline_chs * 0.80   # simulate ~20% improvement
    baseline_scr = sum(r["metrics"]["scr"] for r in rows) / max(len(rows), 1)
    tuned_scr = min(1.0, baseline_scr * 1.15)

    print("=" * 72)
    print("SIMULATED BASELINE vs FINE-TUNED COMPARISON")
    print(f"  {'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("  " + "-" * 44)
    for label, b, t in [
        ("SCR  up", baseline_scr, tuned_scr),
        ("CHS  dn", baseline_chs, tuned_chs),
    ]:
        delta = t - b
        trend = "\u25b2" if delta > 0 else ("\u25bc" if delta < 0 else "\u2014")
        print(f"  {label:<12} {b:>10.3f} {t:>12.3f} {trend} {abs(delta):>5.3f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Failure-Aware RAG end-to-end demo (no GPU, no downloads)."
    )
    parser.add_argument("--queries", nargs="*", default=None, help="Custom query list")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-claim labels, ASCII metric bars, and comparison table",
    )
    parser.add_argument("--save", default=None, help="Save results to a JSONL file")
    args = parser.parse_args()

    queries = args.queries or [
        "Who invented the telephone?",
        "What is DNA?",
        "What caused the 2008 financial crisis?",
        "Who built the Eiffel Tower?",
        "What is climate change?",
    ]

    print("\n" + "\u2588" * 72)
    print("  FAILURE-AWARE RAG \u2014 END-TO-END DEMO")
    print("  (Zero setup: mock retriever + generator, real metrics engine)")
    print("\u2588" * 72)

    retriever, generator, diagnoser = build_mock_pipeline()
    rows = []

    for i, q in enumerate(queries, 1):
        docs = retriever.retrieve(q, top_k=3)
        ans = generator.generate(q, docs)
        diag = diagnoser.diagnose(ans, original_docs=docs)
        print_result(q, ans, diag, i, verbose=args.verbose)
        rows.append({"query": q, "answer": ans, "metrics": diag.metrics.to_dict()})

    print_training_signal_summary(rows)

    if args.verbose:
        print_comparison_table(rows)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Saved demo output to {args.save}")


if __name__ == "__main__":
    main()
