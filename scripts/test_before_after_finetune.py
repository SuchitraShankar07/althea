#!/usr/bin/env python
"""
Simple before/after fine-tuning inspection script.

Shows exactly 4 things:
1) Input query
2) First output of the base model
3) Hallucinated part from the first output
4) Output after fine-tuning (adapter loaded)

Usage:
  .venv/bin/python scripts/test_before_after_finetune.py \
      --config config/config.cpu.yaml \
      --query "Who built the Eiffel Tower?" \
      --adapter outputs/qlora/metric_loss
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnosis.claim_extractor import get_claim_extractor
from src.diagnosis.diagnose import HallucinationDiagnoser
from src.diagnosis.metric_engine import MetricEngine
from src.diagnosis.verification_engine import NLIVerificationEngine, NLILabel
from src.generation.generator import RAGGenerator
from src.retrieval.retriever import Retriever


def build_diagnoser(cfg: dict, retriever: Retriever) -> HallucinationDiagnoser:
    diag_cfg = cfg.get("diagnosis", {})
    extractor = get_claim_extractor(
        use_llm=False,
        min_length=diag_cfg.get("claim_min_length", 6),
    )
    verifier = NLIVerificationEngine(
        model_name=diag_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"),
        batch_size=diag_cfg.get("nli_batch_size", 8),
        entailment_threshold=diag_cfg.get("entailment_threshold", 0.7),
        contradiction_threshold=diag_cfg.get("contradiction_threshold", 0.6),
        device=-1,  # force CPU for stability in this inspection script
    )
    metric_engine = MetricEngine.from_config(cfg)
    return HallucinationDiagnoser(
        retriever=retriever,
        claim_extractor=extractor,
        verifier=verifier,
        metric_engine=metric_engine,
        evidence_top_k=cfg.get("retrieval", {}).get("top_k", 5),
        max_claims=diag_cfg.get("claim_max_per_answer", 20),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.cpu.yaml")
    parser.add_argument("--query", default="Who built the Eiffel Tower?")
    parser.add_argument("--adapter", default="outputs/qlora/metric_loss")
    parser.add_argument("--report", default="outputs/before_after_report.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retriever = Retriever.from_config(cfg["retrieval"])
    docs = retriever.retrieve(args.query, top_k=cfg["retrieval"].get("top_k", 5))

    base_gen = RAGGenerator.from_config(cfg["generation"])
    base_answer = base_gen.generate(args.query, docs)

    diagnoser = build_diagnoser(cfg, retriever)
    base_diag = diagnoser.diagnose(base_answer, original_docs=docs)

    hallucinated_parts = [
        {
            "claim": vr.claim,
            "label": vr.label.value,
            "entailment": round(vr.entailment_score, 3),
            "contradiction": round(vr.contradiction_score, 3),
            "neutral": round(vr.neutral_score, 3),
        }
        for vr in base_diag.verification_results
        if vr.label != NLILabel.ENTAILMENT
    ]

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. Run training first."
        )

    tuned_cfg = dict(cfg["generation"])
    tuned_cfg["adapter_path"] = str(adapter_path)
    tuned_gen = RAGGenerator.from_config(tuned_cfg)
    tuned_answer = tuned_gen.generate(args.query, docs)
    tuned_diag = diagnoser.diagnose(tuned_answer, original_docs=docs)

    tuned_hallucinated_parts = [
        {
            "claim": vr.claim,
            "label": vr.label.value,
            "entailment": round(vr.entailment_score, 3),
            "contradiction": round(vr.contradiction_score, 3),
            "neutral": round(vr.neutral_score, 3),
        }
        for vr in tuned_diag.verification_results
        if vr.label != NLILabel.ENTAILMENT
    ]

    base_hallucinated_claims = {p["claim"] for p in hallucinated_parts}
    tuned_hallucinated_claims = {p["claim"] for p in tuned_hallucinated_parts}
    removed_hallucinations = sorted(base_hallucinated_claims - tuned_hallucinated_claims)

    print("\n" + "=" * 72)
    print("1) INPUT")
    print("-" * 72)
    print(args.query)

    print("\n" + "=" * 72)
    print("2) FIRST OUTPUT (BASE MODEL)")
    print("-" * 72)
    print(base_answer if base_answer else "<empty>")

    print("\n" + "=" * 72)
    print("3) HALLUCINATED PART (FROM FIRST OUTPUT)")
    print("-" * 72)
    if hallucinated_parts:
        for i, part in enumerate(hallucinated_parts, 1):
            print(f"{i}. [{part['label']}] {part['claim']}")
    else:
        print("No hallucinated claim detected by NLI for this sample.")

    print("\n" + "=" * 72)
    print("4) OUTPUT AFTER FINE-TUNING")
    print("-" * 72)
    print(tuned_answer if tuned_answer else "<empty>")

    print("\n" + "=" * 72)
    print("5) EXPLICITLY REMOVED HALLUCINATION")
    print("-" * 72)
    if removed_hallucinations:
        for i, claim in enumerate(removed_hallucinations, 1):
            print(f"{i}. {claim}")
    else:
        print("No hallucination removal detected for this sample.")
        if hallucinated_parts:
            print("(There were base hallucinations, but they still appear after fine-tuning.)")
        else:
            print("(No base hallucinations were detected to remove.)")

    report = {
        "input": args.query,
        "base_output": base_answer,
        "hallucinated_parts": hallucinated_parts,
        "tuned_output": tuned_answer,
        "tuned_hallucinated_parts": tuned_hallucinated_parts,
        "removed_hallucinations": removed_hallucinations,
        "base_metrics": base_diag.metrics.to_dict(),
        "tuned_metrics": tuned_diag.metrics.to_dict(),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
