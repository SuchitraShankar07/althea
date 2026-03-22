"""
diagnosis/diagnose.py
Orchestrator: given an answer and a retriever, runs the full
hallucination diagnosis pipeline and returns HallucinationMetrics.

Pipeline:
    Answer → ClaimExtractor → [claims]
           → EvidenceRetriever (per claim) → [evidence lists]
           → NLIVerificationEngine → [ClaimVerificationResult]
           → MetricEngine → HallucinationMetrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from .claim_extractor import SpacyClaimExtractor, get_claim_extractor
from .metric_engine import HallucinationMetrics, MetricEngine
from .verification_engine import ClaimVerificationResult, NLIVerificationEngine


@dataclass
class DiagnosisOutput:
    answer: str
    claims: List[str]
    verification_results: List[ClaimVerificationResult]
    metrics: HallucinationMetrics
    per_claim_evidence: List[List[str]] = field(default_factory=list)


class HallucinationDiagnoser:
    """
    Ties together:
        - claim extractor
        - evidence retriever (re-uses the main retriever)
        - NLI verification engine
        - metric engine
    """

    def __init__(
        self,
        retriever,           # src.retrieval.retriever.Retriever
        claim_extractor,     # SpacyClaimExtractor or LLMClaimExtractor
        verifier: NLIVerificationEngine,
        metric_engine: MetricEngine,
        evidence_top_k: int = 3,
        max_claims: int = 20,
    ):
        self.retriever = retriever
        self.claim_extractor = claim_extractor
        self.verifier = verifier
        self.metric_engine = metric_engine
        self.evidence_top_k = evidence_top_k
        self.max_claims = max_claims

    # ── Main entry point ──────────────────────────────────────
    def diagnose(self, answer: str, original_docs: Optional[List[dict]] = None) -> DiagnosisOutput:
        """
        Diagnose hallucinations in `answer`.

        original_docs: documents used during generation (if available,
                       they are appended to evidence for each claim).
        """
        # 1. Extract claims
        claims = self.claim_extractor.extract(answer, max_claims=self.max_claims)
        logger.debug(f"Extracted {len(claims)} claims")

        if not claims:
            empty_m = HallucinationMetrics()
            return DiagnosisOutput(answer=answer, claims=[], verification_results=[], metrics=empty_m)

        # 2. Per-claim evidence retrieval
        evidence_lists = []
        docs_per_claim = []
        for claim in claims:
            retrieved = self.retriever.retrieve(claim, top_k=self.evidence_top_k)
            ev_texts = [d["text"] for d in retrieved]
            # Optionally prepend original docs
            if original_docs:
                ev_texts = [d["text"] for d in original_docs] + ev_texts
            evidence_lists.append(ev_texts)
            docs_per_claim.append(len(retrieved))

        # 3. NLI verification
        verification_results = self.verifier.verify_claims(claims, evidence_lists)

        # 4. Compute metrics
        metrics = self.metric_engine.compute(verification_results, docs_per_claim)
        logger.debug(f"Hallucination metrics: {metrics.summary()}")

        return DiagnosisOutput(
            answer=answer,
            claims=claims,
            verification_results=verification_results,
            metrics=metrics,
            per_claim_evidence=evidence_lists,
        )

    # ── Batch diagnosis ───────────────────────────────────────
    def diagnose_batch(
        self,
        answers: List[str],
        original_docs_list: Optional[List[List[dict]]] = None,
    ) -> List[DiagnosisOutput]:
        if original_docs_list is None:
            original_docs_list = [None] * len(answers)
        return [
            self.diagnose(ans, orig)
            for ans, orig in zip(answers, original_docs_list)
        ]

    # ── Factory ───────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: dict, retriever) -> "HallucinationDiagnoser":
        diag_cfg = cfg.get("diagnosis", cfg)
        extractor = get_claim_extractor(
            use_llm=False,
            min_length=diag_cfg.get("claim_min_length", 6),
        )
        verifier = NLIVerificationEngine.from_config(diag_cfg)
        metric_engine = MetricEngine.from_config(cfg)
        return cls(
            retriever=retriever,
            claim_extractor=extractor,
            verifier=verifier,
            metric_engine=metric_engine,
            evidence_top_k=cfg.get("retrieval", {}).get("top_k", 3),
            max_claims=diag_cfg.get("claim_max_per_answer", 20),
        )