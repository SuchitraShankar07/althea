"""
diagnosis/metric_engine.py
Computes the four hallucination metrics from claim verification results.

Metrics:
    SCR   — Support Coverage Ratio
    CR    — Conflict Rate
    TVE   — Temporal Validity Error
    CDEE  — Cross-Document Entailment Error
    CHS   — Composite Hallucination Score
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

from .verification_engine import ClaimVerificationResult, NLILabel


# ── Metric result dataclass ───────────────────────────────────────────────────
@dataclass
class HallucinationMetrics:
    scr: float   = 0.0   # Support Coverage Ratio         ↑ better
    cr: float    = 0.0   # Conflict Rate                  ↓ better
    tve: float   = 0.0   # Temporal Validity Error        ↓ better
    cdee: float  = 0.0   # Cross-Document Entailment Error ↓ better
    chs: float   = 0.0   # Composite Hallucination Score  ↓ better

    # Counts (for transparency)
    total_claims: int       = 0
    supported_claims: int   = 0
    contradicted_claims: int = 0
    temporal_claims: int    = 0
    outdated_claims: int    = 0
    synthesis_errors: int   = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"SCR={self.scr:.3f} | CR={self.cr:.3f} | "
            f"TVE={self.tve:.3f} | CDEE={self.cdee:.3f} | "
            f"CHS={self.chs:.3f}  "
            f"({self.supported_claims}/{self.total_claims} supported)"
        )


# ── Metric engine ─────────────────────────────────────────────────────────────
class MetricEngine:
    """
    Aggregates ClaimVerificationResults into HallucinationMetrics.

    Weights are configurable; defaults match the paper proposal.
    """

    def __init__(
        self,
        scr_weight: float = 1.0,
        cr_weight: float  = 1.0,
        tve_weight: float = 0.8,
        cdee_weight: float = 0.8,
        lambda_: float = 0.5,
    ):
        self.scr_weight   = scr_weight
        self.cr_weight    = cr_weight
        self.tve_weight   = tve_weight
        self.cdee_weight  = cdee_weight
        self.lambda_      = lambda_

    def compute(
        self,
        results: List[ClaimVerificationResult],
        docs_per_claim: List[int] | None = None,
    ) -> HallucinationMetrics:
        """
        results         : per-claim NLI verification results
        docs_per_claim  : number of distinct source documents per claim
                          (used to detect cross-document synthesis errors)
        """
        if not results:
            return HallucinationMetrics()

        n = len(results)
        supported   = sum(1 for r in results if r.label == NLILabel.ENTAILMENT)
        contradicted = sum(1 for r in results if r.label == NLILabel.CONTRADICTION)
        temporal    = sum(1 for r in results if r.is_temporal)
        outdated    = sum(1 for r in results if r.is_outdated)

        # CDEE: claims that required multiple source docs but are NOT entailed
        if docs_per_claim is not None:
            multi_doc_claims = [
                r for r, nd in zip(results, docs_per_claim) if nd > 1
            ]
        else:
            # Approximate: neutral claims with long claim text may be synthesis
            multi_doc_claims = [
                r for r in results if len(r.claim.split()) > 12
            ]
        synthesis_errors = sum(
            1 for r in multi_doc_claims if r.label != NLILabel.ENTAILMENT
        )
        n_multi = max(len(multi_doc_claims), 1)

        # ── Core metrics ──────────────────────────────────────
        scr  = supported / n
        cr   = contradicted / n
        tve  = outdated / max(temporal, 1)
        cdee = synthesis_errors / n_multi

        # ── Composite Hallucination Score ─────────────────────
        # CHS ∈ [0, ∞), lower is better
        chs = (
            self.scr_weight * (1 - scr)
            + self.cr_weight * cr
            + self.tve_weight * tve
            + self.cdee_weight * cdee
        ) * self.lambda_

        return HallucinationMetrics(
            scr=round(scr, 4),
            cr=round(cr, 4),
            tve=round(tve, 4),
            cdee=round(cdee, 4),
            chs=round(chs, 4),
            total_claims=n,
            supported_claims=supported,
            contradicted_claims=contradicted,
            temporal_claims=temporal,
            outdated_claims=outdated,
            synthesis_errors=synthesis_errors,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "MetricEngine":
        m = cfg.get("metrics", cfg)
        return cls(
            scr_weight=m.get("scr_weight", 1.0),
            cr_weight=m.get("conflict_weight", 1.0),
            tve_weight=m.get("tve_weight", 0.8),
            cdee_weight=m.get("cdee_weight", 0.8),
            lambda_=m.get("hallucination_lambda", 0.5),
        )