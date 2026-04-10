"""
diagnosis/verification_engine.py
NLI-based claim verification.

For each (claim, evidence) pair, a cross-encoder NLI model predicts:
    ENTAILMENT  → evidence supports the claim
    CONTRADICTION → evidence contradicts the claim
    NEUTRAL     → evidence neither supports nor contradicts

We also include an optional temporal check for outdated-knowledge claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch
from loguru import logger
from transformers import pipeline


# ── Data structures ───────────────────────────────────────────────────────────
class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass
class ClaimVerificationResult:
    claim: str
    label: NLILabel
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    is_temporal: bool = False
    is_outdated: bool = False

    @property
    def is_supported(self) -> bool:
        return self.label == NLILabel.ENTAILMENT

    @property
    def is_contradicted(self) -> bool:
        return self.label == NLILabel.CONTRADICTION


# ── Temporal keywords ─────────────────────────────────────────────────────────
TEMPORAL_PATTERNS = re.compile(
    r"\b(current(ly)?|now|today|recent(ly)?|latest|as of|this year|"
    r"last year|present|modern|upcoming|still|no longer|formerly)\b",
    re.IGNORECASE,
)

# Year references like "in 2024", "since 2022"
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _is_temporal_claim(claim: str) -> bool:
    return bool(TEMPORAL_PATTERNS.search(claim) or YEAR_PATTERN.search(claim))


# ── NLI Verification Engine ───────────────────────────────────────────────────
class NLIVerificationEngine:
    """
    Uses a cross-encoder NLI model to verify claims against evidence.

    Model: cross-encoder/nli-deberta-v3-base  (recommended)
    Accepts a list of evidence strings per claim and returns the
    *strongest* support / contradiction signal across all evidence.
    """

    LABEL_MAP = {
        "entailment": NLILabel.ENTAILMENT,
        "contradiction": NLILabel.CONTRADICTION,
        "neutral": NLILabel.NEUTRAL,
        # Some models use different label names
        "ENTAILMENT": NLILabel.ENTAILMENT,
        "CONTRADICTION": NLILabel.CONTRADICTION,
        "NEUTRAL": NLILabel.NEUTRAL,
        "LABEL_0": NLILabel.CONTRADICTION,   # mnli ordering
        "LABEL_1": NLILabel.NEUTRAL,
        "LABEL_2": NLILabel.ENTAILMENT,
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        batch_size: int = 16,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.6,
        device: Optional[int] = None,
    ):
        logger.info(f"Loading NLI model: {model_name}")
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,           # return all label scores
            device=device,
        )
        self.batch_size = batch_size
        self.ent_thresh = entailment_threshold
        self.con_thresh = contradiction_threshold

    # ── Core verification ─────────────────────────────────────
    def verify_claim(
        self, claim: str, evidence_list: List[str]
    ) -> ClaimVerificationResult:
        """
        Verify a single claim against multiple evidence passages.
        Returns the aggregated result (strongest signal wins).
        """
        is_temporal = _is_temporal_claim(claim)

        if not evidence_list:
            return ClaimVerificationResult(
                claim=claim,
                label=NLILabel.NEUTRAL,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
                is_temporal=is_temporal,
            )

        # Build premise-hypothesis pairs
        pairs = [{"text": ev, "text_pair": claim} for ev in evidence_list]

        # Run NLI in batches
        all_scores = self.pipe(pairs, batch_size=self.batch_size)

        best_ent, best_con, best_neu = 0.0, 0.0, 0.0
        supporting = []

        for ev_text, scores in zip(evidence_list, all_scores):
            score_map = {self.LABEL_MAP.get(s["label"], NLILabel.NEUTRAL): s["score"] for s in scores}
            ent = score_map.get(NLILabel.ENTAILMENT, 0.0)
            con = score_map.get(NLILabel.CONTRADICTION, 0.0)
            neu = score_map.get(NLILabel.NEUTRAL, 0.0)

            if ent > best_ent:
                best_ent = ent
            if con > best_con:
                best_con = con
            if neu > best_neu:
                best_neu = neu

            if ent >= self.ent_thresh:
                supporting.append(ev_text)

        # Determine final label
        if best_ent >= self.ent_thresh:
            label = NLILabel.ENTAILMENT
        elif best_con >= self.con_thresh:
            label = NLILabel.CONTRADICTION
        else:
            label = NLILabel.NEUTRAL

        return ClaimVerificationResult(
            claim=claim,
            label=label,
            entailment_score=best_ent,
            contradiction_score=best_con,
            neutral_score=best_neu,
            supporting_evidence=supporting,
            is_temporal=is_temporal,
            is_outdated=is_temporal and label != NLILabel.ENTAILMENT,
        )

    def verify_claims(
        self, claims: List[str], evidence_lists: List[List[str]]
    ) -> List[ClaimVerificationResult]:
        """Verify a batch of claims."""
        assert len(claims) == len(evidence_lists)
        return [
            self.verify_claim(c, e)
            for c, e in zip(claims, evidence_lists)
        ]

    @classmethod
    def from_config(cls, cfg: dict) -> "NLIVerificationEngine":
        return cls(
            model_name=cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"),
            batch_size=cfg.get("nli_batch_size", 16),
            entailment_threshold=cfg.get("entailment_threshold", 0.7),
            contradiction_threshold=cfg.get("contradiction_threshold", 0.6),
        )