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
import re
from typing import Dict, List, Tuple

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

    # Extended taxonomy scores [0, 1], higher means more likely hallucination.
    retrieval_conflict: float = 0.0
    overgeneralization: float = 0.0
    outdated_information: float = 0.0
    synthesis_error: float = 0.0

    # Binary labels derived from the above scores.
    retrieval_conflict_label: bool = False
    overgeneralization_label: bool = False
    outdated_information_label: bool = False
    synthesis_error_label: bool = False

    # Aggregated extension signals.
    confidence: float = 0.0
    overall_hallucination_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"SCR={self.scr:.3f} | CR={self.cr:.3f} | "
            f"TVE={self.tve:.3f} | CDEE={self.cdee:.3f} | "
            f"CHS={self.chs:.3f} | OHS={self.overall_hallucination_score:.3f}  "
            f"({self.supported_claims}/{self.total_claims} supported)"
        )

    def hallucination_scores(self) -> Dict[str, float]:
        return {
            "retrieval_conflict": self.retrieval_conflict,
            "overgeneralization": self.overgeneralization,
            "outdated_information": self.outdated_information,
            "synthesis_error": self.synthesis_error,
        }

    def hallucination_labels(self) -> Dict[str, bool]:
        return {
            "retrieval_conflict": self.retrieval_conflict_label,
            "overgeneralization": self.overgeneralization_label,
            "outdated_information": self.outdated_information_label,
            "synthesis_error": self.synthesis_error_label,
        }

    def to_hallucination_record(
        self,
        query: str,
        response: str,
        contexts: List[str],
    ) -> Dict[str, object]:
        return {
            "query": query,
            "response": response,
            "contexts": contexts,
            "hallucination_scores": self.hallucination_scores(),
            "hallucination_labels": self.hallucination_labels(),
            "confidence": self.confidence,
            "overall_hallucination_score": self.overall_hallucination_score,
        }


_GENERALIZATION_PATTERN = re.compile(
    r"\b(always|all|everyone|never|every|entirely|completely|none|must|definitely)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_DUMMY_PATTERN = re.compile(r"\b(lorem ipsum|placeholder|sample response for:)\b", re.IGNORECASE)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _extract_fact_signature(text: str) -> Dict[str, set]:
    years = set(_YEAR_PATTERN.findall(text or ""))
    numbers = set(_NUMBER_PATTERN.findall(text or ""))
    entities = set(e.lower() for e in _ENTITY_PATTERN.findall(text or ""))
    return {"years": years, "numbers": numbers, "entities": entities}


def _pairwise_agreement(contexts: List[str]) -> float:
    if len(contexts) < 2:
        return 1.0
    agreements = []
    token_sets = [set(_tokenize(c)) for c in contexts]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            inter = token_sets[i] & token_sets[j]
            union = token_sets[i] | token_sets[j]
            agreements.append(_safe_ratio(len(inter), len(union) or 1))
    return sum(agreements) / max(len(agreements), 1)


def detect_retrieval_conflict(
    contexts: List[str],
    verification_results: List[ClaimVerificationResult] | None = None,
) -> float:
    """
    Detect contradictions and low agreement across retrieved contexts.
    Returns normalized risk score in [0, 1].
    """
    valid_contexts = [c for c in contexts if c and c.strip()]
    if len(valid_contexts) < 2:
        return 0.0

    signatures = [_extract_fact_signature(c) for c in valid_contexts]
    year_conflicts, number_conflicts = 0, 0
    pair_count = 0
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            pair_count += 1
            yi, yj = signatures[i]["years"], signatures[j]["years"]
            ni, nj = signatures[i]["numbers"], signatures[j]["numbers"]
            if yi and yj and yi != yj and not (yi & yj):
                year_conflicts += 1
            if ni and nj and ni != nj and not (ni & nj):
                number_conflicts += 1

    explicit_conflict = _safe_ratio(year_conflicts + number_conflicts, max(pair_count, 1) * 2)
    agreement_penalty = 1.0 - _pairwise_agreement(valid_contexts)
    contradiction_signal = 0.0
    if verification_results:
        contradiction_signal = _safe_ratio(
            sum(1 for r in verification_results if r.label == NLILabel.CONTRADICTION),
            len(verification_results),
        )

    return _clip01(0.45 * explicit_conflict + 0.35 * agreement_penalty + 0.20 * contradiction_signal)


def detect_overgeneralization(
    response: str,
    contexts: List[str],
    verification_results: List[ClaimVerificationResult] | None = None,
) -> float:
    """
    Detect claims likely extending beyond available evidence.
    Returns normalized risk score in [0, 1].
    """
    if not (response or "").strip():
        return 0.0

    response_tokens = set(_tokenize(response))
    context_tokens = set(_tokenize(" ".join(contexts or [])))
    uncovered = _safe_ratio(len(response_tokens - context_tokens), max(len(response_tokens), 1))

    over_words = len(_GENERALIZATION_PATTERN.findall(response or ""))
    sentence_count = max(1, len(re.findall(r"[.!?]", response or "")) or 1)
    language_signal = _clip01(over_words / sentence_count)

    unsupported_signal = 0.0
    if verification_results:
        unsupported_signal = _safe_ratio(
            sum(1 for r in verification_results if r.label != NLILabel.ENTAILMENT),
            len(verification_results),
        )

    return _clip01(0.40 * unsupported_signal + 0.35 * language_signal + 0.25 * uncovered)


def detect_outdated_information(
    response: str,
    contexts: List[str],
    verification_results: List[ClaimVerificationResult] | None = None,
) -> float:
    """
    Detect temporally stale or inconsistent information.
    Returns normalized risk score in [0, 1].
    """
    if not (response or "").strip():
        return 0.0

    resp_years = [int(y) for y in _YEAR_PATTERN.findall(response or "")]
    context_years = [int(y) for y in _YEAR_PATTERN.findall(" ".join(contexts or []))]

    temporal_mismatch = 0.0
    if resp_years and context_years:
        newest_context = max(context_years)
        newest_response = max(resp_years)
        oldest_response = min(resp_years)
        if newest_response > newest_context + 2:
            temporal_mismatch = 1.0
        elif oldest_response < newest_context - 5:
            temporal_mismatch = 0.6

    outdated_signal = 0.0
    temporal_unsupported = 0.0
    if verification_results:
        temporal_claims = [r for r in verification_results if r.is_temporal]
        outdated_signal = _safe_ratio(sum(1 for r in temporal_claims if r.is_outdated), len(temporal_claims))
        temporal_unsupported = _safe_ratio(
            sum(1 for r in temporal_claims if r.label != NLILabel.ENTAILMENT),
            len(temporal_claims),
        )

    return _clip01(0.45 * outdated_signal + 0.30 * temporal_unsupported + 0.25 * temporal_mismatch)


def detect_synthesis_error(
    verification_results: List[ClaimVerificationResult],
    docs_per_claim: List[int] | None = None,
) -> float:
    """
    Detect unsupported multi-document synthesis and fabricated relationships.
    Returns normalized risk score in [0, 1].
    """
    if not verification_results:
        return 0.0

    if docs_per_claim is None:
        docs_per_claim = [1] * len(verification_results)

    multi = [
        r for r, nd in zip(verification_results, docs_per_claim) if nd > 1
    ]
    if not multi:
        multi = [r for r in verification_results if len((r.claim or "").split()) > 12]

    unsupported_multi = _safe_ratio(
        sum(1 for r in multi if r.label != NLILabel.ENTAILMENT),
        len(multi),
    )
    contradiction_multi = _safe_ratio(
        sum(1 for r in multi if r.label == NLILabel.CONTRADICTION),
        len(multi),
    )

    return _clip01(0.65 * unsupported_multi + 0.35 * contradiction_multi)


def compute_overall_hallucination_score(scores: Dict[str, float]) -> float:
    return _clip01(
        0.3 * _clip01(scores.get("retrieval_conflict", 0.0))
        + 0.25 * _clip01(scores.get("synthesis_error", 0.0))
        + 0.25 * _clip01(scores.get("overgeneralization", 0.0))
        + 0.2 * _clip01(scores.get("outdated_information", 0.0))
    )


def _validate_hallucination_inputs(response: str, contexts: List[str]) -> Tuple[bool, str]:
    if not isinstance(contexts, list) or not any((c or "").strip() for c in contexts):
        return False, "contexts-empty"
    if len((response or "").strip()) < 20:
        return False, "response-too-short"
    if _DUMMY_PATTERN.search(response or ""):
        return False, "dummy-response"
    return True, "ok"


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
        response: str = "",
        contexts: List[str] | None = None,
        enable_hallucination_eval: bool = True,
    ) -> HallucinationMetrics:
        """
        results         : per-claim NLI verification results
        docs_per_claim  : number of distinct source documents per claim
                          (used to detect cross-document synthesis errors)
        """
        contexts = contexts or []

        if not results:
            empty = HallucinationMetrics()
            if enable_hallucination_eval:
                valid, _ = _validate_hallucination_inputs(response=response, contexts=contexts)
                if valid:
                    scores = {
                        "retrieval_conflict": detect_retrieval_conflict(contexts, results),
                        "overgeneralization": detect_overgeneralization(response, contexts, results),
                        "outdated_information": detect_outdated_information(response, contexts, results),
                        "synthesis_error": detect_synthesis_error(results, docs_per_claim),
                    }
                    empty.retrieval_conflict = round(scores["retrieval_conflict"], 4)
                    empty.overgeneralization = round(scores["overgeneralization"], 4)
                    empty.outdated_information = round(scores["outdated_information"], 4)
                    empty.synthesis_error = round(scores["synthesis_error"], 4)
                    empty.overall_hallucination_score = round(compute_overall_hallucination_score(scores), 4)
                    empty.confidence = round(_clip01(0.4 + 0.6 * min(len(contexts), 3) / 3), 4)
                    empty.retrieval_conflict_label = empty.retrieval_conflict >= 0.5
                    empty.overgeneralization_label = empty.overgeneralization >= 0.5
                    empty.outdated_information_label = empty.outdated_information >= 0.5
                    empty.synthesis_error_label = empty.synthesis_error >= 0.5
            return empty

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

        retrieval_conflict = 0.0
        overgeneralization = 0.0
        outdated_information = 0.0
        synthesis_error = 0.0
        overall_hallucination_score = 0.0
        confidence = 0.0
        retrieval_conflict_label = False
        overgeneralization_label = False
        outdated_information_label = False
        synthesis_error_label = False

        if enable_hallucination_eval:
            valid, reason = _validate_hallucination_inputs(response=response, contexts=contexts)
            if valid:
                scores = {
                    "retrieval_conflict": detect_retrieval_conflict(contexts, results),
                    "overgeneralization": detect_overgeneralization(response, contexts, results),
                    "outdated_information": detect_outdated_information(response, contexts, results),
                    "synthesis_error": detect_synthesis_error(results, docs_per_claim),
                }
                retrieval_conflict = scores["retrieval_conflict"]
                overgeneralization = scores["overgeneralization"]
                outdated_information = scores["outdated_information"]
                synthesis_error = scores["synthesis_error"]
                overall_hallucination_score = compute_overall_hallucination_score(scores)
                confidence = _clip01(
                    0.35
                    + 0.35 * min(n, 8) / 8
                    + 0.30 * min(len(contexts), 4) / 4
                )
                retrieval_conflict_label = retrieval_conflict >= 0.5
                overgeneralization_label = overgeneralization >= 0.5
                outdated_information_label = outdated_information >= 0.5
                synthesis_error_label = synthesis_error >= 0.5
            else:
                # Fail safely with zeroed taxonomy signals so existing pipeline continues.
                confidence = 0.1
                if reason == "contexts-empty":
                    retrieval_conflict = 0.0

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
            retrieval_conflict=round(retrieval_conflict, 4),
            overgeneralization=round(overgeneralization, 4),
            outdated_information=round(outdated_information, 4),
            synthesis_error=round(synthesis_error, 4),
            retrieval_conflict_label=retrieval_conflict_label,
            overgeneralization_label=overgeneralization_label,
            outdated_information_label=outdated_information_label,
            synthesis_error_label=synthesis_error_label,
            confidence=round(confidence, 4),
            overall_hallucination_score=round(overall_hallucination_score, 4),
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