"""
diagnosis/claim_extractor.py
Splits an answer text into atomic, verifiable factual claims.

Strategy (two-stage):
  1. spaCy sentence segmentation + dependency heuristics
  2. Optionally a small LLM for higher-quality decomposition
"""

from __future__ import annotations

import re
from typing import List, Optional

from loguru import logger


# ── spaCy-based extractor (no LLM required) ───────────────────────────────────
class SpacyClaimExtractor:
    """
    Rule-based claim extractor using spaCy.
    Extracts sentences that contain at least one named entity or
    a subject-verb-object triple (simple heuristic).
    """

    def __init__(self, model: str = "en_core_web_sm", min_length: int = 6):
        try:
            import spacy
            self.nlp = spacy.load(model)
        except Exception as exc:
            logger.warning(
                f"spaCy unavailable ({exc}). "
                "Using regex fallback claim extraction."
            )
            self.nlp = None
        self.min_length = min_length

    def extract(self, answer: str, max_claims: int = 20) -> List[str]:
        if not answer.strip():
            return []
        if self.nlp is None:
            return self._fallback_split(answer, max_claims)

        doc = self.nlp(answer)
        claims = []
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text.split()) < self.min_length:
                continue
            if self._is_factual(sent):
                claims.append(text)
            if len(claims) >= max_claims:
                break
        return claims if claims else self._fallback_split(answer, max_claims)

    def _is_factual(self, sent) -> bool:
        """Heuristic: sentence has a named entity OR a root verb."""
        has_ent = len(sent.ents) > 0
        has_verb = any(t.pos_ == "VERB" and t.dep_ == "ROOT" for t in sent)
        return has_ent or has_verb

    @staticmethod
    def _fallback_split(text: str, max_claims: int) -> List[str]:
        """Naive sentence split by punctuation when spaCy is unavailable."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.split()) >= 4][:max_claims]


# ── LLM-based extractor (higher quality) ─────────────────────────────────────
CLAIM_EXTRACTION_PROMPT = """\
Break the following answer into a list of short, atomic, self-contained factual claims.
Each claim must be independently verifiable.
Return ONLY the claims, one per line, with no numbering or bullet points.

Answer:
{answer}

Claims:"""


class LLMClaimExtractor:
    """
    Uses a small instruction-tuned LLM (via HuggingFace pipeline)
    for higher-quality claim decomposition.
    Falls back to SpacyClaimExtractor on failure.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
        fallback: Optional[SpacyClaimExtractor] = None,
    ):
        from transformers import pipeline
        logger.info(f"Loading claim extractor LLM: {model_name}")
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=max_new_tokens,
        )
        self.fallback = fallback or SpacyClaimExtractor()

    def extract(self, answer: str, max_claims: int = 20) -> List[str]:
        try:
            prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
            out = self.pipe(prompt)[0]["generated_text"]
            claims = [line.strip() for line in out.strip().split("\n") if line.strip()]
            return claims[:max_claims]
        except Exception as e:
            logger.warning(f"LLM claim extraction failed ({e}), using fallback")
            return self.fallback.extract(answer, max_claims)


# ── Public factory ────────────────────────────────────────────────────────────
def get_claim_extractor(
    use_llm: bool = False,
    llm_model: str = "google/flan-t5-base",
    min_length: int = 6,
) -> SpacyClaimExtractor | LLMClaimExtractor:
    if use_llm:
        return LLMClaimExtractor(model_name=llm_model)
    return SpacyClaimExtractor(min_length=min_length)
