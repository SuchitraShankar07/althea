"""
training/signal_generator.py
Aggregates DiagnosisOutput records and converts them into
TrainingSample objects ready for QLoRA fine-tuning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from loguru import logger

from ..diagnosis.diagnose import DiagnosisOutput
from .qlora_trainer import TrainingSample


_DUMMY_PATTERNS = [
    re.compile(r"sample response for:", re.IGNORECASE),
    re.compile(r"^placeholder", re.IGNORECASE),
    re.compile(r"lorem ipsum", re.IGNORECASE),
    re.compile(r"###\s*question", re.IGNORECASE),
    re.compile(r"###\s*answer", re.IGNORECASE),
]


def _is_placeholder(text: str) -> bool:
    return any(p.search(text or "") for p in _DUMMY_PATTERNS)


def validate_training_sample(sample: TrainingSample, line_no: int | None = None) -> None:
    where = f" (line {line_no})" if line_no is not None else ""
    if not sample.query or len(sample.query) < 8:
        raise ValueError(f"Invalid query{where}: must be at least 8 characters")
    if not sample.response or len(sample.response) < 16:
        raise ValueError(f"Invalid response{where}: must be at least 16 characters")
    if not isinstance(sample.contexts, list) or not sample.contexts:
        raise ValueError(f"Invalid contexts{where}: must contain at least one context string")
    if _is_placeholder(sample.query) or _is_placeholder(sample.response):
        raise ValueError(f"Placeholder/dummy text detected{where}")


class TrainingSignalGenerator:
    """
    Given a list of (query, retrieved_docs, DiagnosisOutput) triples,
    produces TrainingSample objects and optionally persists them to disk.
    """

    def __init__(self, output_path: str = "outputs/training_samples.jsonl"):
        self.output_path = output_path

    def convert(
        self,
        queries: List[str],
        docs_list: List[List[dict]],
        diagnosis_outputs: List[DiagnosisOutput],
    ) -> List[TrainingSample]:
        samples = []
        for query, docs, diag in zip(queries, docs_list, diagnosis_outputs):
            contexts = [d.get("text", "") for d in docs if d.get("text")]
            sample = TrainingSample(
                query=query,
                response=diag.answer,
                contexts=contexts,
                chs=diag.metrics.chs,
                scr=diag.metrics.scr,
                tve=diag.metrics.tve,
                cr=diag.metrics.cr,
                cdee=diag.metrics.cdee,
                retrieval_conflict=diag.metrics.retrieval_conflict,
                overgeneralization=diag.metrics.overgeneralization,
                outdated_information=diag.metrics.outdated_information,
                synthesis_error=diag.metrics.synthesis_error,
                overall_hallucination_score=diag.metrics.overall_hallucination_score,
                hallucination_confidence=diag.metrics.confidence,
            )
            validate_training_sample(sample)
            samples.append(sample)

        logger.info(
            f"Generated {len(samples)} training samples | "
            f"avg CHS={sum(s.chs for s in samples)/max(len(samples),1):.3f}"
        )
        return samples

    def save(self, samples: List[TrainingSample]) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            for s in samples:
                validate_training_sample(s)
                f.write(json.dumps(s.to_dict()) + "\n")
        logger.info(f"Saved {len(samples)} samples to {self.output_path}")

    @staticmethod
    def load(path: str) -> List[TrainingSample]:
        samples = []
        with open(path) as f:
            for idx, line in enumerate(f, start=1):
                d = json.loads(line.strip())
                sample = TrainingSample.from_dict(d)
                validate_training_sample(sample, line_no=idx)
                samples.append(sample)
        return samples

    def generate_and_save(
        self,
        queries: List[str],
        docs_list: List[List[dict]],
        diagnosis_outputs: List[DiagnosisOutput],
    ) -> List[TrainingSample]:
        samples = self.convert(queries, docs_list, diagnosis_outputs)
        self.save(samples)
        return samples