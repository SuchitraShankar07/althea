"""
training/signal_generator.py
Aggregates DiagnosisOutput records and converts them into
TrainingSample objects ready for QLoRA fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from loguru import logger

from ..diagnosis.diagnose import DiagnosisOutput
from ..generation.generator import build_prompt
from .qlora_trainer import TrainingSample


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
            prompt = build_prompt(query, docs)
            sample = TrainingSample(
                query=query,
                answer=diag.answer,
                chs=diag.metrics.chs,
                prompt=prompt,
            )
            samples.append(sample)

        logger.info(
            f"Generated {len(samples)} training samples | "
            f"avg CHS={sum(s.chs for s in samples)/max(len(samples),1):.3f}"
        )
        return samples

    def save(self, samples: List[TrainingSample]) -> None:
        """Save as JSONL (one JSON object per line)."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")
        logger.info(f"Saved {len(samples)} samples to {self.output_path}")

    @staticmethod
    def load(path: str) -> List[TrainingSample]:
        """Load training samples from JSONL or JSON array file."""
        samples = []
        raw_text = Path(path).read_text().strip()

        if raw_text.startswith("["):
            # JSON array format (legacy)
            for item in json.loads(raw_text):
                samples.append(TrainingSample.from_dict(item))
        else:
            # JSONL format (correct)
            for line in raw_text.split("\n"):
                line = line.strip()
                if line:
                    samples.append(TrainingSample.from_dict(json.loads(line)))

        logger.info(f"Loaded {len(samples)} samples from {path}")
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