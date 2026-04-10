"""
src/pipeline.py
Master pipeline — wires all five modules together.

Supports three modes:
    inference  — query → answer + hallucination diagnosis
    evaluate   — run on a dataset, compute aggregate metrics
    train      — collect training samples and fine-tune with QLoRA
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import yaml
from loguru import logger

from .diagnosis.diagnose import DiagnosisOutput, HallucinationDiagnoser
from .evaluation.evaluator import RAGEvaluator
from .generation.generator import RAGGenerator
from .retrieval.retriever import Retriever
from .training.qlora_trainer import MetricGuidedQLoRATrainer
from .training.signal_generator import TrainingSignalGenerator


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Main pipeline class ───────────────────────────────────────────────────────
class FailureAwareRAGPipeline:
    """
    Full Failure-Aware RAG pipeline.

    Usage:
        pipeline = FailureAwareRAGPipeline.from_config("config/config.yaml")
        result = pipeline.run_inference("What is the capital of France?")
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: RAGGenerator,
        diagnoser: HallucinationDiagnoser,
        evaluator: Optional[RAGEvaluator] = None,
        cfg: Optional[dict] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.diagnoser = diagnoser
        self.evaluator = evaluator or RAGEvaluator()
        self.cfg = cfg or {}

    # ── Single-query inference ────────────────────────────────
    def run_inference(self, query: str, diagnose: bool = True) -> dict:
        """
        Returns:
            {query, answer, documents, diagnosis (optional)}
        """
        # 1. Retrieve
        documents = self.retriever.retrieve(query)
        logger.info(f"Retrieved {len(documents)} documents")

        # 2. Generate
        answer = self.generator.generate(query, documents)
        logger.info(f"Answer: {answer[:120]}...")

        result = {"query": query, "answer": answer, "documents": documents}

        # 3. Diagnose (optional)
        if diagnose:
            diag = self.diagnoser.diagnose(answer, original_docs=documents)
            result["diagnosis"] = {
                "claims": diag.claims,
                "metrics": diag.metrics.to_dict(),
                "summary": diag.metrics.summary(),
            }
            logger.info(f"Diagnosis: {diag.metrics.summary()}")

        return result

    # ── Batch inference ───────────────────────────────────────
    def run_batch(
        self,
        queries: List[str],
        ground_truths: Optional[List[str]] = None,
        diagnose: bool = True,
        model_tag: str = "baseline",
    ) -> List[dict]:
        results, diagnoses, predictions, hal_metrics = [], [], [], []

        for i, query in enumerate(queries):
            logger.info(f"[{i+1}/{len(queries)}] {query[:60]}")
            r = self.run_inference(query, diagnose=diagnose)
            results.append(r)
            predictions.append(r["answer"])
            if diagnose:
                hal_metrics.append(r["diagnosis"]["metrics"])

        # Optional evaluation
        if ground_truths and diagnose:
            from .diagnosis.metric_engine import HallucinationMetrics
            hm_objs = [HallucinationMetrics(**m) for m in hal_metrics]
            self.evaluator.evaluate(
                queries, predictions, ground_truths, hm_objs, model_tag=model_tag
            )

        return results

    # ── Training data collection + fine-tuning ────────────────
    def collect_training_data(
        self,
        queries: List[str],
        save_path: str = "outputs/training_samples.jsonl",
        n_samples_per_query: int = 2,
        temperatures: Optional[List[float]] = None,
    ) -> List:
        """
        Run inference + diagnosis on all queries, save training samples.

        For DPO, generates multiple answers per query at different temperatures
        so we get real preference pairs (both sides are actual model outputs).
        """
        if temperatures is None:
            train_cfg = self.cfg.get("training", {})
            temperatures = train_cfg.get("collection_temperatures", [0.1, 0.7])
            n_samples_per_query = train_cfg.get("n_samples_per_query", 2)

        # Ensure enough temperatures
        while len(temperatures) < n_samples_per_query:
            temperatures.append(temperatures[-1] + 0.2)

        sig_gen = TrainingSignalGenerator(output_path=save_path)
        all_queries, docs_list, diag_outputs = [], [], []

        for i, query in enumerate(queries):
            logger.info(f"Collecting [{i+1}/{len(queries)}] {query[:60]}")
            docs = self.retriever.retrieve(query)

            for t_idx in range(n_samples_per_query):
                temp = temperatures[t_idx]
                answer = self.generator.generate(query, docs, temperature=temp)
                diag = self.diagnoser.diagnose(answer, original_docs=docs)
                diag.answer = answer

                all_queries.append(query)
                docs_list.append(docs)
                diag_outputs.append(diag)

                logger.debug(
                    f"  temp={temp:.1f} CHS={diag.metrics.chs:.3f} "
                    f"answer={answer[:80]}"
                )

        return sig_gen.generate_and_save(all_queries, docs_list, diag_outputs)

    def run_training(
        self,
        samples=None,
        samples_path: str = "outputs/training_samples.jsonl",
    ) -> None:
        """
        Fine-tune the generator using the configured strategy.
        """
        if samples is None:
            samples = TrainingSignalGenerator.load(samples_path)
        trainer = MetricGuidedQLoRATrainer(self.cfg)
        trainer.train(samples)

    # ── Factory ───────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "FailureAwareRAGPipeline":
        cfg = load_config(config_path)

        logger.info("=== Initialising Failure-Aware RAG Pipeline ===")

        retriever = Retriever.from_config(cfg["retrieval"])
        generator = RAGGenerator.from_config(cfg["generation"])
        diagnoser = HallucinationDiagnoser.from_config(cfg, retriever)
        evaluator = RAGEvaluator(
            results_dir=cfg.get("evaluation", {}).get("results_dir", "outputs/eval")
        )

        return cls(
            retriever=retriever,
            generator=generator,
            diagnoser=diagnoser,
            evaluator=evaluator,
            cfg=cfg,
        )