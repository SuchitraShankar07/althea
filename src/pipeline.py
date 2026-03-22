"""Althea pipeline: orchestrates retrieval, generation, diagnosis and evaluation."""

from typing import Dict, List, Optional

import yaml


class Pipeline:
    """End-to-end RAG pipeline with built-in hallucination diagnosis."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        self._encoder = None
        self._retriever = None
        self._generator = None
        self._claim_extractor = None
        self._evidence_retriever = None
        self._verification_engine = None
        self._metric_engine = None
        self._evaluator = None

    # ------------------------------------------------------------------
    # Lazy component initialisation
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            from src.retrieval import Encoder

            cfg = self.config.get("retrieval", {})
            self._encoder = Encoder(
                model_name=cfg.get(
                    "model_name", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        return self._encoder

    def _get_retriever(self):
        if self._retriever is None:
            from src.retrieval import Retriever

            cfg = self.config.get("retrieval", {})
            self._retriever = Retriever(
                encoder=self._get_encoder(),
                index_path=cfg.get("index_path", "data/index"),
                top_k=cfg.get("top_k", 5),
            )
        return self._retriever

    def _get_generator(self):
        if self._generator is None:
            from src.generation import Generator

            cfg = self.config.get("generation", {})
            self._generator = Generator(
                model_name=cfg.get(
                    "model_name", "mistralai/Mistral-7B-Instruct-v0.2"
                ),
                max_new_tokens=cfg.get("max_new_tokens", 512),
                temperature=cfg.get("temperature", 0.7),
                top_p=cfg.get("top_p", 0.9),
            )
        return self._generator

    def _get_diagnosis_components(self):
        if self._claim_extractor is None:
            from src.diagnosis import (
                ClaimExtractor,
                EvidenceRetriever,
                MetricEngine,
                VerificationEngine,
            )

            diag_cfg = self.config.get("diagnosis", {})
            ce_cfg = diag_cfg.get("claim_extractor", {})
            ve_cfg = diag_cfg.get("verification_engine", {})

            self._claim_extractor = ClaimExtractor(
                model_name=ce_cfg.get("model_name", "gpt-3.5-turbo")
            )
            self._evidence_retriever = EvidenceRetriever(
                retriever=self._get_retriever(),
                top_k=self.config.get("retrieval", {}).get("top_k", 3),
            )
            self._verification_engine = VerificationEngine(
                entailment_model=ve_cfg.get(
                    "entailment_model", "cross-encoder/nli-deberta-v3-base"
                ),
                confidence_threshold=ve_cfg.get("confidence_threshold", 0.5),
            )
            self._metric_engine = MetricEngine()

    def _get_evaluator(self):
        if self._evaluator is None:
            from src.evaluation import Evaluator

            eval_cfg = self.config.get("evaluation", {})
            self._evaluator = Evaluator(metrics=eval_cfg.get("metrics"))
        return self._evaluator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        run_diagnosis: bool = True,
        run_evaluation: bool = False,
    ) -> Dict:
        """Execute the full pipeline for a single *query*.

        Args:
            query: The user question.
            ground_truth: Optional reference answer used for evaluation metrics.
            run_diagnosis: Whether to run hallucination diagnosis on the answer.
            run_evaluation: Whether to run the RAG quality evaluator.

        Returns:
            A dict with keys ``"query"``, ``"contexts"``, ``"answer"``,
            and optionally ``"diagnosis"`` and ``"evaluation"``.
        """
        # 1. Retrieve relevant passages.
        retriever = self._get_retriever()
        retrieved = retriever.retrieve(query)
        contexts: List[str] = [doc.get("text", "") for doc, _ in retrieved]

        # 2. Generate an answer.
        generator = self._get_generator()
        answer = generator.generate(query, context_passages=contexts)

        result: Dict = {
            "query": query,
            "contexts": contexts,
            "answer": answer,
        }

        # 3. (Optional) Diagnose hallucinations.
        if run_diagnosis:
            self._get_diagnosis_components()
            claims = self._claim_extractor.extract(answer)
            verdicts = []
            for claim in claims:
                evidence = self._evidence_retriever.retrieve_evidence(claim)
                evidence_docs = [doc for doc, _ in evidence]
                verdict = self._verification_engine.verify(claim, evidence_docs)
                verdicts.append(verdict)
            result["diagnosis"] = self._metric_engine.compute(verdicts)

        # 4. (Optional) Evaluate response quality.
        if run_evaluation:
            evaluator = self._get_evaluator()
            result["evaluation"] = evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

        return result
