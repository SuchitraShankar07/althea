"""
training/qlora_trainer.py
Metric-guided QLoRA fine-tuning of the RAG generator.

Three training strategies:
    1. metric_loss   — cross-entropy + λ·hallucination_penalty
    2. dpo           — Direct Preference Optimisation on ranked answer pairs
    3. rejection     — SFT only on low-hallucination answers
"""

from __future__ import annotations

import json
import os
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from loguru import logger
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import DPOTrainer, SFTTrainer

try:
    from trl import SFTConfig
except Exception:  # pragma: no cover
    SFTConfig = None


# ── Training sample dataclasses ───────────────────────────────────────────────
@dataclass
class TrainingSample:
    """A single (query, answer) training example with its hallucination score."""
    query: str
    answer: str
    chs: float           # Composite Hallucination Score (lower = better)
    prompt: str = ""     # full prompt (populated during dataset prep)


@dataclass
class DPOPair:
    prompt: str
    chosen: str          # lower CHS (more factual)
    rejected: str        # higher CHS (more hallucinated)
    chs_gap: float


# ── QLoRA setup helpers ───────────────────────────────────────────────────────
def _load_base_model_for_training(model_name: str, cache_dir: Optional[str]):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU, but none was detected."
        )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def _apply_lora(model, lora_cfg: dict):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("lora_r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        bias="none",
    )
    return get_peft_model(model, config)


def _accepts_kwarg(callable_obj, kwarg_name: str) -> bool:
    """Return True if callable_obj accepts kwarg_name or **kwargs."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    return kwarg_name in sig.parameters


# ── Dataset preparation ───────────────────────────────────────────────────────
def prepare_sft_dataset(
    samples: List[TrainingSample],
    tokenizer,
    max_chs: float = 0.3,
) -> Dataset:
    """
    For rejection-sampling SFT: keep only samples below the CHS threshold.
    """
    filtered = [s for s in samples if s.chs <= max_chs]
    logger.info(f"SFT: {len(filtered)}/{len(samples)} samples pass CHS ≤ {max_chs}")
    texts = [s.prompt + s.answer + tokenizer.eos_token for s in filtered]
    return Dataset.from_dict({"text": texts})


def prepare_dpo_dataset(
    samples: List[TrainingSample],
    min_gap: float = 0.2,
) -> List[DPOPair]:
    """
    Build (chosen, rejected) pairs from samples with the same query.
    Pairs (a, b) where a.chs < b.chs and gap >= min_gap.
    """
    from itertools import combinations
    from collections import defaultdict

    by_query: dict[str, List[TrainingSample]] = defaultdict(list)
    for s in samples:
        by_query[s.query].append(s)

    pairs = []
    for query, group in by_query.items():
        group.sort(key=lambda x: x.chs)
        for a, b in combinations(group, 2):
            gap = b.chs - a.chs
            if gap >= min_gap:
                pairs.append(
                    DPOPair(
                        prompt=a.prompt,
                        chosen=a.answer,
                        rejected=b.answer,
                        chs_gap=gap,
                    )
                )
    logger.info(f"DPO: {len(pairs)} preference pairs created")
    return pairs


# ── Main trainer class ────────────────────────────────────────────────────────
class MetricGuidedQLoRATrainer:
    """
    Wraps model loading + LoRA setup + chosen training strategy.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        tr = cfg.get("training", cfg)
        self.model_name = cfg["generation"]["model_name"]
        self.output_dir = tr.get("output_dir", "outputs/qlora")
        self.method = tr.get("method", "dpo")
        self.cache_dir = cfg.get("paths", {}).get("cache_dir")

        # Training hyperparams
        self.num_epochs = tr.get("num_epochs", 3)
        self.batch_size = tr.get("per_device_train_batch_size", 2)
        self.grad_accum = tr.get("gradient_accumulation_steps", 8)
        self.lr = tr.get("learning_rate", 2e-4)
        self.warmup_ratio = tr.get("warmup_ratio", 0.05)
        self.dpo_beta = tr.get("dpo_beta", 0.1)
        self.min_score_gap = tr.get("min_score_gap", 0.2)
        self.penalty_lambda = tr.get("hallucination_penalty_lambda", 0.3)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_training_args(self, run_name: str) -> TrainingArguments:
        return TrainingArguments(
            output_dir=os.path.join(self.output_dir, run_name),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
            warmup_ratio=self.warmup_ratio,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no", 
            report_to="none",
            remove_unused_columns=False,
        )

    def _make_dpo_trainer(self, model, dataset, tokenizer):
        kwargs = {
            "model": model,
            "args": self._get_training_args("dpo"),
            "beta": self.dpo_beta,
            "train_dataset": dataset,
        }
        if _accepts_kwarg(DPOTrainer, "processing_class"):
            kwargs["processing_class"] = tokenizer
        elif _accepts_kwarg(DPOTrainer, "tokenizer"):
            kwargs["tokenizer"] = tokenizer
        return DPOTrainer(**kwargs)

    def _make_sft_trainer(self, model, dataset, tokenizer):
        if SFTConfig is not None:
            args = SFTConfig(
                output_dir=os.path.join(self.output_dir, "rejection_sft"),
                dataset_text_field="text",
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.grad_accum,
                learning_rate=self.lr,
                num_train_epochs=self.num_epochs,
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="no",
                report_to="none",
            )
        else:
            args = self._get_training_args("rejection_sft")

        kwargs = {
            "model": model,
            "args": args,
            "train_dataset": dataset,
        }
        if _accepts_kwarg(SFTTrainer, "processing_class"):
            kwargs["processing_class"] = tokenizer
        elif _accepts_kwarg(SFTTrainer, "tokenizer"):
            kwargs["tokenizer"] = tokenizer

        if _accepts_kwarg(SFTTrainer, "dataset_text_field"):
            kwargs["dataset_text_field"] = "text"
        if _accepts_kwarg(SFTTrainer, "max_seq_length"):
            kwargs["max_seq_length"] = 2048

        return SFTTrainer(**kwargs)

    # ── Strategy: DPO ─────────────────────────────────────────
    def train_dpo(self, samples: List[TrainingSample]) -> None:
        logger.info("=== DPO Training ===")
        model = _load_base_model_for_training(self.model_name, self.cache_dir)
        model = _apply_lora(model, self.cfg.get("training", {}))
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token

        pairs = prepare_dpo_dataset(samples, min_gap=self.min_score_gap)
        if not pairs:
            raise ValueError("No DPO pairs could be created — check score gaps.")

        dataset = Dataset.from_dict(
            {
                "prompt": [p.prompt for p in pairs],
                "chosen": [p.chosen for p in pairs],
                "rejected": [p.rejected for p in pairs],
            }
        )

        trainer = self._make_dpo_trainer(model=model, dataset=dataset, tokenizer=tokenizer)
        trainer.train()
        trainer.save_model()
        logger.info(f"DPO adapter saved to {self.output_dir}/dpo")

    # ── Strategy: SFT with rejection sampling ─────────────────
    def train_rejection_sft(self, samples: List[TrainingSample]) -> None:
        logger.info("=== Rejection-Sampling SFT ===")
        model = _load_base_model_for_training(self.model_name, self.cache_dir)
        model = _apply_lora(model, self.cfg.get("training", {}))
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token

        dataset = prepare_sft_dataset(samples, tokenizer, max_chs=0.7)
        trainer = self._make_sft_trainer(model=model, dataset=dataset, tokenizer=tokenizer)
        trainer.train()
        trainer.save_model()
        logger.info(f"SFT adapter saved to {self.output_dir}/rejection_sft")

    # ── Strategy: Metric-weighted loss ────────────────────────
    def train_metric_loss(self, samples: List[TrainingSample]) -> None:
        """
        Custom training loop: loss = CE + λ * hallucination_penalty
        hallucination_penalty = CHS score for the sample.
        """
        logger.info("=== Metric-Weighted Loss Training ===")
        from transformers import DataCollatorForLanguageModeling

        model = _load_base_model_for_training(self.model_name, self.cache_dir)
        model = _apply_lora(model, self.cfg.get("training", {}))
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token

        texts = [s.prompt + s.answer + tokenizer.eos_token for s in samples]
        chs_scores = [s.chs for s in samples]
        encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=2048, return_tensors="pt"
        )

        class MetricWeightedDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, chs_scores):
                self.encodings = encodings
                self.chs = chs_scores

            def __len__(self):
                return len(self.chs)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = item["input_ids"].clone()
                item["chs"] = torch.tensor(self.chs[idx], dtype=torch.float)
                return item

        dataset = MetricWeightedDataset(encodings, chs_scores)
        lam = self.penalty_lambda

        class MetricWeightedTrainer(Trainer):
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                chs = inputs.pop("chs")
                outputs = model(**inputs)
                ce_loss = outputs.loss
                penalty = chs.mean()
                loss = ce_loss + lam * penalty
                return (loss, outputs) if return_outputs else loss

        trainer_kwargs = {
            "model": model,
            "args": self._get_training_args("metric_loss"),
            "train_dataset": dataset,
        }
        if _accepts_kwarg(MetricWeightedTrainer, "processing_class"):
            trainer_kwargs["processing_class"] = tokenizer
        elif _accepts_kwarg(MetricWeightedTrainer, "tokenizer"):
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = MetricWeightedTrainer(**trainer_kwargs)
        trainer.train()
        trainer.save_model()
        logger.info(f"Metric-loss adapter saved to {self.output_dir}/metric_loss")

    # ── Dispatch ──────────────────────────────────────────────
    def train(self, samples: List[TrainingSample]) -> None:
        if self.method == "dpo":
            self.train_dpo(samples)
        elif self.method == "rejection":
            self.train_rejection_sft(samples)
        elif self.method == "metric_loss":
            self.train_metric_loss(samples)
        else:
            raise ValueError(f"Unknown training method: {self.method}")
