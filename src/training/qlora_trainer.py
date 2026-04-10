"""
training/qlora_trainer.py
QLoRA fine-tuning guided by hallucination metrics (CHS scores).

Supports three strategies:
    dpo       — Direct Preference Optimization with CHS-based pairs
    rejection — SFT on low-CHS (well-grounded) samples only
    metric_loss — SFT weighted by answer quality (1 - CHS)

Key design decisions:
    - No silent model overrides: trains the SAME model used for inference
    - Proper DPO pairing: groups samples by query, pairs lowest vs highest CHS
    - CHS lower = better: chosen answers have LOW CHS, rejected have HIGH CHS
    - GPU memory auto-detected from hardware, not hardcoded
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class TrainingSample:
    """Single training sample with hallucination score."""

    query: str
    answer: str       # model-generated answer text
    prompt: str       # full prompt (system + context + question)
    chs: float        # Composite Hallucination Score (lower = better)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingSample":
        """Create from dict, handling legacy field names."""
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", data.get("response", "")),
            prompt=data.get("prompt", ""),
            chs=float(data.get("chs", 1.0)),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _detect_gpu_memory() -> float:
    """Return total GPU VRAM in GB, or 0 if no GPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0


def _auto_detect_lora_targets(model) -> List[str]:
    """Find valid LoRA target module names for the model architecture."""
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    candidates = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],   # Llama, Mistral, Qwen
        ["query_key_value"],                          # Falcon
        ["c_attn", "c_proj"],                         # GPT-2
        ["qkv_proj", "o_proj"],                       # Phi
    ]
    for targets in candidates:
        if all(t in module_names for t in targets):
            return targets

    # Fallback: any *_proj modules
    proj = sorted(n for n in module_names if n.endswith("_proj"))
    if proj:
        return proj[:4]

    raise ValueError(
        f"Cannot auto-detect LoRA targets. Available modules: {sorted(module_names)}"
    )


# ── Main trainer ──────────────────────────────────────────────────────────────
class MetricGuidedQLoRATrainer:
    """
    QLoRA trainer driven by CHS (Composite Hallucination Score).

    CHS interpretation:
        0.0 – 0.3  →  good, well-grounded answer
        0.3 – 0.6  →  moderate hallucination
        0.6 – 1.0  →  severe hallucination
        > 1.0      →  active contradictions

    For DPO:
        chosen  = answer with LOWEST CHS  (best grounded)
        rejected = answer with HIGHEST CHS (most hallucinated)
    """

    def __init__(self, config: dict):
        self.config = config
        self.gen_cfg = config.get("generation", {})
        self.train_cfg = config.get("training", {})

        # Use the SAME model as inference — no silent overrides
        self.model_name = self.gen_cfg.get(
            "model_name", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.cache_dir = config.get("paths", {}).get("cache_dir", ".cache")
        self.output_dir = self.train_cfg.get("output_dir", "outputs/qlora_model")

        # GPU auto-detection
        self.gpu_memory = _detect_gpu_memory()
        if self.gpu_memory > 0:
            logger.info(f"GPU VRAM: {self.gpu_memory:.1f} GB")
        else:
            logger.warning("No GPU detected — training will be very slow or fail")

        # Tokenizer (lightweight, loaded early)
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── Public API ────────────────────────────────────────────────────────────
    def train(self, samples: list) -> None:
        method = self.train_cfg.get("method", "dpo")
        logger.info(f"Training strategy: {method} | Model: {self.model_name}")

        if method == "dpo":
            self._train_dpo(samples)
        elif method == "rejection":
            self._train_rejection_sft(samples)
        elif method == "metric_loss":
            self._train_metric_weighted_sft(samples)
        else:
            raise ValueError(f"Unknown training method: {method}")

    # ── Normalise samples ─────────────────────────────────────────────────────
    def _normalise(self, sample) -> dict:
        """Convert any sample format to {query, answer, prompt, chs}."""
        if isinstance(sample, TrainingSample):
            return sample.to_dict()
        if isinstance(sample, dict):
            d = sample
        else:
            d = {
                a: getattr(sample, a)
                for a in dir(sample)
                if not a.startswith("_") and not callable(getattr(sample, a))
            }
        return {
            "query": d.get("query", ""),
            "answer": d.get("answer", d.get("response", "")),
            "prompt": d.get("prompt", ""),
            "chs": float(d.get("chs", 1.0)),
        }

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        """Load base model in 4-bit for QLoRA training."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Use 85% of detected VRAM (leave headroom for activations)
        max_mem = f"{int(self.gpu_memory * 0.85)}GB" if self.gpu_memory else None
        mem_kw = {"max_memory": {0: max_mem}} if max_mem else {}

        logger.info(
            f"Loading {self.model_name} (4-bit) for training "
            f"[max_memory={max_mem or 'auto'}]"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **mem_kw,
        )
        return model

    def _apply_lora(self, model):
        """Prepare 4-bit model and attach LoRA adapters."""
        model = prepare_model_for_kbit_training(model)

        targets = self.train_cfg.get("lora_target_modules")
        if not targets:
            targets = _auto_detect_lora_targets(model)

        lora_cfg = LoraConfig(
            r=self.train_cfg.get("lora_r", 16),
            lora_alpha=self.train_cfg.get("lora_alpha", 32),
            target_modules=targets,
            lora_dropout=self.train_cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )

        try:
            model = get_peft_model(model, lora_cfg)
        except ValueError:
            logger.warning(
                f"Configured LoRA targets {targets} failed; auto-detecting..."
            )
            targets = _auto_detect_lora_targets(model)
            lora_cfg = LoraConfig(
                r=self.train_cfg.get("lora_r", 16),
                lora_alpha=self.train_cfg.get("lora_alpha", 32),
                target_modules=targets,
                lora_dropout=self.train_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

        logger.info(f"LoRA targets: {targets}")
        model.print_trainable_parameters()
        return model

    def _save(self, model) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Adapter saved to {self.output_dir}")

    def _cleanup(self, *objs) -> None:
        for o in objs:
            if o is not None:
                del o
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    #  DPO  —  Direct Preference Optimization
    # ══════════════════════════════════════════════════════════════════════════
    def _train_dpo(self, samples: list) -> None:
        """
        Proper DPO training:
          1. Group samples by query
          2. For queries with 2+ answers, pair LOWEST CHS (chosen) vs HIGHEST CHS (rejected)
          3. Only include pairs where CHS gap ≥ min_score_gap
        """
        logger.info("=== DPO Training ===")

        normalised = [self._normalise(s) for s in samples]
        logger.info(f"Total samples: {len(normalised)}")

        # Group by query
        query_groups: dict[str, list] = defaultdict(list)
        for s in normalised:
            if s["query"] and s["answer"]:
                query_groups[s["query"]].append(s)

        logger.info(f"Unique queries: {len(query_groups)}")

        # Build DPO pairs
        min_gap = self.train_cfg.get("min_score_gap", 0.2)
        dpo_pairs = []

        for query, group in query_groups.items():
            if len(group) < 2:
                # Single sample per query — cannot create a preference pair
                continue

            # Sort by CHS ascending: best (lowest CHS) first
            group.sort(key=lambda x: x["chs"])
            best = group[0]    # lowest CHS = most grounded = CHOSEN
            worst = group[-1]  # highest CHS = most hallucinated = REJECTED

            gap = worst["chs"] - best["chs"]
            if gap < min_gap:
                logger.debug(
                    f"Skipping query (CHS gap {gap:.3f} < {min_gap}): {query[:60]}"
                )
                continue

            # Use the full prompt if available, else just the query
            prompt = best.get("prompt") or best["query"]

            dpo_pairs.append(
                {
                    "prompt": prompt,
                    "chosen": best["answer"],    # LOW CHS  = good
                    "rejected": worst["answer"],  # HIGH CHS = bad
                }
            )

        logger.info(
            f"Created {len(dpo_pairs)} DPO pairs "
            f"(from {len(query_groups)} queries, min_gap={min_gap})"
        )

        if not dpo_pairs:
            logger.error(
                "No DPO pairs created! Possible causes:\n"
                "  1. Only 1 sample per query (need ≥2; set n_samples_per_query ≥ 2)\n"
                "  2. CHS gap too small (lower min_score_gap in config)\n"
                "  → Falling back to rejection sampling SFT"
            )
            self._train_rejection_sft(samples)
            return

        # Load model + LoRA
        model = self._load_model()
        model = self._apply_lora(model)

        train_dataset = Dataset.from_list(dpo_pairs)

        # DPO config
        from trl import DPOConfig, DPOTrainer

        dpo_args = DPOConfig(
            output_dir=str(Path(self.output_dir) / "checkpoints"),
            per_device_train_batch_size=self.train_cfg.get(
                "per_device_train_batch_size", 1
            ),
            gradient_accumulation_steps=self.train_cfg.get(
                "gradient_accumulation_steps", 8
            ),
            num_train_epochs=self.train_cfg.get("num_epochs", 3),
            learning_rate=self.train_cfg.get("learning_rate", 2e-4),
            lr_scheduler_type="cosine",
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.05),
            logging_steps=1,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            dataloader_num_workers=0,
            report_to="none",
            eval_strategy="no",
            max_grad_norm=0.5,
            beta=self.train_cfg.get("dpo_beta", 0.1),
            max_length=512,
        )

        trainer = None
        try:
            trainer = DPOTrainer(
                model=model,
                args=dpo_args,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
            )

            logger.info("Starting DPO training...")
            trainer.train()
            self._save(model)
            logger.info("✓ DPO training completed")

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU OOM during DPO training. Try:\n"
                "  training.per_device_train_batch_size: 1\n"
                "  training.gradient_accumulation_steps: 16"
            )
            raise
        finally:
            self._cleanup(trainer, model)

    # ══════════════════════════════════════════════════════════════════════════
    #  Rejection Sampling SFT
    # ══════════════════════════════════════════════════════════════════════════
    def _train_rejection_sft(self, samples: list) -> None:
        """
        Keep only well-grounded answers (CHS ≤ 0.3) and SFT on them.
        """
        logger.info("=== Rejection Sampling SFT ===")

        normalised = [self._normalise(s) for s in samples]
        chs_threshold = 0.3

        good = [s for s in normalised if s["chs"] <= chs_threshold and s["answer"]]
        logger.info(
            f"Filtered: {len(good)}/{len(normalised)} samples with CHS ≤ {chs_threshold}"
        )

        if not good:
            logger.error(
                "No samples pass the CHS filter! Try:\n"
                "  - Collect more diverse training data\n"
                "  - Use metric_loss method instead (uses all samples)"
            )
            return

        self._run_sft(good, tag="rejection")

    # ══════════════════════════════════════════════════════════════════════════
    #  Metric-Weighted SFT
    # ══════════════════════════════════════════════════════════════════════════
    def _train_metric_weighted_sft(self, samples: list) -> None:
        """
        SFT where good answers (low CHS) are up-weighted via duplication.
        Weight = max(1, round((1 - CHS) * 3))
        """
        logger.info("=== Metric-Weighted SFT ===")

        normalised = [self._normalise(s) for s in samples]
        valid = [s for s in normalised if s["answer"]]

        if not valid:
            logger.error("No valid samples for training!")
            return

        # Duplicate good samples to approximate quality weighting
        weighted = []
        for s in valid:
            copies = max(1, round((1 - s["chs"]) * 3))
            weighted.extend([s] * copies)

        logger.info(
            f"Expanded {len(valid)} → {len(weighted)} samples (quality-weighted)"
        )

        self._run_sft(weighted, tag="metric_weighted")

    # ── Shared SFT implementation ─────────────────────────────────────────────
    def _run_sft(self, samples: List[dict], tag: str = "sft") -> None:
        """Run supervised fine-tuning on a list of normalised samples."""
        texts = []
        for s in samples:
            prompt = s.get("prompt") or s["query"]
            texts.append(f"{prompt}\n{s['answer']}")

        model = self._load_model()
        model = self._apply_lora(model)

        dataset = Dataset.from_dict({"text": texts})

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_special_tokens_mask=True,
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        args = TrainingArguments(
            output_dir=str(Path(self.output_dir) / "checkpoints"),
            per_device_train_batch_size=self.train_cfg.get(
                "per_device_train_batch_size", 2
            ),
            gradient_accumulation_steps=self.train_cfg.get(
                "gradient_accumulation_steps", 8
            ),
            num_train_epochs=self.train_cfg.get("num_epochs", 3),
            learning_rate=self.train_cfg.get("learning_rate", 2e-4),
            lr_scheduler_type="cosine",
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.05),
            logging_steps=1,
            save_steps=100,
            save_total_limit=2,
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            dataloader_num_workers=0,
            report_to="none",
            eval_strategy="no",
        )

        trainer = None
        try:
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized,
                data_collator=collator,
            )
            logger.info(f"Starting {tag} SFT training...")
            trainer.train()
            self._save(model)
            logger.info(f"✓ {tag} SFT completed")

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM! Reduce per_device_train_batch_size in config.")
            raise
        finally:
            self._cleanup(trainer, model)