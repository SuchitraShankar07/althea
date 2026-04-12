import gc
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import json
from pathlib import Path
from loguru import logger

@dataclass
class TrainingSample:
    """Strict training schema used across collection, loading, and fine-tuning."""

    query: str
    response: str
    contexts: List[str] = field(default_factory=list)
    chs: float = 0.0
    scr: float = 0.0
    tve: float = 0.0
    cr: float = 0.0
    cdee: float = 0.0
    retrieval_score: float = 0.0
    retrieval_conflict: float = 0.0
    overgeneralization: float = 0.0
    outdated_information: float = 0.0
    synthesis_error: float = 0.0
    overall_hallucination_score: float = 0.0
    hallucination_confidence: float = 0.0

    def __post_init__(self) -> None:
        self.query = str(self.query or "").strip()
        self.response = str(self.response or "").strip()
        self.contexts = [str(c).strip() for c in (self.contexts or []) if str(c).strip()]
        self.chs = float(self.chs)
        self.scr = float(self.scr)
        self.tve = float(self.tve)
        self.cr = float(self.cr)
        self.cdee = float(self.cdee)
        self.retrieval_score = float(self.retrieval_score)
        self.retrieval_conflict = float(self.retrieval_conflict)
        self.overgeneralization = float(self.overgeneralization)
        self.outdated_information = float(self.outdated_information)
        self.synthesis_error = float(self.synthesis_error)
        self.overall_hallucination_score = float(self.overall_hallucination_score)
        self.hallucination_confidence = float(self.hallucination_confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "contexts": self.contexts,
            "chs": self.chs,
            "scr": self.scr,
            "tve": self.tve,
            "cr": self.cr,
            "cdee": self.cdee,
            "retrieval_score": self.retrieval_score,
            "retrieval_conflict": self.retrieval_conflict,
            "overgeneralization": self.overgeneralization,
            "outdated_information": self.outdated_information,
            "synthesis_error": self.synthesis_error,
            "overall_hallucination_score": self.overall_hallucination_score,
            "hallucination_confidence": self.hallucination_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict):
        expected = {
            "query", "response", "contexts", "chs", "scr", "tve", "cr", "cdee", "retrieval_score",
            "retrieval_conflict", "overgeneralization", "outdated_information", "synthesis_error",
            "overall_hallucination_score", "hallucination_confidence"
        }
        unexpected = set(data.keys()) - expected
        if unexpected:
            raise ValueError(f"Unexpected keys in training sample: {sorted(unexpected)}")
        return cls(**data)


def _required_vram_gb(model_name: str, load_in_4bit: bool = True) -> float:
    name = model_name.lower()
    if any(k in name for k in ["70b", "34b"]):
        return 48.0 if load_in_4bit else 80.0
    if any(k in name for k in ["13b"]):
        return 16.0 if load_in_4bit else 28.0
    if any(k in name for k in ["8b", "7b", "mistral", "llama-2", "llama-3"]):
        return 10.0 if load_in_4bit else 16.0
    if any(k in name for k in ["3b", "2.7b", "2b", "1.5b", "phi-3.5-mini"]):
        return 6.0 if load_in_4bit else 10.0
    return 4.0 if load_in_4bit else 8.0

def _load_base_model_for_training(model_name: str, cache_dir: str):
    """Load model with aggressive memory optimization for training"""
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
    # Extremely aggressive quantization for small GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    logger.info(f"Loading {model_name} with 4-bit quantization for training")
    
    # Get available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        free_memory = gpu_memory - allocated
        logger.info(f"GPU: {free_memory:.1f}GB free of {gpu_memory:.1f}GB total")
        required_gb = _required_vram_gb(model_name=model_name, load_in_4bit=True)
        if free_memory < required_gb:
            raise RuntimeError(
                f"Insufficient GPU memory for training {model_name}. "
                f"Required ~{required_gb:.1f}GB free, found {free_memory:.1f}GB free. "
                "Use a smaller model or reduce memory pressure."
            )
    else:
        raise RuntimeError("CUDA GPU is required for QLoRA training.")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    return model

class QLoRATrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct", 
                 cache_dir: str = "cache",
                 output_dir: str = "./outputs/qlora_model",
                 checkpoint_dir: str = "./outputs/qlora_checkpoints",
                 dpo_beta: float = 0.1):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.dpo_beta = float(dpo_beta)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Don't load model in __init__ - load only when needed
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _clear_gpu_memory(self):
        """Aggressive GPU memory cleanup"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        gc.collect()
        
        # Print memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def train_dpo(self, samples):
        logger.info("=== DPO Training ===")
        
        # Clear memory before loading training model
        self._clear_gpu_memory()
        
        # Load model for training
        model = _load_base_model_for_training(self.model_name, self.cache_dir)
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration - select targets by model architecture
        base_model = model
        default_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        gpt2_targets = ["c_attn", "c_proj"]

        try_targets = [default_targets, gpt2_targets]
        model = None
        last_error = None

        for targets in try_targets:
            try:
                lora_config = LoraConfig(
                    r=8,  # Very small rank for memory savings
                    lora_alpha=16,
                    target_modules=targets,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(base_model, lora_config)
                logger.info(f"Using LoRA target modules: {targets}")
                break
            except ValueError as e:
                last_error = e
                logger.warning(f"LoRA targets {targets} failed: {e}")

        if model is None:
            raise ValueError(f"Unable to apply LoRA adapters for model {self.model_name}: {last_error}")
        
        # Convert samples to DPO format
        train_dataset = self._prepare_dpo_dataset(samples)
        
        # Check if we have data to train on
        if len(train_dataset) == 0:
            logger.error("No training samples prepared! Cannot proceed with training.")
            return
        
        def _filter_kwargs(callable_obj, kwargs):
            sig = inspect.signature(callable_obj)
            allowed = set(sig.parameters.keys())
            return {k: v for k, v in kwargs.items() if k in allowed}

        dpo_kwargs = {
            "output_dir": self.checkpoint_dir,
            "per_device_train_batch_size": 1,  # Minimum batch size
            "gradient_accumulation_steps": 2,  # Smaller accumulation
            "num_train_epochs": 1,
            "learning_rate": 1e-5,  # Lower learning rate
            "lr_scheduler_type": "linear",
            "warmup_steps": 5,
            "logging_steps": 1,
            "save_steps": 100,
            "save_total_limit": 1,  # Keep only one checkpoint
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "gradient_checkpointing": True,
            "fp16": False,
            "bf16": False,
            "dataloader_num_workers": 0,  # No multiprocessing
            "report_to": "none",
            # Memory optimizations
            "max_grad_norm": 0.5,
            "eval_strategy": "no",  # Skip evaluation to save memory
            # DPO specific parameters
            "beta": self.dpo_beta,
            "max_length": 256,
            "max_prompt_length": 128,
        }
        training_args = DPOConfig(**_filter_kwargs(DPOConfig.__init__, dpo_kwargs))
        
        # Initialize DPO trainer
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "tokenizer": self.tokenizer,
            "processing_class": self.tokenizer,
        }
        trainer = DPOTrainer(**_filter_kwargs(DPOTrainer.__init__, trainer_kwargs))
        
        logger.info("Starting DPO training...")
        try:
            trainer.train()
            
            # Save the model
            model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info("DPO training completed!")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during training: {e}")
            logger.info("Try reducing batch size or sequence length further")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.info("Trying fallback with minimal DPO config...")
            
            # Try with even more aggressive memory settings
            try:
                fallback_kwargs = {
                    "output_dir": self.checkpoint_dir,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "num_train_epochs": 1,
                    "learning_rate": 5e-6,
                    "warmup_steps": 2,
                    "logging_steps": 1,
                    "save_steps": 50,
                    "save_total_limit": 1,
                    "fp16": False,
                    "bf16": False,
                    "gradient_checkpointing": True,
                    "dataloader_num_workers": 0,
                    "report_to": "none",
                    "eval_strategy": "no",
                    "beta": self.dpo_beta,
                    "max_length": 128,  # Even shorter
                    "max_prompt_length": 64,  # Shorter prompt length
                }
                training_args_fallback = DPOConfig(**_filter_kwargs(DPOConfig.__init__, fallback_kwargs))
                
                fallback_trainer_kwargs = {
                    "model": model,
                    "args": training_args_fallback,
                    "train_dataset": train_dataset,
                    "tokenizer": self.tokenizer,
                    "processing_class": self.tokenizer,
                }
                trainer = DPOTrainer(**_filter_kwargs(DPOTrainer.__init__, fallback_trainer_kwargs))
                
                trainer.train()
                
                # Save the model
                model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                
                logger.info("DPO training completed with fallback settings!")
                
            except Exception as e2:
                logger.error(f"Both DPO training attempts failed: {e2}")
                # Let's try a final simpler approach without DPO
                logger.info("Attempting basic LoRA training without DPO...")
                
                try:
                    # Simple training without DPO - just basic language modeling
                    from transformers import Trainer, DataCollatorForLanguageModeling
                    
                    # Prepare data for language modeling
                    def tokenize_function(examples):
                        # Combine prompt and chosen response
                        texts = [f"{ex['prompt']} {ex['chosen']}" for ex in examples]
                        return self.tokenizer(texts, truncation=True, padding=True, max_length=128)
                    
                    # Convert dataset for language modeling
                    raw_data = [{"text": f"{item['prompt']} {item['chosen']}"} for item in samples]
                    lm_dataset = Dataset.from_list(raw_data)
                    tokenized_dataset = lm_dataset.map(
                        lambda x: self.tokenizer(x["text"], truncation=True, padding=True, max_length=128),
                        batched=True
                    )
                    
                    # Data collator
                    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer,
                        mlm=False,
                    )
                    
                    # Basic training arguments
                    basic_args = TrainingArguments(
                        output_dir=self.checkpoint_dir,
                        per_device_train_batch_size=1,
                        num_train_epochs=1,
                        learning_rate=5e-6,
                        logging_steps=1,
                        save_steps=50,
                        save_total_limit=1,
                        fp16=False,
                        bf16=False,
                        gradient_checkpointing=True,
                        dataloader_num_workers=0,
                        report_to="none",
                        eval_strategy="no",
                    )
                    
                    # Basic trainer
                    basic_trainer = Trainer(
                        model=model,
                        args=basic_args,
                        train_dataset=tokenized_dataset,
                        data_collator=data_collator,
                    )
                    
                    basic_trainer.train()
                    
                    # Save the model
                    model.save_pretrained(self.output_dir)
                    self.tokenizer.save_pretrained(self.output_dir)
                    
                    logger.info("Basic LoRA training completed successfully!")
                    
                except Exception as e3:
                    logger.error(f"All training methods failed: {e3}")
                    raise
        finally:
            # Clean up
            if 'trainer' in locals():
                del trainer
            del model
            self._clear_gpu_memory()

    def _prepare_dpo_dataset(self, samples):
        """Convert preference pairs to DPO format using strict keys."""
        dpo_data = []
        
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("DPO samples must be dictionaries with prompt/chosen/rejected keys")

            prompt = str(sample.get("prompt", "")).strip()
            chosen = str(sample.get("chosen", "")).strip()
            rejected = str(sample.get("rejected", "")).strip()

            if not prompt or not chosen or not rejected:
                continue
            if chosen == rejected:
                continue
            
            # Truncate long prompts to save memory
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            if len(chosen) > 300:
                chosen = chosen[:300] + "..."
            if len(rejected) > 300:
                rejected = rejected[:300] + "..."
            
            dpo_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })
        
        logger.info(f"Prepared {len(dpo_data)} DPO training samples")
        return Dataset.from_list(dpo_data)

    def train(self, samples):
        """Main training method"""
        self.train_dpo(samples)

# Add the MetricGuidedQLoRATrainer class that's referenced in train.py
class MetricGuidedQLoRATrainer:
    def __init__(self, config):
        self.config = config
        training_cfg = config.get('training', {})
        self.min_scr = float(training_cfg.get('min_scr_for_positive', 0.75))
        self.max_tve = float(training_cfg.get('max_tve_for_positive', 0.20))
        self.max_cr = float(training_cfg.get('max_cr_for_positive', 0.20))
        self.max_chs = float(training_cfg.get('max_chs_for_positive', 0.40))
        self.min_score_gap = float(training_cfg.get('min_score_gap', 0.20))
        model_name = config.get('generation', {}).get('model_name', 'microsoft/Phi-3.5-mini-instruct')
            
        cache_dir = config.get('paths', {}).get('cache_dir', 'cache')
        output_dir = training_cfg.get('output_dir', './outputs/qlora_model')
        checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', './outputs/qlora_checkpoints')
        dpo_beta = float(training_cfg.get('dpo_beta', 0.1))
        
        self.trainer = QLoRATrainer(
            model_name=model_name,
            cache_dir=cache_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            dpo_beta=dpo_beta,
        )

    @staticmethod
    def _abstention_response(query: str) -> str:
        return (
            "The provided context does not contain enough verifiable evidence to answer this "
            f"question reliably: {query}."
        )

    @staticmethod
    def _unsupported_temporal_negative(query: str) -> str:
        return (
            f"As of today, there is a latest update that fully answers '{query}', "
            "even though no supporting evidence is provided."
        )
        
    def train(self, samples):
        """Train using the specified method"""
        method = self.config.get('training', {}).get('method', 'dpo')
        
        if method == 'dpo':
            self._train_dpo(samples)
        elif method == 'rejection':
            self._train_rejection(samples)
        elif method == 'metric_loss':
            self._train_metric_loss(samples)
        else:
            raise ValueError(f"Unknown training method: {method}")
    
    def _normalize_sample(self, sample):
        """Convert sample to strict query/response/contexts format."""
        if isinstance(sample, dict):
            if "prompt" in sample or "answer" in sample:
                raise ValueError("Invalid sample schema: use query/response/contexts only.")
            return {
                'query': sample.get('query', ''),
                'response': sample.get('response', ''),
                'chs': float(sample.get('chs', 0.5)),
                'scr': float(sample.get('scr', 0.0)),
                'tve': float(sample.get('tve', 0.0)),
                'cr': float(sample.get('cr', 0.0)),
                'cdee': float(sample.get('cdee', 0.0)),
                'contexts': sample.get('contexts', [])
            }
        else:
            if hasattr(sample, "prompt") or hasattr(sample, "answer"):
                raise ValueError("Invalid sample schema object: use query/response/contexts only.")

            result = {
                'query': str(getattr(sample, 'query', '') or ''),
                'response': str(getattr(sample, 'response', '') or ''),
                'chs': float(getattr(sample, 'chs', 0.5)),
                'scr': float(getattr(sample, 'scr', 0.0)),
                'tve': float(getattr(sample, 'tve', 0.0)),
                'cr': float(getattr(sample, 'cr', 0.0)),
                'cdee': float(getattr(sample, 'cdee', 0.0)),
                'contexts': list(getattr(sample, 'contexts', []) or []),
            }
            return result
    
    def _train_dpo(self, samples):
        """DPO training with real model-output preference pairs."""
        logger.info("Training with DPO method")
        dpo_samples = self._build_real_preference_pairs(samples)
        logger.info(f"Created {len(dpo_samples)} DPO pairs from {len(samples)} candidate samples")
        self.trainer.train_dpo(dpo_samples)

    def _quality_score(self, normalized: dict) -> float:
        # Lower CHS/TVE/CR and higher SCR indicate better responses.
        return (
            normalized['scr']
            - normalized['chs']
            - normalized['tve']
            - normalized['cr']
            - 0.25 * normalized.get('cdee', 0.0)
        )

    def _build_real_preference_pairs(self, samples):
        by_query = {}
        for sample in samples:
            normalized = self._normalize_sample(sample)
            query = normalized['query'].strip()
            response = normalized['response'].strip()
            if not query or not response:
                continue
            by_query.setdefault(query, []).append(normalized)

        dpo_samples = []
        for query, candidates in by_query.items():
            unique = []
            seen_responses = set()
            for c in candidates:
                r = c['response']
                if r in seen_responses:
                    continue
                seen_responses.add(r)
                unique.append(c)

            if len(unique) < 2:
                continue

            ranked = sorted(unique, key=self._quality_score, reverse=True)
            chosen = ranked[0]
            rejected = ranked[-1]

            quality_gap = self._quality_score(chosen) - self._quality_score(rejected)
            if quality_gap < self.min_score_gap:
                continue

            assert self._quality_score(chosen) > self._quality_score(rejected), (
                "DPO preference inversion detected: chosen quality must exceed rejected quality"
            )

            dpo_samples.append(
                {
                    "prompt": query,
                    "chosen": chosen['response'],
                    "rejected": rejected['response'],
                }
            )

        if not dpo_samples:
            raise ValueError(
                "No valid DPO pairs were created from real outputs. "
                "Collect at least two response candidates per query with measurable quality differences."
            )
        return dpo_samples
    
    def _train_rejection(self, samples):
        """Rejection sampling based training"""
        logger.info("Training with rejection sampling method")
        dpo_samples = self._build_real_preference_pairs(samples)
        logger.info(f"Created {len(dpo_samples)} rejection sampling pairs")
        self.trainer.train_dpo(dpo_samples)
    
    def _train_metric_loss(self, samples):
        """Metric-guided loss training"""
        logger.info("Training with metric loss method")

        weighted_samples = self._build_real_preference_pairs(samples)
        logger.info(f"Created {len(weighted_samples)} metric-guided samples")
        self.trainer.train_dpo(weighted_samples)