import gc
import inspect
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

# Add the missing TrainingSample class
class TrainingSample:
    """Simple data structure to hold training samples with quality metrics"""
    def __init__(self, query: str, response: str = "", contexts: list = None, 
                 chs: float = 0.0, retrieval_score: float = 0.0, **kwargs):
        self.query = query
        self.response = response
        self.contexts = contexts or []
        self.chs = chs  # Context-Response Harmony Score
        self.retrieval_score = retrieval_score
        
        # Store any additional metrics
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        result = {
            'query': self.query,
            'response': self.response,
            'contexts': self.contexts,
            'chs': self.chs,
            'retrieval_score': self.retrieval_score
        }
        
        # Add any additional attributes
        for attr in dir(self):
            if not attr.startswith('_') and attr not in result:
                value = getattr(self, attr)
                if not callable(value):
                    result[attr] = value
                    
        return result
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)

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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # More aggressive memory constraints for T4
        max_memory={0: "10GB"}  # Conservative limit for T4
    )
    
    return model

class QLoRATrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct", 
                 cache_dir: str = "cache"):
        # Use a much smaller model for T4 GPU
        if "mistral" in model_name.lower() or "7b" in model_name.lower():
            logger.warning(f"Model {model_name} may be too large for T4 GPU, switching to Phi-3.5-mini")
            model_name = "microsoft/Phi-3.5-mini-instruct"
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        
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
            "output_dir": "./outputs/qlora_checkpoints",
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
            "beta": 0.1,
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
            model.save_pretrained("./outputs/qlora_model")
            self.tokenizer.save_pretrained("./outputs/qlora_model")
            
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
                    "output_dir": "./outputs/qlora_checkpoints",
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
                    "beta": 0.1,
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
                model.save_pretrained("./outputs/qlora_model")
                self.tokenizer.save_pretrained("./outputs/qlora_model")
                
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
                        output_dir="./outputs/qlora_checkpoints",
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
                    model.save_pretrained("./outputs/qlora_model")
                    self.tokenizer.save_pretrained("./outputs/qlora_model")
                    
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
        """Convert training samples to DPO format"""
        dpo_data = []
        
        for sample in samples:
            # Handle both dict and object formats
            if isinstance(sample, dict):
                prompt = sample.get('prompt', sample.get('query', ''))
                chosen = sample.get('chosen', sample.get('response', ''))
                rejected = sample.get('rejected', '')
            else:
                prompt = getattr(sample, 'prompt', getattr(sample, 'query', ''))
                chosen = getattr(sample, 'chosen', getattr(sample, 'response', ''))
                rejected = getattr(sample, 'rejected', '')
            
            # Skip empty samples
            if not prompt or not chosen:
                continue
                
            # Generate rejected response if not provided
            if not rejected:
                rejected = "I don't have enough information to answer this question properly."
            
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
        # Override with smaller model for T4
        model_name = config.get('generation', {}).get('model_name', 'microsoft/Phi-3.5-mini-instruct')
        if "mistral" in model_name.lower() or "7b" in model_name.lower():
            logger.warning(f"Overriding large model {model_name} with Phi-3.5-mini for T4 GPU")
            model_name = "microsoft/Phi-3.5-mini-instruct"
            
        cache_dir = config.get('paths', {}).get('cache_dir', 'cache')
        
        self.trainer = QLoRATrainer(model_name=model_name, cache_dir=cache_dir)

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
        """Convert sample to consistent format - handles both dict and object formats"""
        if isinstance(sample, dict):
            return {
                'query': sample.get('query', ''),
                'response': sample.get('response', sample.get('answer', '')),
                'chs': float(sample.get('chs', 0.5)),
                'scr': float(sample.get('scr', 0.0)),
                'tve': float(sample.get('tve', 0.0)),
                'cr': float(sample.get('cr', 0.0)),
                'cdee': float(sample.get('cdee', 0.0)),
                'contexts': sample.get('contexts', [])
            }
        else:
            # For TrainingSample objects, inspect all attributes to find the data
            sample_dict = {}
            for attr in dir(sample):
                if not attr.startswith('_') and not callable(getattr(sample, attr)):
                    sample_dict[attr] = getattr(sample, attr)
            
            logger.debug(f"Sample attributes: {sample_dict}")
            
            # Extract query and response from available attributes
            query = (sample_dict.get('query') or 
                    sample_dict.get('prompt') or 
                    sample_dict.get('question') or '')
            
            response = (sample_dict.get('response') or 
                       sample_dict.get('answer') or 
                       sample_dict.get('text') or '')
            
            chs = float(sample_dict.get('chs', 0.5))
            scr = float(sample_dict.get('scr', 0.0))
            tve = float(sample_dict.get('tve', 0.0))
            cr = float(sample_dict.get('cr', 0.0))
            cdee = float(sample_dict.get('cdee', 0.0))
            contexts = sample_dict.get('contexts', [])
            
            result = {
                'query': str(query) if query else '',
                'response': str(response) if response else '',
                'chs': chs,
                'scr': scr,
                'tve': tve,
                'cr': cr,
                'cdee': cdee,
                'contexts': contexts if isinstance(contexts, list) else []
            }
            
            logger.debug(
                "Normalized sample: "
                f"query='{result['query'][:50]}...', "
                f"response='{result['response'][:50]}...', "
                f"chs={result['chs']}, scr={result['scr']}, tve={result['tve']}, cr={result['cr']}"
            )
            return result
    
    def _train_dpo(self, samples):
        """DPO training with preference pairs"""
        logger.info("Training with DPO method")
        logger.info(f"Input samples type: {type(samples[0]) if samples else 'empty'}")
        
        # Debug: log first sample's attributes
        if samples:
            first_sample = samples[0]
            sample_attrs = {}
            for attr in dir(first_sample):
                if not attr.startswith('_') and not callable(getattr(first_sample, attr)):
                    sample_attrs[attr] = getattr(first_sample, attr)
            logger.info(f"First sample attributes: {sample_attrs}")
        
        # Convert samples to DPO format
        dpo_samples = []
        for i, sample in enumerate(samples):
            normalized = self._normalize_sample(sample)
            
            query = normalized['query']
            response = normalized['response']
            chs = normalized['chs']
            scr = normalized['scr']
            tve = normalized['tve']
            cr = normalized['cr']
            
            logger.debug(
                f"Sample {i}: query='{query[:50]}...' response='{response[:50]}...' "
                f"chs={chs} scr={scr} tve={tve} cr={cr}"
            )
            
            # Skip empty queries/responses
            if not query or not response:
                logger.warning(f"Skipping sample {i}: empty query='{query}' or response='{response}'")
                continue

            # Metric-aware preference shaping for stricter grounding and temporal validity.
            is_grounded = scr >= self.min_scr
            is_temporally_safe = tve <= self.max_tve
            is_low_conflict = cr <= self.max_cr

            if is_grounded and is_temporally_safe and is_low_conflict:
                dpo_samples.append({
                    'prompt': query,
                    'chosen': response,
                    'rejected': self._unsupported_temporal_negative(query),
                })
            else:
                dpo_samples.append({
                    'prompt': query,
                    'chosen': self._abstention_response(query),
                    'rejected': response,
                })
        
        logger.info(f"Created {len(dpo_samples)} DPO training pairs from {len(samples)} input samples")
        
        if len(dpo_samples) == 0:
            logger.error("No valid DPO samples created! Check your input data format.")
            # Log first few samples for debugging
            for i, sample in enumerate(samples[:3]):
                logger.error(f"Sample {i}: {sample}")
                if hasattr(sample, '__dict__'):
                    logger.error(f"  Dict: {sample.__dict__}")
            return
            
        self.trainer.train_dpo(dpo_samples)
    
    def _train_rejection(self, samples):
        """Rejection sampling based training"""
        logger.info("Training with rejection sampling method")
        
        # Filter samples based on quality scores
        high_quality_samples = []
        low_quality_samples = []
        
        for s in samples:
            normalized = self._normalize_sample(s)
            chs = normalized['chs']
            
            if chs > 0.6:
                high_quality_samples.append(normalized)
            elif chs < 0.4:
                low_quality_samples.append(normalized)
        
        logger.info(f"High quality samples: {len(high_quality_samples)}")
        logger.info(f"Low quality samples: {len(low_quality_samples)}")
        
        # Convert to DPO format for training
        dpo_samples = []
        for good, bad in zip(high_quality_samples, low_quality_samples):
            good_query = good['query']
            good_response = good['response']
            bad_response = bad['response']
            
            if good_query and good_response and bad_response:
                dpo_samples.append({
                    'prompt': good_query,
                    'chosen': good_response,
                    'rejected': bad_response
                })
        
        logger.info(f"Created {len(dpo_samples)} rejection sampling pairs")
        self.trainer.train_dpo(dpo_samples)
    
    def _train_metric_loss(self, samples):
        """Metric-guided loss training"""
        logger.info("Training with metric loss method")
        
        # Use all samples with their quality scores as weights
        weighted_samples = []
        for sample in samples:
            normalized = self._normalize_sample(sample)
            
            query = normalized['query']
            response = normalized['response']
            chs = normalized['chs']
            
            if not query or not response:
                continue
                
            # Higher CHS scores get preference
            if chs > 0.5:
                weighted_samples.append({
                    'prompt': query,
                    'chosen': response,
                    'rejected': f"Low quality response for: {query}"
                })
        
        logger.info(f"Created {len(weighted_samples)} metric-guided samples")
        self.trainer.train_dpo(weighted_samples)