"""
generation/generator.py
Wraps an instruction-tuned LLM (optionally with QLoRA adapters) for
RAG-style generation. Supports 4-bit quantisation via BitsAndBytes.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful and factual assistant. "
    "Answer the question using ONLY the provided context. "
    "If the context does not contain enough information, say so explicitly."
)

def build_prompt(query: str, documents: List[dict]) -> str:
    """Format retrieved docs + query into an instruction prompt."""
    context_blocks = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("metadata", {}).get("title", f"Document {i}")
        context_blocks.append(f"[{i}] {title}\n{doc['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{query}\n\n"
        f"### Answer\n"
    )


# ── Generator class ───────────────────────────────────────────────────────────
class RAGGenerator:
    """
    Loads an LLM with optional 4-bit quantisation and LoRA adapters,
    then generates answers from (query, retrieved-docs) pairs.

    CPU guard: models with ≥7B parameters (mistral, llama, 7b, 8b, 13b, 70b)
    raise RuntimeError immediately if CUDA is unavailable.  Use --mock-generate,
    demo.py, TinyLlama, or a GPU machine to avoid this.
    """

    # Keywords indicating a model too large for CPU inference
    _LARGE_MODEL_KEYWORDS = [
        "7b", "8b", "13b", "70b", "mistral", "llama-2", "llama-3",
    ]

    @classmethod
    def _is_large_model(cls, model_name: str) -> bool:
        """Return True if the model name suggests a ≥7B parameter model."""
        name_lower = model_name.lower()
        return any(kw in name_lower for kw in cls._LARGE_MODEL_KEYWORDS)

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.device_map = device_map

        has_cuda = torch.cuda.is_available()

        # ── CPU guard (FIRST check — fail fast before any loading) ────────
        if not has_cuda and self._is_large_model(model_name):
            raise RuntimeError(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║         CUDA GPU REQUIRED FOR THIS LARGE MODEL               ║\n"
                "╠══════════════════════════════════════════════════════════════╣\n"
                f"║  Model : {model_name:<54} ║\n"
                "║  No CUDA GPU detected. This model cannot run on CPU.         ║\n"
                "║                                                              ║\n"
                "║  Options:                                                    ║\n"
                "║  1. Run the zero-setup demo:  python scripts/demo.py         ║\n"
                "║  2. Use the mock flag:                                       ║\n"
                "║        run_inference.py --mock-generate --query '...'        ║\n"
                "║  3. Switch to a CPU-friendly model (edit config.yaml):       ║\n"
                "║        model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0       ║\n"
                "║  4. Run on a machine with a CUDA GPU (≥16 GB VRAM)           ║\n"
                "╚══════════════════════════════════════════════════════════════╝"
            )

        # ── Quantisation config ───────────────────────────────────────────
        bnb_cfg = None
        if load_in_4bit and not has_cuda:
            logger.warning(
                "4-bit loading requested but CUDA is unavailable; "
                "falling back to full-precision loading."
            )
            load_in_4bit = False
            self.device_map = None
        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit NF4 quantisation (QLoRA mode)")

        # ── Tokenizer ─────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Base model ────────────────────────────────────────
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=self.device_map,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model.eval()

        # ── LoRA adapters (post-fine-tune) ────────────────────
        if adapter_path:
            self._load_adapter(adapter_path)

    def _load_adapter(self, adapter_path: str) -> None:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ── Inference ─────────────────────────────────────────────
    def generate(self, query: str, documents: List[dict]) -> str:
        prompt = build_prompt(query, documents)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens
        new_tokens = output_ids[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

    def generate_batch(
        self, queries: List[str], docs_list: List[List[dict]]
    ) -> List[str]:
        return [self.generate(q, d) for q, d in zip(queries, docs_list)]

    # ── Factory ───────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: dict) -> "RAGGenerator":
        return cls(
            model_name=cfg["model_name"],
            load_in_4bit=cfg.get("load_in_4bit", True),
            adapter_path=cfg.get("adapter_path"),
            max_new_tokens=cfg.get("max_new_tokens", 512),
            temperature=cfg.get("temperature", 0.1),
            do_sample=cfg.get("do_sample", False),
            device_map=cfg.get("device_map", "auto"),
            cache_dir=cfg.get("cache_dir"),
        )
