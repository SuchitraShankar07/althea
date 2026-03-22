"""
generation/generator.py
Wraps an instruction-tuned LLM (optionally with QLoRA adapters) for
RAG-style generation.  Supports 4-bit quantisation via BitsAndBytes.
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
    """

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

        # ── Quantisation config ───────────────────────────────
        bnb_cfg = None
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
            device_map=device_map,
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
            cache_dir=cfg.get("cache_dir"),
        )