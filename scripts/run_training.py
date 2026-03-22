"""run_training.py – Fine-tune the generation model with QLoRA.

Usage
-----
    python scripts/run_training.py \
        --train_file data/train.json \
        --eval_file  data/eval.json \
        --config     config/config.yaml
"""

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Althea.")
    parser.add_argument(
        "--train_file",
        required=True,
        help="JSON file with training examples (list of dicts with prompt/chosen/rejected keys).",
    )
    parser.add_argument(
        "--eval_file",
        default=None,
        help="Optional JSON file with evaluation examples.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def main():
    import yaml

    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    training_cfg = config.get("training", {}).get("qlora", {})

    from src.training import QLoRATrainer

    trainer = QLoRATrainer(
        base_model=training_cfg.get("base_model"),
        output_dir=training_cfg.get("output_dir", "data/checkpoints"),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        lora_r=training_cfg.get("lora_r", 16),
        lora_alpha=training_cfg.get("lora_alpha", 32),
        lora_dropout=training_cfg.get("lora_dropout", 0.05),
    )

    with open(args.train_file, "r", encoding="utf-8") as fh:
        train_data = json.load(fh)

    eval_data = None
    if args.eval_file:
        with open(args.eval_file, "r", encoding="utf-8") as fh:
            eval_data = json.load(fh)

    print(f"Starting QLoRA fine-tuning on {len(train_data)} examples …")
    trainer.train(train_data=train_data, eval_data=eval_data)
    print("Training complete.")


if __name__ == "__main__":
    main()
