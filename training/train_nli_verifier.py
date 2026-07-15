import os
import argparse
from pathlib import Path
import json

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from loguru import logger
import numpy as np

# Deviation A1: We do NOT train from scratch on MNLI+SNLI.
# We start from 'cross-encoder/nli-deberta-v3-base' which is already trained on MNLI+SNLI,
# and only do a small domain fine-tune to right-size compute and avoid catastrophic forgetting.

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/training/domain_nli.jsonl", help="Path to domain NLI synthetic pairs")
    parser.add_argument("--output_dir", type=str, default="models/nli_verifier", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Run with 3 samples for testing")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    logger.info(f"Using device: {device}")

    # Load Base Model
    model_name = "cross-encoder/nli-deberta-v3-base"
    logger.info(f"Loading tokenizer and model from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The model has 3 labels: 0: CONTRADICTION, 1: ENTAILMENT, 2: NEUTRAL
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Label mapping in cross-encoder/nli-deberta-v3-base:
    # 0 -> contradiction
    # 1 -> entailment
    # 2 -> neutral
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2, "baseless": 2}

    # Load data
    logger.info(f"Loading domain data from {args.data_path}")
    data = []
    if os.path.exists(args.data_path):
        with open(args.data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                label_str = item["label"].lower()
                if label_str in label_map:
                    data.append({
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "label": label_map[label_str]
                    })
    else:
        logger.warning(f"Data file {args.data_path} not found. Creating dummy data for testing/dry-run.")
        data = [
            {"premise": "The sky is blue.", "hypothesis": "The sky has a blue color.", "label": 1},
            {"premise": "The sky is blue.", "hypothesis": "The sky is green.", "label": 0},
            {"premise": "The sky is blue.", "hypothesis": "Cats are mammals.", "label": 2},
        ]
        if not args.dry_run:
            raise FileNotFoundError(f"Data file {args.data_path} missing and not a dry run.")

    if args.dry_run:
        data = data[:3]
        logger.info("DRY RUN enabled: limiting to 3 samples")

    dataset = Dataset.from_list(data)

    def preprocess_function(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=256)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # 90-10 split
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Phase 2 cleanup ticket compliance
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
