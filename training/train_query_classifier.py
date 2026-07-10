import os
import sys
import json
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import mlflow

from loguru import logger
from omegaconf import OmegaConf


class QueryDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        self.data = []
        self.labels = []
        self.label_map = {"simple": 0, "multi-hop": 1, "comparative": 2}

        with open(data_file, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item["query"])
                self.labels.append(self.label_map[item["label"]])

        self.encodings = tokenizer(
            self.data, truncation=True, padding=True, max_length=max_length
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}


def main():
    cfg = OmegaConf.load("configs/training_config.yaml")
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    dry_run = cfg.get("dry_run", False)

    out_dir = Path(cfg.classifier.output_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("VeritasRAG_QueryClassifier")

    model_name = cfg.classifier.base_model
    logger.info(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "simple", 1: "multi-hop", 2: "comparative"},
        label2id={"simple": 0, "multi-hop": 1, "comparative": 2},
    )

    # Load dataset
    data_dir = cfg.get("data_dir", "data/training")
    data_path = Path(data_dir) / "query_classifier_data_v2.jsonl"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Run generate_training_data.py first.")
        return

    dataset = QueryDataset(data_path, tokenizer)

    # Split train/eval
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size], generator=generator
    )
    
    if dry_run:
        logger.info("DRY RUN: Limiting to 3 samples")
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(3, len(train_dataset))))
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(3, len(eval_dataset))))
        cfg.classifier.epochs = 1

    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch" if not dry_run else "no",
        learning_rate=cfg.classifier.learning_rate,
        per_device_train_batch_size=cfg.classifier.batch_size,
        per_device_eval_batch_size=cfg.classifier.batch_size,
        num_train_epochs=cfg.classifier.epochs,
        weight_decay=0.01,
        load_best_model_at_end=not dry_run,
        metric_for_best_model="f1" if not dry_run else None,
        report_to="none",
        use_cpu=dry_run,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": model_name,
                "epochs": cfg.classifier.epochs,
                "batch_size": cfg.classifier.batch_size,
                "learning_rate": cfg.classifier.learning_rate,
                "train_samples": len(train_dataset),
            }
        )
        logger.info("Starting training...")
        trainer.train()

        if not dry_run:
            logger.info(f"Saving final model to {out_dir}")
            trainer.save_model(str(out_dir))


if __name__ == "__main__":
    main()
