import os
import argparse
import json
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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
                
        self.encodings = tokenizer(self.data, truncation=True, padding=True, max_length=max_length)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="models/query_classifier/", help="Directory to save the trained model")
    parser.add_argument("--data_dir", type=str, default="data/training", help="Directory containing query_classifier_data.jsonl")
    args = parser.parse_args()
    
    cfg = OmegaConf.load("configs/training_config.yaml")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow Setup
    mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "VeritasRAG_QueryClassifier"
    
    model_name = cfg.classifier.base_model
    logger.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label={0: "simple", 1: "multi-hop", 2: "comparative"},
        label2id={"simple": 0, "multi-hop": 1, "comparative": 2}
    )
    
    # Load dataset
    data_path = Path(args.data_dir) / "query_classifier_data.jsonl"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Run generate_training_data.py first.")
        return
        
    dataset = QueryDataset(data_path, tokenizer)
    
    # Split train/eval
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=generator)
    
    logger.info(f"Train size: {train_size}, Eval size: {eval_size}")
    
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.classifier.learning_rate,
        per_device_train_batch_size=cfg.classifier.batch_size,
        per_device_eval_batch_size=cfg.classifier.batch_size,
        num_train_epochs=cfg.classifier.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="mlflow",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving final model to {out_dir}")
    trainer.save_model(str(out_dir))

if __name__ == "__main__":
    main()
