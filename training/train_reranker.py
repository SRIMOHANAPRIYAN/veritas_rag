import os
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import mlflow

from loguru import logger
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="training_config")
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=cfg.reranker.output_dir, help="Directory to save the trained model")
    parser.add_argument("--data_dir", type=str, default="data/training", help="Directory containing domain_triplets.jsonl")
    args, _ = parser.parse_known_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow Setup
    mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
    mlflow.set_experiment("VeritasRAG_Reranker")
    
    logger.info("Loading CrossEncoder model...")
    model_name = cfg.reranker.base_model
    model = CrossEncoder(model_name, num_labels=1)
    
    # Prepare data
    logger.info("Preparing training data...")
    train_samples = []
    
    domain_triplets_path = Path(args.data_dir) / "domain_triplets.jsonl"
    if domain_triplets_path.exists():
        with open(domain_triplets_path, "r") as f:
            for line in f:
                data = json.loads(line)
                train_samples.append(InputExample(texts=[data["query"], data["positive"]], label=1.0))
                train_samples.append(InputExample(texts=[data["query"], data["negative"]], label=0.0))
    else:
        logger.warning(f"Domain triplets not found at {domain_triplets_path}")
        
    random.shuffle(train_samples)
    
    # Split into train/eval (90/10)
    split_idx = int(len(train_samples) * 0.9)
    train_data = train_samples[:split_idx]
    eval_data = train_samples[split_idx:]
    
    if not train_data:
        logger.error("No training data available. Run generate_training_data.py first.")
        return
        
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=cfg.reranker.batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_data, name="domain-eval")
    
    epochs = cfg.reranker.epochs
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    with mlflow.start_run():
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "batch_size": cfg.reranker.batch_size,
            "learning_rate": cfg.reranker.learning_rate,
            "train_samples": len(train_data),
        })
        
        logger.info("Starting training...")
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=str(out_dir),
            optimizer_params={'lr': cfg.reranker.learning_rate},
            show_progress_bar=True
        )
        
        logger.info(f"Training complete. Model saved to {out_dir}")
        model.save(str(out_dir))

if __name__ == "__main__":
    main()
