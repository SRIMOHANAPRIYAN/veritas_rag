"""Train cross-encoder reranker with MS MARCO mixing.

Standalone script for Colab A100. No imports from local state.
All inputs via CLI args + configs/training_config.yaml.

v2 changes:
- LR bug fixed: explicitly passed to optimizer_params
- Epochs: 1 (protect zero-shot priors)
- MS MARCO mixing: ~90:10 MS MARCO:domain triplets
"""

import sys
import json
import random
from pathlib import Path

import os
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb explicitly

from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)
from torch.utils.data import DataLoader

import mlflow
import hydra
from loguru import logger
from omegaconf import OmegaConf, DictConfig


def load_domain_triplets(data_dir: Path) -> list[InputExample]:
    """Load domain triplets from JSONL file."""
    triplets_path = data_dir / "domain_triplets.jsonl"
    samples: list[InputExample] = []
    if not triplets_path.exists():
        logger.warning(f"Domain triplets not found at {triplets_path}")
        return samples

    with open(triplets_path, "r") as f:
        for line in f:
            data = json.loads(line)
            samples.append(
                InputExample(texts=[data["query"], data["positive"]], label=1.0)
            )
            samples.append(
                InputExample(texts=[data["query"], data["negative"]], label=0.0)
            )
    logger.info(f"Loaded {len(samples)} domain samples from {triplets_path}")
    return samples


def load_msmarco_samples(sample_size: int, seed: int = 42) -> list[InputExample]:
    """Download and sample MS MARCO triplets for mixing.

    Uses sentence-transformers/msmarco-msmarco-distilbert-base-tas-b triplet split.
    """
    logger.info(f"Downloading MS MARCO training triples (sampling {sample_size})...")
    ds = load_dataset(
        "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
        "triplet",
        split="train",
    )
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))

    samples: list[InputExample] = []
    for idx in indices:
        row = ds[idx]
        samples.append(InputExample(texts=[row["query"], row["positive"]], label=1.0))
        samples.append(InputExample(texts=[row["query"], row["negative"]], label=0.0))
    logger.info(f"Loaded {len(samples)} MS MARCO mixing samples")
    return samples


@hydra.main(version_base=None, config_path="../configs", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Train the cross-encoder reranker."""
    
    dry_run = cfg.get("dry_run", False)

    out_dir = Path(cfg.reranker.output_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("VeritasRAG_Reranker")

    model_name: str = cfg.reranker.base_model
    lr: float = float(cfg.reranker.learning_rate)
    epochs: int = int(cfg.reranker.epochs)
    batch_size: int = int(cfg.reranker.batch_size)
    msmarco_size: int = int(cfg.reranker.msmarco_sample_size)
    msmarco_ratio: float = float(cfg.reranker.msmarco_mix_ratio)

    logger.info(f"Model: {model_name}")
    logger.info(f"LR: {lr}, Epochs: {epochs}, Batch: {batch_size}")
    logger.info(f"MS MARCO mix: {msmarco_ratio:.0%} of {msmarco_size} samples")

    # Load model
    device = "cpu" if dry_run else None
    model = CrossEncoder(model_name, num_labels=1, device=device)

    # Load data
    data_dir = cfg.get("data_dir", "data/training")
    domain_samples = load_domain_triplets(Path(data_dir))
    msmarco_samples = load_msmarco_samples(msmarco_size)

    # Mix
    all_samples = domain_samples + msmarco_samples
    random.shuffle(all_samples)

    logger.info(
        f"Total training samples: {len(all_samples)} "
        f"(domain={len(domain_samples)}, msmarco={len(msmarco_samples)})"
    )

    # Split 90/10
    split_idx = int(len(all_samples) * 0.9)
    train_data = all_samples[:split_idx]
    eval_data = all_samples[split_idx:]
    
    if dry_run:
        logger.info("DRY RUN: Limiting to 3 samples")
        train_data = train_data[:3]
        eval_data = eval_data[:3]
        epochs = 1

    if not train_data:
        logger.error("No training data. Run generate_training_data.py first.")
        return

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_data, name="mixed-eval"
    )

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "train_samples_total": len(train_data),
                "domain_samples": len(domain_samples),
                "msmarco_dataset": "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
                "msmarco_samples": len(msmarco_samples),
                "msmarco_mix_ratio": msmarco_ratio,
                "version": "v2",
            }
        )

        logger.info("Starting training...")
        # CrossEncoder doesn't support report_to natively, but it doesn't log to WANDB unless explicitly configured.
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=1000 if not dry_run else 1,
            warmup_steps=warmup_steps,
            output_path=str(out_dir) if not dry_run else None,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
            save_best_model=not dry_run,
        )

        if not dry_run:
            logger.info(f"Training complete. Model saved to {out_dir}")
            model.save(str(out_dir))


if __name__ == "__main__":
    main()
