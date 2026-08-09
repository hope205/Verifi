import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.data import load_financial_dataset, prepare_dataset
from src.utils.model import load_qlora_model, load_tokenizer
from src.utils.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def save_test_dataset(test_dataset, output_dir: str):
    test_dir = Path(output_dir) / "test_dataset"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_dataset.save_to_disk(str(test_dir))
    logger.info("Test dataset saved to %s", test_dir)


def save_training_metadata(config, output_dir: str):
    metadata = {
        "model_id": config.model.model_id,
        "dataset_id": config.dataset.dataset_id,
        "seed": config.seed,
        "epochs": config.training.num_train_epochs,
        "learning_rate": config.training.learning_rate,
        "max_seq_length": config.model.max_seq_length,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
    }
    output_path = Path(output_dir) / "training_metadata.json"
    with open(output_path, "w") as file:
        json.dump(metadata, file, indent=4)
    logger.info("Training metadata saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a financial QA LLM using QLoRA.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set reproducibility
    set_reproducibility(config.seed)

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA training.")
    logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model.model_id)

    # Load dataset
    logger.info("Loading financial QA dataset...")
    dataset = load_financial_dataset(
        dataset_id=config.dataset.dataset_id,
        split=config.dataset.split,
        validation_size=config.dataset.validation_size,
        test_size=config.dataset.test_size,
        seed=config.seed,
    )

    logger.info("Train samples: %d", len(dataset["train"]))
    logger.info("Validation samples: %d", len(dataset["validation"]))
    logger.info("Test samples: %d", len(dataset["test"]))

    # Save exact test set
    save_test_dataset(dataset["test"], config.training.output_dir)

    # Prepare training datasets
    logger.info("Tokenizing datasets...")
    tokenized_dataset = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=config.model.max_seq_length,
    )

    # Load QLoRA model
    logger.info("Loading QLoRA model...")
    model = load_qlora_model(
        model_id=config.model.model_id,
        lora_r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
    )

    # Train
    logger.info("Starting model training...")
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        validation_dataset=tokenized_dataset["validation"],
        config=config,
    )

    # Save adapter
    logger.info("Saving LoRA adapter...")
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

    # Save metadata
    save_training_metadata(config, config.training.output_dir)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()