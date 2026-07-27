from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    model_id: str
    max_seq_length: int


@dataclass
class DatasetConfig:
    dataset_id: str
    split: str
    validation_size: float
    test_size: float


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int


@dataclass
class LoRAConfig:
    r: int
    alpha: int
    dropout: float


@dataclass
class CheckpointConfig:
    save_steps: int
    save_total_limit: int


@dataclass
class EvaluationConfig:
    output_dir: str
    max_new_tokens: int
    num_samples: Optional[int]


@dataclass
class WandBConfig:
    project: str
    run_name: str


@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    lora: LoRAConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    wandb: WandBConfig
    seed: int


def load_config(config_path: str) -> Config:

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return Config(
        model=ModelConfig(**config["model"]),
        dataset=DatasetConfig(**config["dataset"]),
        training=TrainingConfig(**config["training"]),
        lora=LoRAConfig(**config["lora"]),
        checkpoint=CheckpointConfig(**config["checkpoint"]),
        evaluation=EvaluationConfig(**config["evaluation"]),
        wandb=WandBConfig(**config["wandb"]),
        seed=config["seed"],
    )

