import os

import wandb

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def train_model(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
    config,
):

    if os.getenv("WANDB_API_KEY"):

        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
        )

        report_to = ["wandb"]

    else:

        report_to = []

    training_args = TrainingArguments(

        output_dir=config.training.output_dir,

        num_train_epochs=(
            config.training.num_train_epochs
        ),

        learning_rate=(
            config.training.learning_rate
        ),

        per_device_train_batch_size=(
            config.training.train_batch_size
        ),

        per_device_eval_batch_size=(
            config.training.eval_batch_size
        ),

        gradient_accumulation_steps=(
            config.training
            .gradient_accumulation_steps
        ),

        bf16=True,

        optim="paged_adamw_8bit",

        logging_steps=10,

        report_to=report_to,

        eval_strategy="steps",

        eval_steps=(
            config.checkpoint.save_steps
        ),

        save_strategy="steps",

        save_steps=(
            config.checkpoint.save_steps
        ),

        save_total_limit=(
            config.checkpoint.save_total_limit
        ),

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        seed=config.seed,
    )

    data_collator = (
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,
    )

    trainer.train()

    return trainer