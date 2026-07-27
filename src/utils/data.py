from datasets import Dataset, DatasetDict, load_dataset

REQUIRED_COLUMNS = {"context", "question", "answer"}

def validate_dataset(dataset: Dataset) -> None:
    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def format_example(example: dict) -> dict:
    prompt = (
        "### Financial Context:\n"
        f"{example['context']}\n\n"
        "### Question:\n"
        f"{example['question']}\n\n"
        "### Verified Answer:\n"
        f"{example['answer']}"
    )
    return {
        "text": prompt,
        "context": example["context"],
        "question": example["question"],
        "answer": example["answer"],
    }

def load_financial_dataset(
    dataset_id: str, split: str, validation_size: float, test_size: float, seed: int
) -> DatasetDict:
    dataset = load_dataset(dataset_id, split=split)
    validate_dataset(dataset)

    # First split: (1 - validation_size - test_size) train, remainder temporary
    first_split = dataset.train_test_split(test_size=(validation_size + test_size), seed=seed)
    train_dataset = first_split["train"]
    temporary_dataset = first_split["test"]

    # Split temporary dataset into validation and test
    validation_ratio = validation_size / (validation_size + test_size)
    second_split = temporary_dataset.train_test_split(test_size=(1 - validation_ratio), seed=seed)
    validation_dataset = second_split["train"]
    test_dataset = second_split["test"]

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    })

def prepare_dataset(dataset: DatasetDict, tokenizer, max_seq_length: int) -> DatasetDict:
    # Format examples
    formatted_dataset = dataset.map(format_example)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    columns_to_remove = ["text", "context", "question", "answer"]

    # Tokenize training dataset
    tokenized_train = formatted_dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing training dataset",
    )

    # Tokenize validation dataset
    tokenized_validation = formatted_dataset["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing validation dataset",
    )

    # Return the tokenized datasets
    return DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_validation,
    })