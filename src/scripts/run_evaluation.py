import argparse

from datasets import load_dataset

from src.utils.config import (
    load_config,
)

from src.utils.model import (
    load_tokenizer,
)

from src.utils.evaluate import (
    evaluate_model,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    tokenizer = load_tokenizer(
        config.model.model_id
    )

    # Load your trained adapter
    from peft import PeftModel

    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    import torch

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.model_path,
    )

    model.eval()

    dataset = load_dataset(
        config.dataset.dataset_id,
        split=config.dataset.split,
    )

    # Recreate exactly the same split
    first_split = dataset.train_test_split(
        test_size=(
            config.dataset.validation_size
            + config.dataset.test_size
        ),
        seed=config.seed,
    )

    temporary_dataset = first_split["test"]

    second_split = (
        temporary_dataset.train_test_split(
            test_size=0.5,
            seed=config.seed,
        )
    )

    test_dataset = second_split["test"]

    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        output_dir=(
            config.evaluation.output_dir
        ),
        max_new_tokens=(
            config.evaluation.max_new_tokens
        ),
        num_samples=(
            config.evaluation.num_samples
        ),
    )


if __name__ == "__main__":
    main()