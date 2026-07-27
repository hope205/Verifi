import json
import re
from pathlib import Path
import evaluate
import torch
from tqdm import tqdm

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def exact_match(predictions, references):
    matches = 0
    for prediction, reference in zip(predictions, references):
        if normalize_text(prediction) == normalize_text(reference):
            matches += 1
    return matches / len(predictions)

def build_prompt(context: str, question: str) -> str:
    return (
        "### Financial Context:\n"
        f"{context}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Verified Answer:\n"
    )

def generate_predictions(model, tokenizer, test_dataset, max_new_tokens: int, num_samples=None):
    if num_samples:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

    predictions = []
    references = []

    for example in tqdm(test_dataset, desc="Evaluating model"):
        prompt = build_prompt(example["context"], example["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        predictions.append(prediction.strip())
        references.append(example["answer"])

    return predictions, references

def evaluate_model(model, tokenizer, test_dataset, output_dir, max_new_tokens=256, num_samples=None):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    predictions, references = generate_predictions(model, tokenizer, test_dataset, max_new_tokens, num_samples)

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    em_score = exact_match(predictions, references)

    results = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_results["bleu"],
        "exact_match": em_score,
        "num_samples": len(predictions),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "evaluation_results.json", "w") as file:
        json.dump(results, file, indent=4)

    with open(output_path / "predictions.json", "w") as file:
        json.dump(
            [{"prediction": p, "reference": r} for p, r in zip(predictions, references)],
            file,
            indent=4
        )

    print("\n===== EVALUATION RESULTS =====")
    for metric, score in results.items():
        if isinstance(score, float):
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")

    return results