import torch
import GPUtil
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
from dotenv import load_dotenv
load_dotenv()


if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU instead")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


try:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
except Exception as e:
    print("Warning: WANDB_API_KEY not found")

wandb.init(project="Verifi-Finance-LLM", name="kaggle-llama-run-1")


dataset = load_dataset("virattt/financial-qa-10K", split="train")

base_model_id = "meta-llama/Llama-3.2-1B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

if tokenizer.pad_token is None:
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def formatting_prompts_func(example):
    text = (
        f"### Financial Context:\n{example['context']}\n\n"
        f"### Question:\n{example['question']}\n\n"
        f"### Verified Answer:\n{example['answer']}"
    )
    return {"text": text}

formatted_dataset = dataset.map(formatting_prompts_func)

tokenized_train_dataset = []
for phrase in formatted_dataset:
  tokenized_train_dataset.append(tokenizer(phrase["text"]))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

#lora configuration
config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
    
)


#parameter efficient model
model = get_peft_model(model, config)


output_dir="../finetunedModel"

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args= transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        # max_steps=20,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_dir="./log",

        # W&B Logging
        report_to="wandb",
        logging_steps=10,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2, # Keep only 2 to avoid exceeding Kaggle's 20GB disk limit

),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache=False
trainer.train()


wandb.finish()


# 1. Define the final save path in Kaggle's permanent working directory
final_save_path = "../finetunedModel"

# 2. Save the trained LoRA adapter and model configuration
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)

print(f"Model and tokenizer successfully saved to {final_save_path}")