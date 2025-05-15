"""
This file to train Llama-3.1-8B
Quantization off
Dataset from Huggingface

"""

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load tokenizer and model
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Ensure pad token is defined
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load public dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Data collator for autoregressive LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./falcon-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()
