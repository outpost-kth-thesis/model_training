from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from finetuning.dataloader import OPKDataset
from peft import LoraConfig
from trl import SFTTrainer
from torch.utils.data import random_split

model_name = 'meta-llama/Llama-3.1-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
dataset = OPKDataset("/home/jovyan/outpost-thesis/datasets/minified")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules='all-linear',
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)


training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=val_set,
    peft_config=peft_config,
    dataset_text_field="review",
    tokenizer=tokenizer,
    max_seq_length=2048,
    args=training_arguments,
)

trainer.train()
trainer.save_model("./lora-Llama-3.1-8B")
