from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from dataloader import OPKDataset
from peft import LoraConfig


model_name = 'Qwen/Qwen2.5-3B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_4bit=True)


max_input_length = 512
max_output_length = 512

def preprocess(example):
    model_inputs = tokenizer(
        example["input"], max_length=max_input_length, truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"], max_length=max_output_length, truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

dataset = OPKDataset(transform=preprocess)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    modules_to_save=["lm_head"],
)

model.add_adapter(lora_config)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    remove_unused_columns=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
