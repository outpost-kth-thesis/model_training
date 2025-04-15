from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from finetuning.dataloader import OPKDataset

model_name = 'meta-llama/Llama-3.1-8B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = OPKDataset()

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

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
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
