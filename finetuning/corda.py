from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from finetuning.dataloader import OPKDataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import random_split
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from trl import SFTTrainer, SFTConfig
from torch import no_grad


model_name = 'meta-llama/Llama-3.1-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
dataset = OPKDataset("/home/jovyan/outpost-thesis/datasets/minified")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

sample_set = train_set.train_set[:256]
train_set = train_set[256:]

def run_model():
    for each in sample_set:
        with no_grad():
            model(each)

corda_config = CordaConfig(
    corda_method="ipm",
)

lora_config = LoraConfig(
    init_lora_weights="corda",
    corda_config=corda_config,
)

preprocess_corda(model, lora_config, run_model=run_model)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("corda-llama-3.1-8B")