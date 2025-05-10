import lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "meta-llama/Meta-Llama-3-8B"


class CausalLM(pl.LightningModule):
    def __init__(self, lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = os.getenv("MODEL_NAME")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config)
        # self.model.resize_token_embeddings(len(get_tokenizer()))
        self.model.config.pad_token = pad_token

    def forward(self, input_ids, attention_mask, labels):        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attn_mask, labels)
        loss = output.loss
        # print("getting loss", loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)