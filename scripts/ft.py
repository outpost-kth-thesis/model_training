from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from data.tokenization import pad_token
from data.dataset_lightning import OPKDatasetLightning
from dotenv import load_dotenv
from data.tokenization import get_tokenizer
import os
import torch
import pytorch_lightning as pl

load_dotenv()

class CausalLM(pl.LightningModule):
    def __init__(self, lr=5e-5):
        self.save_hyperparameters()
        self.model_name = os.getenv("MODEL_NAME")

    def forward(self, input_ids, attention_mask, labels):
        self.model(input_ids, attention_mask, labels)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def configure_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config)
        self.model.resize_token_embeddings(len(get_tokenizer()))
        self.model.config.pad_token = pad_token




if __name__ == "__main__":
    model = CausalLM()
    dataset = OPKDatasetLightning()
    trainer = pl.Trainer(max_epochs=3, accelerator='auto', device='auto')
    trainer.fit(model=model, datamodule=dataset)

