"""
Transform dataset into a lightning dataset
"""


from torch.utils.data import DataLoader
from dotenv import load_dotenv
import pytorch_lightning as pl
import tokenization
import dataset
from torch.utils.data import random_split, DataLoader


load_dotenv()


class OPKDatasetLightning(pl.LightningDataModule):
    def __init__(self):
        self.batch_size = 1

    def setup(self):
        self.dataset = dataset.OPKDatasetPT(transform=tokenization.tokenize)
        self.total_len = len(self.dataset)
        self.val_len = int(0.2 * len(self.dataset))
        self.train_len = self.total_len - self.val_len
        self.train_set, self.val_set = random_split(self.dataset, [self.train_len, self.val_len])
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
    


if __name__ == "__main__":
    datamodule = OPKDatasetLightning()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        print(batch)
        break
