"""
Dataloader as a Pytorch Dataset module
"""

from torch.utils.data import Dataset
import os
from dotenv import load_dotenv

load_dotenv()

class OPKDatasetPT(Dataset):
    all_files = []
    train_files = []
    val_files = []
    split = 'train'
    root_dir = os.getenv("DATASET_DIR")
    transform=None
    

    def __init__(self, transform=None, split='train'):
        self._walk_root_directory()
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        minified_filepath = self.all_files[index]
        original_filepath = minified_filepath.replace("_terser.min.js", ".js") if "_terser" in minified_filepath else minified_filepath.replace("_google.min.js", ".js")
        minified_file_content = open(file=minified_filepath).read()
        original_file_content = open(file=original_filepath).read()
        if self.transform:
            return self.transform(minified_file_content, original_file_content)
        else:
            return minified_file_content, original_file_content
    
    def _walk_root_directory(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if ".min.js" in file:
                    self.all_files.append(os.path.join(root, file))

if __name__ == "__main__":
    dt = OPKDatasetPT()
    print(len(dt))
