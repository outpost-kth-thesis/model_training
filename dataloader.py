from torch.utils.data import DataLoader, Dataset
import os
from dotenv import load_dotenv

load_dotenv()

class OPKDataset(Dataset):
    all_files = []
    train_files = []
    val_files = []
    split = 'train'
    root_dir = 'minifier/'
    

    def __init__(self, split='train'):
        self.root_dir = os.getenv("DATASET_DIR")
        self._walk_root_directory()
        self.split = split
        
        super().__init__()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        minified_filepath = self.all_files[index]
        original_filepath = minified_filepath.replace("_terser.min.js", ".js") if "_terser" in minified_filepath else minified_filepath.replace("_google.min.js", ".js")
        minified_file_content = open(file=minified_filepath).read()
        original_file_content = open(file=original_filepath).read()
        return {"input": minified_file_content, "output": original_file_content}
    
    def _walk_root_directory(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if ".min.js" in file:
                    self.all_files.append(os.path.join(root, file))

dt = OPKDataset(root_dir="/home/jovyan/outpost-thesis/datasets/minified")
print(len(dt))
