from torch.utils.data import DataLoader, Dataset
import os

class OPKDataset(Dataset):
    all_files = []
    train_files = []
    val_files = []
    split = 'train'
    root_dir = 'minifier/'
    

    def __init__(self, split='train', root_dir='data_cache/'):
        self.root_dir = root_dir

        self._walk_root_directory()
        # self.train_files = self.all_files[:int(0.8*len(self.all_files))]
        # self.val_files = self.all_files[-int(0.2*len(self.all_files)):]
        self.split = split
        
        super().__init__()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        minified_filepath = self.all_files[index]
        original_filepath = minified_filepath.replace("_terser.min.js", ".js") if "_terser" in minified_filepath else minified_filepath.replace("_google.min.js", ".js")
        minified_file_content = open(file=minified_filepath).read()
        original_file_content = open(file=original_filepath).read()
        return minified_file_content, original_file_content
    
    def _walk_root_directory(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if ".min.js" in file:
                    self.all_files.append(os.path.join(root, file))

dt = OPKDataset(root_dir="/home/jovyan/outpost-thesis/datasets/minified")
print(len(dt))
