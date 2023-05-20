import torch 
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

def extract_params(filename):
    return float(filename.split("__")[1].replace(".wav","")) / 100.0, float(filename.split("__")[2].replace(".wav","")) / 100.0 

class LA2A_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, cut_sec=8, mode="train"):
        self.dataset_dir = Path(dataset_dir)
        self.mode = mode

        if mode == 'train':
            mode_dir = self.dataset_dir / 'Train'
        elif mode == 'valid':
            mode_dir = self.dataset_dir / 'Val'
        
        self.files = list(mode_dir.glob('*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_tensor, output_tensor = torch.load(str(self.files[idx]))
        param_parts = self.files[idx].name.split('_')
        param1 = param_parts[1]
        param2 = param_parts[2].split('.')[0]
        params = torch.Tensor([float(param1), float(param2)])
        return input_tensor, output_tensor, params