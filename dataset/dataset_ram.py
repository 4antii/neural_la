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
        elif mode == 'test':
            mode_dir = self.dataset_dir / 'Test' 
        else:
            raise RuntimeError("Only train, valid and test modes are avalible for dataset")
        
        self.input_files = list(mode_dir.glob("input_*.wav"))
        self.output_files = list(mode_dir.glob("target_*.wav"))
        self.input_files.sort()
        self.output_files.sort()
        
        self.params = [extract_params(str(filename)) for filename in self.output_files]

        self.input_audios = []
        self.output_audios = []
        self.params_list = []
        
        for idx, (ouput_file, input_file, parameters) in tqdm(enumerate(zip(self.output_files, self.input_files, self.params)), total=len(self.input_files)):
            inp_id = int(str(input_file.stem).split("_")[1])
            oup_id = int(str(ouput_file.stem).split("_")[1])
            assert inp_id == oup_id

            input_audio, sr = torchaudio.load(input_file)
            output_audio, sr = torchaudio.load(ouput_file)
            assert input_audio.size()[-1] == output_audio.size()[-1]
            assert sr == 44100

            self.file_examples = []
            
            cut_samples = cut_sec * sr
            padding_to_end = cut_samples - (input_audio.size()[-1] % cut_samples)
            
            input_audio = F.pad(input_audio, (0, padding_to_end), "constant", 0)
            output_audio = F.pad(output_audio, (0, padding_to_end), "constant", 0)

            input_chunks = input_audio.view(-1, cut_samples)
            output_chunks = output_audio.view(-1, cut_samples)

            params_chunks = torch.Tensor(parameters)[None, :]
            params_chunks = params_chunks.repeat(input_chunks.size()[0],1)

            self.input_audios.append(input_chunks)
            self.output_audios.append(output_chunks)
            self.params_list.append(params_chunks)

        self.input_audios = torch.cat(self.input_audios, dim=0)
        self.output_audios = torch.cat(self.output_audios, dim=0)
        self.params_list = torch.cat(self.params_list, dim=0)

    def __len__(self):
        return len(self.input_audios)

    def __getitem__(self, idx):
        return self.input_audios[idx], self.output_audios[idx], self.params_list[idx]