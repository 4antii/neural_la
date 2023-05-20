import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

#from models.lstm_mlp import LSTMModel
#from models.lstm_mlp_input import LSTMModelID
from models.squeeze_lstm import Squeeze_LSTM

path_to_data = Path('path_to_audio')
out_path = Path('output_path')
out_path.mkdir(exist_ok=True)
params = [0.0, 50.0]

params = torch.tensor([par / 100.0 for par in params])[None, :]

model_config = {
    'input_size':  1,
    'output_size': 1, 
    'hidden_size': 64,
    'train_loss': 'l1+stft'
}

model = Squeeze_LSTM(input_s=model_config['input_size'], 
                  output_s=model_config['output_size'], 
                  hidden_size=model_config['hidden_size'],
                  train_loss=model_config['train_loss']).load_from_checkpoint(checkpoint_path='./assets/squeeze_lstm/lstm_mlp_input-epoch=80-val_loss=1.78.ckpt')

with torch.no_grad():
    for p in tqdm(list(path_to_data.glob('*.wav'))):
        input_audio, sr = torchaudio.load(p)
        print(input_audio)
        print('inp', input_audio.dtype)
        params_for_input = params.repeat(input_audio.size()[0], 1)
        out = model.forward(input_audio, params_for_input)
        out = torch.squeeze(out, 1)
        print(input_audio)
        out_name = f"{p.stem}_processed.wav"
        out_p = out_path / out_name
        print('oup', out.dtype)
        #soundfile.write(out_p, out.cpu(), samplerate=44100)
        torchaudio.save(out_p, out.cpu(), sample_rate=44100, format='wav', bits_per_sample=32)