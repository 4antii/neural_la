import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset.dataset_ram import LA2A_Dataset
from models.lstm_mlp_input import LSTMModelID

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision('highest')

config = {
    'dataset_dir': "path_to_dataset/SignalTrain_LA2A_Dataset_1.1",
    'train_data_len_sec': 5,
    'batch_size': 50,
    'learning_rate': 1e-3,
    'num_workers': 10,
}

model_config = {
    'input_size':  32,
    'output_size': 32, 
    'hidden_size': 80,
    'train_loss': 'l1+stft'
}

model = LSTMModelID(input_s=model_config['input_size'], 
                  output_s=model_config['output_size'], 
                  hidden_size=model_config['hidden_size'],
                  train_loss=model_config['train_loss'])


train_dataset = LA2A_Dataset(config['dataset_dir'],  config['train_data_len_sec'],  mode="train")
val_dataset = LA2A_Dataset(config['dataset_dir'],  config['train_data_len_sec'],  mode="valid")

train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                batch_size=config['batch_size'],
                                                num_workers=config['num_workers'],
                                                pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                shuffle=False,
                                                batch_size=16,
                                                num_workers=config['num_workers'],
                                                pin_memory=True)


checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_loss",
    mode="min",
    dirpath="./artefacts/lstm_mlp_input_80_stft_l1_v2",
    filename="lstm_mlp_input-{epoch:02d}-{val_loss:.2f}",
)

lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(gradient_clip_val=0.3, callbacks=[lr_monitor, checkpoint_callback], accelerator="gpu")
trainer.fit(model, train_dataloader, val_dataloader)