import torch
import auraloss
import pytorch_lightning as pl

class Base(pl.LightningModule):
    def __init__(self, 
            lr = 1e-3, 
            train_loss = "l1+stft",
            sample_rate=44100):
        super(Base, self).__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.L1Loss()
        self.esr = auraloss.time.ESRLoss()
        self.stft = auraloss.freq.STFTLoss()
        self.mel_stft = auraloss.freq.MelSTFTLoss(sample_rate=sample_rate)

    def forward(self, x, p):
        pass
 
    def training_step(self, batch, batch_idx):
        input, output, params = batch

        pred = self.forward(input, params)
        output=output[:,None,:]
 
        if self.hparams.train_loss == "l1":
            loss = self.l1(pred, output)
        elif self.hparams.train_loss == "esr":
            loss = self.esr(pred, output)
        elif self.hparams.train_loss == "stft":
            loss = self.stft(pred, output)
        elif self.hparams.train_loss == "l1+stft":
            loss = self.l1(pred, output) + self.stft(pred, output) 
        elif self.hparams.train_loss == "esr+mel_stft":
            loss = self.esr(pred, output) + self.mel_stft(pred, output)
        else:
            raise NotImplementedError(f"Losses options: l1, esr, stft, l1+stft, esr+mel_stft")

        del pred, output

        self.log('train_loss', 
                 loss.detach(), 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    def validation_step(self, batch):
        input, output, params = batch
        output=output[:,None,:]

        pred = self.forward(input, params)

        l1_loss      = self.l1(pred, output).detach()
        stft_loss    = self.stft(pred, output).detach()
        
        aggregate_loss = l1_loss + stft_loss 

        self.log('val_loss', aggregate_loss)
        self.log('val_loss/L1', l1_loss)
        self.log('val_loss/STFT', stft_loss)

    def test_step(self, batch):
        return self.validation_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }