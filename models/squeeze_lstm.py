import torch
from models.base import Base

class Squeeze_LSTM(Base):
    def __init__(self,
                 nparams=2,
                 ninputs=1,
                 noutputs=1,
                 hidden_size=32,
                 buffer_size=32, 
                 num_layers=1):
        super(Squeeze_LSTM, self).__init__()
        self.save_hyperparameters()

        input_size = ninputs + nparams
        self.buffer_size = buffer_size
        self.lstm = torch.nn.LSTM(input_size,
                                   self.hparams.hidden_size,
                                   self.hparams.num_layers,
                                   batch_first=False,
                                   bidirectional=False)
        
        self.linear = torch.nn.Linear(self.hparams.hidden_size, 
                                      self.hparams.noutputs)

    def forward(self, x, p):
        x = x[:,None,:]
        p = p[:,None,:]
        bs = x.size(0)

        to_pad = self.buffer_size - x.size()[-1] % self.buffer_size
        pad_tensor = torch.zeros(bs, 1, to_pad, device=self.device)
        x_padded = torch.cat([x, pad_tensor], dim=-1)
        bufferized = x_padded.view(bs, -1, self.buffer_size)

        rms = torch.sqrt(torch.sum(torch.square(bufferized), dim=-1)) 
        rms_w_p = rms.view(bs,-1,1)
        p = p.repeat(1, bufferized.size()[1], 1)
        rms_w_p = torch.cat([rms_w_p, p], dim=-1)
        out, _ = self.lstm(rms_w_p)
        out = self.linear(out)
        out = torch.nn.functional.leaky_relu(out)
        
        rms_flat = torch.flatten(rms)
        out_flat = torch.flatten(out)
        gain_coef = out_flat * rms_flat
        
        gain_coef = gain_coef[None,None,:]
        gain_up_inter = torch.nn.functional.interpolate(gain_coef, x.size()[-1], mode='linear')
        compressed = x * gain_up_inter
        return compressed