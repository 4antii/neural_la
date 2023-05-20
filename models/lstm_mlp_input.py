import torch
from models.base import Base

class LSTMModelID(Base):
    def __init__(self, 
                 nparams=2,
                 input_s=32,
                 output_s=32,
                 hidden_size=256,
                 num_layers=1):
        super(LSTMModelID, self).__init__()
        self.save_hyperparameters()
        
        self.ninputs = input_s
        input_size = output_s + nparams
        self.input_size = input_size
        self.lstm = torch.nn.LSTM(input_size,
                                   self.hparams.hidden_size,
                                   self.hparams.num_layers,
                                   bias=False,
                                   batch_first=True,
                                   bidirectional=False)
        
        self.linear1 = torch.nn.Linear(self.hparams.hidden_size + input_size, 
                                      self.hparams.hidden_size + input_size, bias=False)
        
        self.linear2 = torch.nn.Linear(self.hparams.hidden_size + input_size, 
                                      self.hparams.hidden_size + input_size, bias=False)
        
        self.linear3 = torch.nn.Linear(self.hparams.hidden_size + input_size, 
                                      self.hparams.output_s, bias=False)
        
        self.linear4 = torch.nn.Linear(self.hparams.output_s, 
                                      self.hparams.output_s, bias=False)

    def forward(self, x, p):
        x = x[:,None,:]
        p = p[:,None,:]

        bs = x.size(0)
        to_pad = self.ninputs - x.size()[-1] % self.ninputs
        pad_tensor = torch.zeros(bs,1,to_pad, device=self.device)
        x_padded = torch.cat([x, pad_tensor], dim=-1)
        x_padded = x_padded.view(bs, -1, self.ninputs)
        if p is not None:
            p = p.repeat(1, x_padded.size()[1], 1) 
            x_padded = torch.cat((x_padded, p), dim=-1)
        out, _ = self.lstm(x_padded)
        out = torch.cat([out, x_padded], dim=-1)
        out = torch.tanh(self.linear1(out))
        out = torch.tanh(self.linear2(out))
        out = torch.tanh(self.linear3(out))
        out = self.linear4(out)
        new_batch = [torch.unsqueeze(batch.flatten()[:x.size()[-1]], dim=0) for batch in out]
        out = torch.cat(new_batch, dim=0)
        out = torch.unsqueeze(out, dim=1)
        return out