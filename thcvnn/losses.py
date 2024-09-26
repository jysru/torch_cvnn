import torch
import torch.nn as nn


class ComplexMSELoss(nn.Module):

    def __init__(self):
        super(ComplexMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        real_loss = self.mse(pred.real, target.real)
        imag_loss = self.mse(pred.imag, target.imag)
        return real_loss + imag_loss
    

