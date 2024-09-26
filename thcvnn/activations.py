import torch
import torch.nn as nn


class ComplexReLU(nn.Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            torch.relu(input.real),
            torch.relu(input.imag)
        )
    

class ModReLU(nn.Module):

    def __init__(self, in_features, init_threshold=0.5):
        super(ModReLU, self).__init__()
        self.in_features = in_features
        self.threshold = nn.Parameter(torch.Tensor(in_features))
        self.threshold.data.fill_(init_threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        modulus = torch.abs(input)
        angle = torch.angle(input)
        return torch.relu(modulus + self.threshold) * torch.exp(1j * angle)


class ComplexLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            nn.functional.leaky_relu(input.real, self.negative_slope),
            nn.functional.leaky_relu(input.imag, self.negative_slope)
        )

    
class ComplexModLeakyReLU(nn.Module):

    def __init__(self, in_features, negative_slope=0.01, init_threshold=0.5):
        super(ComplexModLeakyReLU, self).__init__()
        self.in_features = in_features
        self.negative_slope = negative_slope
        self.threshold = nn.Parameter(torch.Tensor(in_features))
        self.threshold.data.fill_(init_threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        modulus = torch.abs(input)
        angle = torch.angle(input)
        
        activated_modulus = torch.where(
            modulus + self.threshold > 0,
            modulus + self.threshold,
            self.negative_slope * (modulus + self.threshold)
        )
        
        return activated_modulus * torch.exp(1j * angle)
    

class ComplexSigmoid(nn.Module):

    def __init__(self):
        super(ComplexSigmoid, self).__init__()

    def forward(self, input):
        return torch.sigmoid(input.real) + 1j * torch.sigmoid(input.imag)


class ComplexTanh(nn.Module):
    
    def __init__(self):
        super(ComplexTanh, self).__init__()

    def forward(self, input):
        return torch.tanh(input.real) + 1j * torch.tanh(input.imag)

