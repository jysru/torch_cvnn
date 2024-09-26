import torch
import torch.nn as nn

import thcvnn.layers as layers
import thcvnn.activations as activations


class ComplexMLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, hidden_activations, output_activation=None):
        super(ComplexMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        
        # Create the layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0], dtype=torch.complex64))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], dtype=torch.complex64))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size, dtype=torch.complex64))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.apply_activation(x, self.hidden_activations[i])
        
        x = self.layers[-1](x)
        if self.output_activation:
            x = self.apply_activation(x, self.output_activation)
        
        return x
    
    def apply_activation(self, x, activation):
        if activation == 'relu':
            return torch.relu(x.real) + 1j * torch.relu(x.imag)
        elif activation == 'leaky_relu':
            return torch.leaky_relu(x.real) + 1j * torch.leaky_relu(x.imag)
        elif activation == 'tanh':
            return torch.tanh(x.real) + 1j * torch.tanh(x.imag)
        elif activation == 'sigmoid':
            return torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)
        elif activation == 'mod_relu':
            return torch.relu(torch.abs(x)) * torch.exp(1j * torch.angle(x))
        elif activation is None:
            return x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        

class ComplexUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexUNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Expanding Path)
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)

        self.final_conv = ComplexConv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            layers.ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            activations.ComplexReLU(),
            layers.ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            activations.ComplexReLU()
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            layers.ComplexConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            activations.ComplexReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1.abs(), 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2.abs(), 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3.abs(), 2))

        # Bottleneck
        b = self.bottleneck(nn.functional.max_pool2d(e4.abs(), 2))

        # Decoder with skip connections
        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))

        out = self.final_conv(torch.cat([d1, e1], dim=1))

        return out