import torch
import torch.nn as nn
import lightning as L
import pytorch_lightning as pl

import thcvnn.layers as layers
import thcvnn.losses as losses
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
        self.layers.append(layers.ComplexLinear(input_size, hidden_sizes[0], bias=True))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(layers.ComplexLinear(hidden_sizes[i], hidden_sizes[i+1], bias=True))
        
        # Output layer
        self.layers.append(layers.ComplexLinear(hidden_sizes[-1], output_size, bias=True))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if type(self.hidden_activations) == list:
                x = self.hidden_activations[i](x)
            else:
                x = self.hidden_activations(x)
        
        x = self.layers[-1](x)
        if self.output_activation:
            x = self.output_activation(x, self.output_activation)
        
        return x
    


class LitComplexMLP(L.LightningModule):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list[int],
            output_size: int,
            hidden_activations: list,
            output_activation = None,
            train_loss_fn = losses.ComplexMSELoss(),
            add_dropout: bool = False,
            ):
        super().__init__()
        
        # Define the layers of the MLP
        layers = []
        previous_size = input_size

        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LeakyReLU())  # Activation function (ReLU)
            if add_dropout:
                layers.append(nn.Dropout(0.2))
            previous_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(previous_size, output_size))
        if last_activation_layer is not None:
            layers.append(last_activation_layer)
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Define loss function
        self.loss_fn = train_loss_fn


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-4)
        optimizer = optim.RMSprop(self.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=10,
            cooldown=3,
            threshold=1e-2,
            min_lr=1e-7,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            }
        }

    

        

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

        self.final_conv = layers.ComplexConv2d(64, out_channels, kernel_size=1)

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