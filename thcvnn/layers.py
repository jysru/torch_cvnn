import torch
import torch.nn as nn


class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=torch.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=torch.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / torch.sqrt(fan_in)
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input):
        real = input.real
        imag = input.imag
        
        out_real = torch.matmul(real, self.weight_real.t()) - torch.matmul(imag, self.weight_imag.t())
        out_imag = torch.matmul(real, self.weight_imag.t()) + torch.matmul(imag, self.weight_real.t())
        
        if self.bias_real is not None:
            out_real += self.bias_real
            out_imag += self.bias_imag
        
        return torch.complex(out_real, out_imag)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_real is not None
        )


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, input):
        real = input.real
        imag = input.imag
        
        real_real = self.conv_real(real)
        real_imag = self.conv_real(imag)
        imag_real = self.conv_imag(real)
        imag_imag = self.conv_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        return torch.complex(real_out, imag_out)
    

class ComplexConvTranspose2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        
        self.conv_tran_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 
                                                 output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        
    def forward(self, input):
        real = input.real
        imag = input.imag
        
        real_real = self.conv_tran_real(real)
        real_imag = self.conv_tran_real(imag)
        imag_real = self.conv_tran_imag(real)
        imag_imag = self.conv_tran_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        return torch.complex(real_out, imag_out)



class ComplexDropout(nn.Module):

    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout_real = nn.Dropout(p)
        self.dropout_imag = nn.Dropout(p)

    def forward(self, input):
        return torch.complex(
            self.dropout_real(input.real),
            self.dropout_imag(input.imag)
        )
