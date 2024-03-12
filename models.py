import torch
import torch.nn as nn
import math

class convolutional_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
            
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = convolutional_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = convolutional_block(out_channels + out_channels, out_channels)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
        
      
class UNETLatentEdgePredictor(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        """ Encoder """
        self.e1 = encoder_block(in_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = convolutional_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ Edge Map """
        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x, t):
        pos_elem = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_layers)]
        pos_encoding = torch.cat(pos_elem, dim=1)
        x = torch.cat((x, t, pos_encoding), dim=1)
        """ Encoder """
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs.to(torch.float16)
    
class SketchSimplificationNetwork(nn.Sequential):
    def __init__(self):
        super().__init__( # Sequential,
            nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid(),
        )