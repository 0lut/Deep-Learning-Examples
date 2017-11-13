import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import numpy as np 
from util import initialize_weights


class generator_model(nn.Module):

    def __init__(self, h, n, output_dims=(3, 64, 64)):
        super(generator_model, self).__init__()
        self.n = n
        self.h = h
        channel, width, height = output_dims

        self.blocks = int(np.log2(width) - 2)

        self.fc = nn.Linear(h, 8*8*n)

        conv_layers = []


        for i in range(self.blocks):
            
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, padding=1))
            conv_layers.append(nn.ELU())
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, padding=1))
            conv_layers.append(nn.ELU())



            if self.blocks - 1 > i :
                conv_layers.append(nn.UpsamplingNearest2d(scale_factor=2))


        
        conv_layers.append(nn.Conv2d(n, channel, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*conv_layers)


    def forward(self, x):

        fcout = self.fc(x).view(-1, self.n, 8, 8)
        return self.conv(fcout)
    
class dicriminator_model(nn.Module):

    def __init__(self, h, n, input_dims=(3, 64, 64)):
        super(dicriminator_model, self).__init__()


        self.n = n
        self.h = h
        channel, width, height = input_dims

        encoder = []

        self.blocks = int(np.log2(width) - 2)

        encoder.append(nn.Conv2d(n, n, kernel_size=3, padding=1))
        prev_channel_size = n

        for i in range(self.blocks):
            channel_size = (i+1) * n
            encoder.append(nn.Conv2d(prev_channel_size, channel_size, kernel_size=3, padding=1))
            encoder.append(nn.ELU())
            encoder.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1))
            encoder.append(nn.ELU())
            

            if i < self.blocks - 1:
                encoder.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1))
                encoder.append(nn.ELU())


            prev_channel_size = channel_size


        
        self.encoder_model = nn.Sequential(*encoder)



        self.fc1 = nn.Linear(8 * 8 * self.blocks * n, h)

        decoder = []

        for i in range(self.blocks):
            decoder.append(nn.Conv2d(n, n, kernel_size=3, padding=1))
            decoder.append(nn.ELU())
            decoder.append(nn.Conv2d(n, n, kernel_size=3, padding=1))
            decoder.append(nn.ELU())
            
            if i < self.blocks - 1:
                decoder.append(nn.UpsamplingNearest2d(scale_factor=2))

        
        
        decoder.append(nn.Conv2d(n, channel, kernel_size=3, padding=1))
        decoder_model = nn.Sequential(*decoder)
        
        self.fc2 = nn.Linear(h, 8 * 8 * n)


        def forward(self, x):

            x = self.encoder_model(x).view(x.size(0), -1)
            x = self.fc1(x)

            x = self.fc2(x).view(-1, self.n, 8, 8)
            x = self.decoder_model(x)

            return x
        