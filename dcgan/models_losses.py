import torch
import torch.nn as nn
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor


class Generator(nn.Module):

    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh())
        
    def weight_init(self):
        for m in self.model.parameters():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m[0])
                nn.init.xavier_uniform(m[1])
    
            
    def forward(self, input):
        out = self.model(input)
        return out


class Discriminator(nn.Module):

    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d*8, 1, 4, 1, 0),
            nn.Sigmoid())

    def weight_init(self):
        for m in self.model.parameters():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m[0])
                nn.init.xavier_uniform(m[1])
    
    def forward(self, x):
        out = self.model(x)
        return out


def discriminator_loss(logits_real, logits_fake):
    
    bce_loss = nn.BCELoss()
    loss_real = bce_loss(logits_real, Variable(torch.ones(logits_real.size())).type(dtype))
    loss_fake = bce_loss(logits_fake, Variable(torch.zeros(logits_fake.size())).type(dtype))
    loss = (loss_real + loss_fake)
    return loss
        
def generator_loss(logits_fake):
    
    bce_loss = nn.BCELoss()
    loss = bce_loss(logits_fake, Variable(torch.ones(logits_fake.size())).type(dtype))
    return loss
