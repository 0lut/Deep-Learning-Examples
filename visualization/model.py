import torch
import torchvision
import torch.nn as nn
from utils import *
dtype = torch.FloatTensor
batch_size = 128
class Discriminator(nn.Module):
    def __init__(self, nChannels):

        super(Discriminator, self).__init__()
        self.modelD = nn.Sequential(
            #Unflatten(batch_size, 1, 28, 28),
            nn.Conv2d(nChannels, 64, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64*2, 64*4, 2, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64*4, 64*8, 2, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
        )
        self.modelD.type(dtype)
    def forward(self, input):
        output = self.modelD(input)
        return output.view(-1, 1)



class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator,self).__init__()
        self.modelG = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.modelG.type(dtype)
    def forward(self, input):
        output = self.modelG(input)
        return output



def ls_discriminator_loss(scores_real, scores_fake):
   
    scores_real -=1
    loss = Variable(torch.zeros(1)).type(dtype)
    loss = 0.5 * ((scores_real ** 2).mean() + (scores_fake ** 2).mean())
    return loss


def ls_generator_loss(scores_fake):
    
    scores_fake -=1
    loss = Variable(torch.zeros(1)).type(dtype)
    loss = 0.5 * (scores_fake ** 2).mean()
    return loss
