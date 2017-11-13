from __future__ import print_function
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DataSetFromFolder
import torch.cuda
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from models import generator_model,dicriminator_model
from util import initialize_weights
from torchvision import transforms
import numpy as np 
from os.path import join


h = 64
n = 128
batch_size = 16
cudnn.benchmark = True
lr = 1e-3

dtype = torch.cuda.FloatTensor
torch.cuda.manual_seed(619)

path = "dataset/"


train_transform = transforms.Compose([transforms.CenterCrop(160), 
                    transforms.Scale(size=64), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# train_set = DataSetFromFolder(join(join(path,"CelebA"), "train"), train_transform)


# train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=16, shuffle=True)



G = generator_model(h, n).type(dtype)
G.apply(initialize_weights)


D = dicriminator_model(h, n).type(dtype)
D.apply(initialize_weights)

real_data = Variable(torch.FloatTensor(batch_size, 3, 64, 64)).type(dtype)
logits_real = Variable(torch.FloatTensor(batch_size, h)).type(dtype)
logits_fake = Variable(torch.FloatTensor(batch_size, h)).type(dtype)

optimG = optim.Adam(G.parameters(), lr=lr)
optimD = optim.Adam(D.parameters(), lr=lr)


