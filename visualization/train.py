from model import *
from utils import *
from dataloader import *
from torch.autograd import Variable
import pandas as pd 
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader

NUM_TRAIN = 1000
NUM_VAL = 128
mnist_train = dset.MNIST('../datasets/MNIST', train=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))

mnist_val = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))                          