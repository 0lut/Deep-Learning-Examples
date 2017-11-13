import torch
import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
from utils import ChunkSampler
class dataSets():
    def __init__(self, batch_size=128, num_train=49000, num_val=1000):
        transforms_normal = T.Compose([T.ToTensor()])
        negative_image = lambda x: 255 - x  
        self.cifar10_training = dset.CIFAR10('./datasets/Cifar10', train=True, transform=transforms_normal,download=True)
        self.cifar10_train_loader = DataLoader(self.cifar10_training, batch_size=batch_size, sampler=ChunkSampler(num_train, 0))

        self.cifar10_val = dset.CIFAR10('./datasets/Cifar10', train=True, transform=transforms_normal, download=True)
        self.cifar10_val_loader = DataLoader(self.cifar10_val, batch_size=batch_size,sampler=ChunkSampler(num_val, num_train)) 

        self.cifar10_test = dset.CIFAR10('./datasets/Cifar10', train=False, transform=transforms_normal, download=True)
        self.cifar10_test_loader = DataLoader(self.cifar10_test, batch_size=batch_size)

        # transforms_neg = T.Compose([T.Lambda(negative_image)])
        # self.cifar10_neg_training = dset.CIFAR10('../datasets/Cifar10', train=True, transform=transforms_neg, download=True)
        # self.cifar10_neg_train_loader = DataLoader(self.cifar10_neg_training, batch_size=batch_size/8, sampler=ChunkSampler(num_train/8, 0))



asdd = dataSets()