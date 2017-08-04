import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import copy

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
    
def train(model, loss_fn, optimizer, num_epochs = 1 ,valset=None, check_valset=False):
    for epoch in range(num_epochs):
        if epoch % 5 and check_valset == 1:
            check_accuracy(model, valset)
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() 
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()



class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                          transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)






dtype = torch.FloatTensor 

print_every = 100

fixed_model_base = nn.Sequential(
                                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                                nn.MaxPool2d(5, stride=2, padding=(0,1,0,1)),
                                nn.Conv2d(32,32,kernel_size=7, stride=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128,64,kernel_size=5,stride=1,padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64,32,kernel_size=1,stride=1,padding=1),
                                nn.AvgPool2d(7, stride=2,padding=(0,1,0,1)),
                                Flatten(),
                                nn.Linear(96,10),
        
                                
            )

fixed_model = fixed_model_base.type(dtype)



x = torch.randn(64, 3, 32, 32).type(dtype)
x_var = Variable(x.type(dtype))
ans = fixed_model(x_var)       


gpu_dtype = torch.cuda.FloatTensor

fixed_model_gpu = copy.deepcopy(fixed_model_base).type(gpu_dtype)

x_gpu = (torch.randn(64, 3, 32, 32) * np.sqrt(2/64*3*32*32)) .type(gpu_dtype)


x_var_gpu = Variable(x.type(gpu_dtype)) # Construct a PyTorch Variable out of your input data
ans = fixed_model_gpu(x_var_gpu)        # Feed it through the model! 




ans = fixed_model(x_var)


torch.cuda.synchronize() # Make sure there are no pending GPU computations
ans = fixed_model_gpu(x_var_gpu)        # Feed it through the model! 
torch.cuda.synchronize() # Make sure there are no pending GPU computations


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(fixed_model_gpu.parameters(),lr = 1e-3, weight_decay = 1e-4, nesterov=True, momentum = 0.9)


torch.cuda.random.manual_seed(12345)
fixed_model_gpu.apply(reset)
train(fixed_model_gpu, loss_fn, optimizer, num_epochs=61, valset=loader_val)
check_accuracy(fixed_model_gpu, loader_val)
check_accuracy(fixed_model_gpu, loader_test)


