import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import Flatten

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(16,32, kernel_size=3),
                                nn.MaxPool2d(3, stride=2, padding=(0,1,0,1)),
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
                                nn.Linear(96,10))
        #self.model.type(torch.cuda.FloatTensor)

    def forward(self, x):
        out = self.model(x)
        return out
        
        
model = Classifier()
x = Variable((torch.randn(1,3,32,32)))

print(model(x).size())



