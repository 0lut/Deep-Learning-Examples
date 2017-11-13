import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import random
import torch.nn as nn
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt


def compute_saliency_map(X, y, model):
    '''
    X = input images (N, C, H, W)
    y = labels of X (N, 1)
    model = the cnn model
    '''

    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    N,C,H,W = X.size()
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(Y)
    scores = model(X_var)
    loss = loss_fn(scores, y_var)
    loss.backward()
    saliency = torch.max(X_var.grad.abs(), 1)    

    return saliency[0].data.view(N,H,W)



def show_saliency_map(X, y):
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    
    saliency = saliency.cpu().numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


