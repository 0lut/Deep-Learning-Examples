import numpy as np
import torch
import cv2
import torch.nn as nn
from torch.nn import init

def is_image_file(file):
    
    return any(file.endswith(extension) for extension in [".jpg", ".jpeg", ".png"])

def load_image(path, returnShape=False):
    img = cv2.imread(path)
    shape = img.shape
    
    if len(shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img,3, axis=2)
    
    
    img = cv2.resize(img, (64,64))
    img = preprocess(img)
    
    
    if returnShape:
        return (img,shape)
    
    return img


def save_img(img, path):
    
    img = deprocess(img)
    img = img.numpy()
    img *= 255.0
    img = img.clip(0, 255)
    img = np.transpose(img ,(1, 2, 0))
    img = cv2.resize(img ,(255, 200, 3))
    img = img.astype(np.uint8)
    cv2.imsave(img, path)
    
def preprocess(img):
    
    min = img.min()
    max = img.max()
    diff = max - min
    if diff == 0.0 :
        diff = 0.01
    
    img = (img-min)/diff





    assert img.max() <= 1 or img.min() >= 0, "bad scaled inputs"

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = torch.FloatTensor(img.size()).copy_(img)
    #RGB > BGR transformation
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    return img

def deprocess(img):
    #BGR > RBG tranformation

    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    #scaling between 0,1 instead -1,1
    img = img.add_(1).div_(2)

    return img


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)