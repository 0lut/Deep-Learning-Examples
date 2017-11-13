import visdom
import numpy as np
import torch

vis = visdom.Visdom()
def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    vis.images(images.reshape([16,1,64,64]))

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2
