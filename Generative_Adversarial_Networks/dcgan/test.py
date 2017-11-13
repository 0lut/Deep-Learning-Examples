import models_losses as ml
import torch 
import torch.nn as nn
from torch.autograd import Variable
import visdom

vis = visdom.Visdom(env='test')
G = ml.Generator()
D = ml.Discriminator()
G.cuda()
D.cuda()

G.load_state_dict(torch.load('./models/14 G'))
D.load_state_dict(torch.load('./models/14 D'))
G.eval()
D.eval()

for i in range(20):
    rand_noise = Variable(torch.randn(128, 100, 1, 1)).cuda()
    fake_imgs = G(rand_noise)
    imgs_np = fake_imgs.data.cpu().numpy()
    vis.images(imgs_np[0:15])


