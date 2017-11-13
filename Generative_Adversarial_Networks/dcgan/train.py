from models_losses import Generator
from models_losses import Discriminator
from models_losses import discriminator_loss
from models_losses import generator_loss
from utils import *
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--filter', type=int, default=128, help='enter the number of filters')
opt = parser.parse_args()
d = opt.filter
img_size = 64
batch_size = 128
vis = visdom.Visdom()


transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./datasets/MNIST', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

lr = 0.0002

G = Generator(d)
G.cuda()
G.weight_init()

D = Discriminator(d)
D.cuda()
D.weight_init()



G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))




num_iter = 0
train_epoch = 20
print ('training start!')
if not os.path.isdir('models'):
    os.mkdir('models')

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, _ in train_loader:
        if len(x_) != 128:
            continue

        #Discriminator training
        D.zero_grad()
        mini_batch = x_.size()[0]
        x_ = Variable(x_.cuda())
    
        D_real = D(x_).squeeze()
        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        fake_images = G(z_)

        D_fake = D(fake_images).squeeze()
       

        D_train_loss = discriminator_loss(D_real, D_fake)

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        #Generator training
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        fake_images = G(z_)
        D_result = D(fake_images).squeeze()
        G_train_loss = generator_loss(D_result)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        num_iter += 1
        if num_iter % 250 == 249:
                img_np = fake_images.data.cpu().numpy()
                vis.images(img_np[0:15])
                print ('[%d/%d], loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch,  torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))

    


    torch.save(G.state_dict(), './models/' + str(epoch) + ' G')
    torch.save(D.state_dict(), './models/' +    str(epoch) + ' D')


