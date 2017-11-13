from dataLoad import dataSets
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import Classifier
import datetime
import time
import math
gpu_dtype = torch.cuda.FloatTensor
def check_accuracy(model, loader, isTest=False):
   
    print('Checking accuracy on validation set')
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
    if isTest:
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        torch.save(model.state_dict(), './models/'+ st + '   accuracy'+ str(acc))




def train(model, loader_train, loss_fn, optimizer, print_every=50, num_epochs=66, valset=None, check_valset=True):
    for epoch in range(num_epochs):
        if epoch % 5 == 1 and check_valset:
            check_accuracy(model, valset)

        print("Starting epoch %d / %d" % (epoch+1, num_epochs))
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

data = dataSets()
torch.cuda.synchronize()
loss_fn = nn.CrossEntropyLoss()
model = Classifier()
optimizer = optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4)

for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

train(model, data.cifar10_train_loader ,loss_fn , optimizer, valset=data.cifar10_val_loader)
print("-----checking accuracy on test set-------")
check_accuracy(model, data.cifar10_test_loader, True)