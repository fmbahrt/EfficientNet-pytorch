"""
Just a minimal training script for trying out the efficientnet implementation
on CIFAR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

from efficientnet import EfficientNet, Config
from efficientnet.randaugment import RandAugment

# Load the model
model = EfficientNet(Config.B0)

# some data fun
transform = transforms.Compose(
    [
        RandAugment(2, 3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

import time

start = time.time()
batches = 0
for i, data in enumerate(trainloader):
    batches += 1
    imgs, lbl = data
end = (time.time() - start) / batches
print("time per batch", end)

#import matplotlib.pyplot as plt
#def imshow(img):
#    img = img / 2 + 0.5
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()
#
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
#img_grid = torchvision.utils.make_grid(images)
#imshow(img_grid)

#testset = torchvision.datasets.CIFAR10(
#    root='./data',
#    train=False,
#    download=True,
#    transform=transform
#)
#
#testloader = torch.utils.data.DataLoader(
#    testset,
#    batch_size=32,
#    shuffle=False
#)
#
#criterion = nn.CrossEntropyLoss()
#opt       = optim.Adam(model.parameters())
#
## Time to train
#EPOCHS = 1
#for epoch in range(EPOCHS):
#
#    for i, data in enumerate(trainloader):
#        inputs, labels = data
#
#        opt.zero_grad()
#
#        outputs = model(inputs)
#        loss    = criterion(outputs, labels)
#        loss.backward()
#        opt.step()
#
#        print("loss: {}".format(loss.item()))
#
#print("BOOM")
