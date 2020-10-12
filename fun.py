"""
Just a minimal training script for trying out the efficientnet implementation
on CIFAR
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

from tqdm import tqdm

from efficientnet import EfficientNet, Config
from efficientnet.randaugment import RandAugment

parser = argparse.ArgumentParser(description="CIFAR10 Training")
parser.add_argument('--lr', default=0.1, type=float, help="learning_rate")
parser.add_argument('--epochs', default=10, type=int, help="number of epochs")
parser.add_argument('--timm', action="store_true", help="Use TIMM implementation")

args = parser.parse_args()

# Cuda stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
if args.timm:
    import timm
    model = timm.create_model('efficientnet_b0', pretrained=False)
else:
    model = EfficientNet(Config.B0, num_classes=10)

# some data fun
transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False
)

EPOCHS = 10

criterion = nn.CrossEntropyLoss()
opt       = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)

def train(epoch):
    model.train()

    train_loss = 0
    correct    = 0
    total      = 0

    loop = tqdm(enumerate(trainloader), total=len(trainloader))
    for i, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        loop.set_description("EPOCH [{}/{}]".format(epoch+1, EPOCHS))
        loop.set_postfix(loss=train_loss/(i+1),
                         acc=100.*correct/total)

def test(epoch):
    model.eval()

    test_loss = 0
    correct   = 0
    total     = 0

    loop = tqdm(enumerate(testloader), total=len(testloader))
    for i, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        loop.set_description("EPOCH [{}/{}]".format(epoch+1, EPOCHS))
        loop.set_postfix(loss=test_loss/(i+1),
                         acc=100.*correct/total)


for i in range(EPOCHS):
    train(i)
    test(i)
