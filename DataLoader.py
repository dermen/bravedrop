#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
pa = ArgumentParser()
pa.add_argument("logfile", type=str, help="path to a log file")
pa.add_argument("--ntrain", type=int, help="Number of training images to load", default=1e5)
pa.add_argument("--lr", type=float, default=0.001, help="learning rate")
pa.add_argument("--cpu", action="store_true", help="Run training on the CPU (slow)")
args = pa.parse_args()

import os
import pandas as pd
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

print("Done Import")

# Define the labels map
labels_map = {
    0: "Clear",
    1: "Crystals",
    2: "Other",
    3: "Precipitate",
}


def getLog(filename=None, level="info", do_nothing=False):
    """
    :param filename: optionally log to a file
    """
    levels = {"info": 20, "debug": 10, "critical": 50}
    if do_nothing:
        logger = logging.getLogger()
        logger.setLevel(levels["critical"])
        return logger
    logger = logging.getLogger("bravedrop")
    logger.setLevel(levels["info"])

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(levels[level])
    logger.addHandler(console)

    if filename is not None:
        logfile = logging.FileHandler(filename)
        logfile.setFormatter(logging.Formatter("%(asctime)s >>  %(message)s"))
        logfile.setLevel(levels["info"])
        logger.addHandler(logfile)
    return logger



class MARCODataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None, maximages=None):
        self.img_data = pd.read_csv(annotations_file, nrows=maximages)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data.iloc[idx, 0]
        label = self.img_data.iloc[idx, 2]  # Assuming label_id is the third column
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label



log = getLog(args.logfile)
# Paths to the dataset files
training_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/train_out/info.csv'
testing_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/test_out/info.csv'

# Define transformations with resizing
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # Resize to 600 x 600 or any consistent size
    transforms.ToTensor()
])

# Create datasets
log.info("Loading datasets...")
ntrain = args.ntrain
ntest = int(0.1*ntrain)
training_dataset = MARCODataset(training_file, transform=transform, maximages=ntrain)
testing_dataset = MARCODataset(testing_file, transform=transform, maximages=ntest)
log.info("Doing %d train images and %d test images" % (ntrain, ntest))

# Create DataLoaders
bs=10
log.info("Loading dataloader...")
train_loader = DataLoader(training_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=bs, shuffle=True)

tag="TestTra"
# Output folder for saving images
output_folder_root = "/mnt/data/ns1/brave/MARCO/MS/savedmodelfolder"
output_folder = os.path.join(output_folder_root,tag)

import torch.nn as nn
import torch.nn.functional as F

import torch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 147 * 147, 120) #For 600 x 600 images
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


log.info("Loading Net")
net = Net()

dev='cuda:1'
if args.cpu:
    dev= "cpu"
net=net.to(dev)

import torch.optim as optim

log.info("Optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)

log.info("Script local vars:")
log.info(globals())
for epoch in range(300):  # loop over the dataset multiple times

    net.train()
    training_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels=labels.to(dev)
        inputs=inputs.to(dev)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log.info statistics
        lossi = loss.item()
        training_loss += lossi
        if i % 5 == 0:
            log.info(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] loss: {lossi:.3f}')           
    log.info('epoch',epoch + 1,'training loss',training_loss/len(train_loader))
    net.eval()
    test_loss=0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
        
            inputs, labels = data
            labels=labels.to(dev)
            inputs=inputs.to(dev)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            lossi = loss.item()
            test_loss += lossi
            if i% 5==0:
                log.info(f'Epoch {epoch + 1}, Batch {i + 1}/{len(test_loader)}] loss: {lossi:.3f}')           
        log.info('epoch',epoch + 1,'test loss',test_loss/len(test_loader))
            
log.info('Finished Training')
