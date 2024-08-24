#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
pa = ArgumentParser()
pa.add_argument("logfile", type=str, help="path to a log file")
pa.add_argument("--ntrain", type=int, help="Number of training images to load", default=1e5)
pa.add_argument("--lr", type=float, default=0.001, help="learning rate")
pa.add_argument("--cpu", action="store_true", help="Run training on the CPU (slow)")
pa.add_argument("--devID", type=int, default=1, help="GPU device Id")
pa.add_argument("--bs", type=int, help="batch size", default=40)
pa.add_argument("--nwork", type=int, default=10, help="number of data loader workers")
pa.add_argument("--adam", action="store_true")
args = pa.parse_args()

import os
import pandas as pd
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18

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

# Updated log file name
log = getLog("NewBL.log")

# Paths to the dataset files
training_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/train_out/info.csv'
testing_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/test_out/info.csv'

dev='cuda:%d' % args.devID
if args.cpu:
    dev= "cpu"
log.info(f"Running model on device {dev}.")

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
log.info("Loading dataloader...")
train_loader = DataLoader(training_dataset, batch_size=args.bs, shuffle=True, num_workers=args.nwork)
test_loader = DataLoader(testing_dataset, batch_size=args.bs, shuffle=True, num_workers=args.nwork)

tag="TestTra"
# Output folder for saving images
output_folder_root = "/mnt/data/ns1/brave/MARCO/MS/savedmodelfolder"
output_folder = os.path.join(output_folder_root, tag)

# Define the model
log.info("Loading Net")
net = resnet18(pretrained=False)  # Use resnet18 model
net.fc = nn.Linear(net.fc.in_features, 4)  # Adjust the final layer for 4 classes
net = net.to(dev)

log.info("Optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
if args.adam:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

log.info("Script local vars:")
log.info(globals())
for epoch in range(300):  # loop over the dataset multiple times

    net.train()
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(dev), labels.to(dev)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()

        # log.info statistics
        lossi = loss.item()
        train_loss += lossi
        if i % 5 == 0:
            log.info(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] loss: {lossi:.3f}')
    log.info(f'Done with epoch {epoch + 1}; train loss= {train_loss/len(train_loader):.6f}')
    
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            lossi = loss.item()
            test_loss += lossi
            if i % 5 == 0:
                log.info(f'Epoch {epoch + 1}, Batch {i + 1}/{len(test_loader)}] loss: {lossi:.3f}')
        log.info(f'Done with epoch {epoch + 1}; test loss= {test_loss/len(test_loader):.6f}')

log.info('Finished Training')
