#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser
import os
import pandas as pd
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models  # Fixed import here
import torch
import torch.nn as nn
import torch.optim as optim

# Argument parsing
pa = ArgumentParser()
pa.add_argument("logfile", type=str, help="path to a log file")
pa.add_argument("--cpu", action="store_true", help="Run training on the CPU (slow)")
pa.add_argument("--devID", type=int, default=1, help="GPU device Id")
pa.add_argument("--nwork", type=int, default=10, help="number of data loader workers")
pa.add_argument("--resnet", type=int, choices=[18, 34, 50], default=18, help="ResNet architecture to use (18, 34, or 50)")
args = pa.parse_args()

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

# Initialize logging
log = getLog(args.logfile)
# Paths to the dataset files
#testing_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/test_out/info.csv'
testing_file = '/mnt/data/ns1/brave/MARCO/MS/ARI.csv'  # Updated testing file path

dev = 'cuda:%d' % args.devID
if args.cpu:
    dev = "cpu"
log.info(f"Running model on device {dev}.")

from dataset import MARCODataset

# Create datasets
log.info("Loading datasets...")
# ntrain = args.ntrain
# ntest = int(0.1 * ntrain)

# training_dataset = MARCODataset(training_file, use_complex_transform=True, maximages=ntrain, dev=dev)
testing_dataset = MARCODataset(testing_file, use_complex_transform=False, dev=dev)


# Create DataLoaders
log.info("Loading dataloader...")
test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=1)


log.info(f"Loading ResNet{args.resnet}")

# Load the specified ResNet model
if args.resnet == 18:
    net = models.resnet18(pretrained=True)
elif args.resnet == 34:
    net = models.resnet34(pretrained=True)
elif args.resnet == 50:
    net = models.resnet50(pretrained=True)

# Adding a dropout before the final linear layer and a new linear layer sequence
net.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # Dropout with a probability of 0.5
    nn.Linear(net.fc.in_features, 1000),  # First linear layer
    nn.ReLU(),  # Activation function
    nn.Linear(1000, 300),  # Second linear layer
    nn.ReLU(),  # Activation function
    nn.Linear(300, 4)  # Final layer for 4 classes
)
net = net.to(dev)
net.load_state_dict(torch.load("../model_epoch_300.net", weights_only=True))

log.info("Optimizer...")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
# if args.adam:
#   optimizer = optim.Adam(net.parameters(), lr=args.lr)

log.info("Script local vars:")
log.info(globals())
for epoch in range(1):  # loop over onceover dataset

    s=torch.nn.Softmax(dim=1)

    # Evaluate on the test set
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    #from IPython import embed
    #embed()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            #from IPython import embed
            #embed()
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            probs=s(outputs)
            

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            p = probs[0][predicted[0].item()]
            #from IPython import embed
            #embed()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            lossi = loss.item() 
            test_loss += lossi
            log.info(f'Image Number{i}; {predicted[0].item()}; Probs; {p}') #format to include the predictions and probab. (Predicted with )
            #if i % 5 == 0:
            #    log.info(f'Batch {i + 1}/{len(test_loader)}] loss: {lossi:.3f}')
        
        accuracy = 100 * correct / total
        log.info(f'Done with evaluation; test loss= {test_loss/len(test_loader):.6f}; accuracy= {accuracy:.2f}%')
        
        
