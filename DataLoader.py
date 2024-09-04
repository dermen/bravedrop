#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser
import os
import pandas as pd
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim

# Argument parsing
pa = ArgumentParser()
pa.add_argument("logfile", type=str, help="path to a log file")
pa.add_argument("--ntrain", type=int, help="Number of training images to load", default=1e5)
pa.add_argument("--lr", type=float, default=0.001, help="learning rate")
pa.add_argument("--cpu", action="store_true", help="Run training on the CPU (slow)")
pa.add_argument("--devID", type=int, default=1, help="GPU device Id")
pa.add_argument("--bs", type=int, help="batch size", default=40)
pa.add_argument("--nwork", type=int, default=10, help="number of data loader workers")
pa.add_argument("--adam", action="store_true")
pa.add_argument("--resnet", type=int, choices=[18, 34, 50], default=18, help="ResNet architecture to use (18, 34, or 50)")
pa.add_argument("--savepath", type=str, required=True, help="Path to save the model's state after each epoch")
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
training_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/train_out/info.csv'
testing_file = '/mnt/data/ns1/brave/MARCO/marco.ccr.buffalo.edu/data/archive/test_out/info.csv'

dev = 'cuda:%d' % args.devID
if args.cpu:
    dev = "cpu"
log.info(f"Running model on device {dev}.")

# Define the MARCODataset class
class MARCODataset(Dataset):
    def __init__(self, annotations_file, use_complex_transform=True, target_transform=None, maximages=-1, dev="cpu"):
        if maximages == -1:
            maximages = None
        self.img_data = pd.read_csv(annotations_file, nrows=maximages)
        self.target_transform = target_transform
        self.dev = dev
        self.use_complex_transform = use_complex_transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data.iloc[idx, 0]
        label = self.img_data.iloc[idx, 2]  # Assuming label_id is the third column
        image = Image.open(img_path)
        
        if self.use_complex_transform:
            transform = self.complex_preprocess()
        else:
            transform = self.simple_preprocess()

        image = transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

    def simple_preprocess(self):
        transform = transforms.Compose([
            transforms.Resize((600, 600)),  # Resize to 600 x 600 or any consistent size
            transforms.ToTensor()  # Convert image to tensor
        ])
        return transform

    def complex_preprocess(self):
        transform = transforms.Compose([
            transforms.Resize((600, 600)),  # Resize to 600 x 600 or any consistent size
            transforms.RandomRotation(degrees=90),  # Random rotation
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),  # Random vertical flip
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Random Gaussian blur
            transforms.ToTensor()  # Convert image to tensor
        ])
        return transform

# Create datasets
log.info("Loading datasets...")
ntrain = args.ntrain
ntest = int(0.1 * ntrain)

training_dataset = MARCODataset(training_file, use_complex_transform=True, maximages=ntrain, dev=dev)
testing_dataset = MARCODataset(testing_file, use_complex_transform=True, maximages=ntest, dev=dev)
log.info("Doing %d train images and %d test images" % (ntrain, ntest))

# Create DataLoaders
log.info("Loading dataloader...")
train_loader = DataLoader(training_dataset, batch_size=args.bs, shuffle=True, num_workers=args.nwork)
test_loader = DataLoader(testing_dataset, batch_size=args.bs, shuffle=True, num_workers=args.nwork)

tag = "AdamOpt" if args.adam else "SGDOpt"
# Output folder for saving images
output_folder_root = "/mnt/data/ns1/brave/MARCO/MS/savedmodelfolder"
output_folder = os.path.join(output_folder_root, tag)

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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log info statistics
        lossi = loss.item()
        train_loss += lossi
        if i % 5 == 0:
            log.info(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] loss: {lossi:.3f}')           
    log.info(f'Done with epoch {epoch + 1}; train loss= {train_loss/len(train_loader):.6f}')
    
    # Save the model's state as a .net file
    epoch_save_path = os.path.join(args.savepath, f'model_epoch_{epoch + 1}.net')
    torch.save(net.state_dict(), epoch_save_path)
    log.info(f'Model saved at {epoch_save_path}')
    
    # Evaluate on the test set
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            lossi = loss.item() 
            test_loss += lossi
            if i % 5 == 0:
                log.info(f'Epoch {epoch + 1}, Batch {i + 1}/{len(test_loader)}] loss: {lossi:.3f}')
        
        accuracy = 100 * correct / total
        log.info(f'Done with epoch {epoch + 1}; test loss= {test_loss/len(test_loader):.6f}; accuracy= {accuracy:.2f}%')
            
log.info('Finished Training')
