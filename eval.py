from dataset import MARCODataset

info_file = "/path/to/new/csv"
dset = MARCODataset(info_file)
dataloader = ... # follow whats in DataLoader.py

# load the torch model

for img,_ in dataloader:
    
    predicted = model(img)



