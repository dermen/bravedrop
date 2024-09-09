import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


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

