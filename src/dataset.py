import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.datasets import VOCDetection

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        # For reconstruction, target is the image itself
        return image, image

def get_transforms(image_size=(128, 128)): # Made image_size a parameter
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1] for Tanh output
    ])

class VOCReconstructionDataset(VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=True, transform=None):
        super().__init__(root=root, year=year, image_set=image_set, download=download, transform=transform)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index) # We only need the image, discard annotations
        # The transform is already applied by VOCDetection if passed to its constructor.
        # For reconstruction, the input and target are the same.
        return img, img