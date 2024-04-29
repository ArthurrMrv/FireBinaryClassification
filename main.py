import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) < 4:
        print("Usage: python main.py <model_type (1: 64x64 |2: 128x128)> <data_path> <model_path>")
        sys.exit(1)
        return
    else:
        model_type = int(sys.argv[1])
        data_path = sys.argv[2]
        model_path = sys.argv[3]
        data_width = 64 if model_type == 1 else 128
        
    # Define transformation to be applied to images
    transform = transforms.Compose([
        transforms.Resize((data_width, data_width)),  # Resize images to desired format
        transforms.ToTensor()          # Convert images to PyTorch tensors
    ])
    
    # 1. Load model
    model = CNN1() if model_type == 1 else CNN2()
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print("Model not loaded")
        return
    
    return try_image(data_path, transform, model, bool(sys.argv[4]) if len(sys.argv) > 4 else True)
    
    
# Define custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
    
class CNN1(nn.Module):
    
    def __init__(self):
        super(CNN1, self).__init__()
        # Convolutional layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer with 2 classes
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        

    def forward(self, x):
        
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 16 * 8 * 8)
        
        # Fully connected layers with ReLU activation and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        
        # Output layer with softmax activation
        x = F.softmax(self.fc3(x), dim=1)
        
        return x

class CNN2(nn.Module):
    
    def __init__(self):
        super(CNN2, self).__init__()
        # Convolutional layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)  # Output layer with 2 classes
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3) #We need to avoind overfiting
        

    def forward(self, x):
        
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 16 * 16 * 16)
        
        # Fully connected layers with ReLU activation and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # Output layer with softmax activation
        x = F.softmax(self.fc4(x), dim=1)
        
        return x

def try_image(image_path, transformation, model, *, show=True):
    
    #Image has to be transformed to a 64x64 image
    img = Image.open(image_path)
    img = transformation(img)
    
    output = [model(img.unsqueeze(0)).argmax(dim=1).item()]
    if show:
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"{image_path}")
        print("Prediction:", ["Fire", "No Fire"][model(img.unsqueeze(0)).argmax(dim=1).item()])
    return output

if __name__ == "__main__":
    main()