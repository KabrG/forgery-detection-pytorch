import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# import pandas as pd
import numpy
# import csv
import cv2
import numpy as np
from PIL import Image

# self.data = ["dataset_photos/Screen Shot 2023-10-27 at 19.57.01.png"]  # Replace with entire dataset # Image paths
# self.label = ["value"]

custom_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=(200, 200), scale=(0.9, 1.0)),
    # transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # Load image paths and labels from the database
        # Implement __len__ and __getitem__ methods

    def __len__(self): # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and preprocess the image at index idx
        img_size = (100, 100)  # width, height
        # Img Object
        img = cv2.imread(self.data[idx])
        # Gray Scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        img = cv2.resize(img, img_size)  # Recall that img_size was a tuple
        num_array = np.array(img)
        print(num_array)
        # (num_array, label) tuple
        print(num_array.shape)
        #---------- Transformation -------------------------------------------------------------
        image_PIL = Image.fromarray(num_array) # Convert to PIL format
        image_PIL = custom_transforms(image_PIL) # Add the transformation to PIL
        num_array = np.array(image_PIL) # Convert PIL to np.array
        image_PIL.show()
        # -------------------------------------------------------------------------------------
        return num_array, self.labels[idx]
        # Return image and label

data = ["handwriting_dataset/authentic_signatures/Screen Shot 2023-10-28 at 18.53.43.png"]
labels = [0]

# Create DataLoader
dataset = CustomDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True) # returns two arrays

training_dataset, labels = dataset[0]
print(training_dataset)
print(labels)
# <<<<<<<<<<<<<<<<<<<<< Training >>>>>>>>>>>>>>>>>>>>> #

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # Input layers, output layers, kernel (nxn)
        self.conv1 = nn.Conv2d(1, 30, (3, 3))
        self.conv2 = nn.Conv2d(30, 15, (3, 3))
        self.conv3 = nn.Conv2d(30, 15, (3, 3))

        # (Input, output). Fully connected
        self.fc1 = nn.Linear(3**3*10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 1 input image channel, 6 output channels, 5x5 square convolution

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


detector = Network().to('cpu')
optimizer = Adam(detector.parameters(), lr=1e-3)  # learning rate
loss_function = nn.CrossEntropyLoss()

tx, ty = data_loader


