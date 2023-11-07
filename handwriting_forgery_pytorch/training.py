import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# import pandas as pd
import numpy
# import csv
import cv2
import numpy as np
from PIL import Image

from creating_dataset import CreateDatasetString

# self.data = ["dataset_photos/Screen Shot 2023-10-27 at 19.57.01.png"]  # Replace with entire dataset # Image paths
# self.label = ["value"]

class CustomDataset(Dataset):
    # Load image paths and labels from the database
    def __init__(self, dataset_data, dataset_labels):
        self.data = dataset_data
        self.labels = dataset_labels

    def __len__(self): # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and preprocess the image at index idx
        img_size = (100, 100)  # width, height
        # Img Object
        img = cv2.imread(self.data[idx])
        print(self.data[idx])
        # Gray Scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        img = cv2.resize(img, img_size)  # Recall that img_size was a tuple
        num_array = np.array(img)
        # print(num_array)
        # (num_array, label) tuple
        print("Numpy Array Size:", num_array.shape)

        #---------- Transformation -------------------------------------------------------------
        image_PIL = Image.fromarray(num_array) # Convert to PIL format
        custom_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.0)),
            # transforms.ToTensor(),
        ])
        image_PIL = custom_transforms(image_PIL) # Add the transformation to PIL
        num_array = np.array(image_PIL) # Convert PIL to np.array
        # image_PIL.show() # Display image
        # -------------------------------------------------------------------------------------
        num_array = num_array.reshape(-1) # Reshape to a 1D array
        print("Processed Num Array Shape: ", num_array) # Convert to 1D array
        return num_array, self.labels[idx]
        # Return image and label


# Data will have a name with an associated label
data, labels = CreateDatasetString()

# Create DataLoader
dataset = CustomDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True) # returns two arrays: array of 100x100 and labels

# training_dataset, labels = dataset[0]
# print(training_dataset)
# print(labels)
# <<<<<<<<<<<<<<<<<<<<< Training >>>>>>>>>>>>>>>>>>>>> #

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Order does not matter. Only stating what layers will the model have.
        # "hl" means hidden layer
        # nn.{type of function(input_size, hidden_size, bias=True, dtype=torch.float)
        self.layer1 = nn.Linear(10000, 100)
        self.layer2 = nn.Linear(100, 1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, x):  # x is the batch of inputs
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        print("tanush", x.shape)
        return torch.sigmoid((x.view(-1))) # Keeps between 0, 1


# Loss function for binary classification
model = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(4):
    for inputs, target_labels in data_loader:
        print("Input:\n", inputs)
        print("Labels:\n", target_labels)
        optimizer.zero_grad()
        outputs = model(inputs.to(torch.float)) # .to(torch.float) converts inputs to floats
        print("success")
        loss = criterion(outputs.to(torch.float), target_labels.to(torch.float))
        loss.backward()
        optimizer.step()
        print("Epoch cycle complete", epoch)

# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         # Input #, output #, kernel (nxn)
#         self.conv1 = nn.Conv2d(1, 30, (3, 3))
#         self.conv2 = nn.Conv2d(30, 15, (3, 3))
#         self.conv3 = nn.Conv2d(30, 15, (3, 3))
#
#         # (Input, output). Fully connected
#         self.fc1 = nn.Linear(3**3*10, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#
#     def forward(self, x):
#         x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = f.max_pool2d(f.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
#         x = f.relu(self.fc1(x))
#         x = f.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# detector = Network().to('cpu')
# optimizer = Adam(detector.parameters(), lr=1e-3)  # learning rate
# loss_function = nn.CrossEntropyLoss()
#
# tx, ty = data_loader


