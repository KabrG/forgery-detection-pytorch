import sys
import time

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
from PIL import Image, ImageFilter

from creating_dataset import create_dataset_string

from playsound import playsound

# Ringtone plays three times once training is done
def ringtone():
    for i in range(3):
        playsound('/Users/kabirguron/Documents/forgery-detection-pytorch/handwriting_forgery_pytorch/notification_sound.mp3')
        time.sleep(1)

def save_model(model, optimizer):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'epoch': epoch,
    }
    torch.save(checkpoint, 'model_checkpoint.pth')


def load_model(model, optimizer):
    checkpoint = torch.load('model_checkpoint.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


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
        # print(self.data[idx])
        # Gray Scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        img = cv2.resize(img, img_size)  # Recall that img_size was a tuple
        num_array = np.array(img)
        # print(num_array)
        # (num_array, label) tuple
        # print("Numpy Array Size:", num_array.shape)

        # ---------- Transformation -------------------------------------------------------------
        image_PIL = Image.fromarray(num_array) # Convert to PIL format
        custom_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=7),
            transforms.RandomResizedCrop(size=img_size, scale=(0.92, 1.0)),
            # transforms.ToTensor(),
        ])
        # Add the transformation to PIL
        image_PIL = custom_transforms(image_PIL)
        # Filter image according to edge detection kernel
        image_PIL = image_PIL.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1, -1, 9.4, -1, -1, -1, -1],
            scale=1
        ))
        num_array = np.array(image_PIL) # Convert PIL to np.array
        # image_PIL.show() # Display image
        # sys.exit()
        # -------------------------------------------------------------------------------------
        # num_array = num_array.reshape(-1) # Reshape to a 1D array
        # print("Processed Num Array Shape: ", num_array.shape) # Convert to 1D array
        num_array = num_array / 255.0
        # tensor_array = torch.from_numpy(num_array)
        # tensor_array = tensor_array.view(1, 1, 100, 100)  # 1 batch, 1 channel, 100x100 image
        # print(num_array.shape)
        # print(num_array/255.0)
        return num_array, self.labels[idx]
        # Return image and label

# <<<<<<<<<<<<<<<<<<<<< Training >>>>>>>>>>>>>>>>>>>>> #

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Order does not matter. Only stating what layers will the model have.
        # "hl" means hidden layer
        # nn.{type of function(input_size, hidden_size, bias=True, dtype=torch.float)

        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Convolutional Layers
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # 32 different "filters", feature maps
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # Linear Layers (Fully Connected Layers)
        self.fcl_1 = nn.Linear(589824, 128) # x.size() to obtain 589824
        self.fcl_2 = nn.Linear(128, 1) # int(128*3/8)

        # Dropout
        self.dropout1 = nn.Dropout(0.5)  # 0.2 is the dropout probability (adjust as needed)
        self.dropout2 = nn.Dropout(0.1)  # 0.2 is the dropout probability (adjust as needed)



    def forward(self, x):  # x is the batch of inputs
        # x = x.view(x.size(0), -1)  # Flatten the input if it's not already
        x = x.view(x.size(0), 1, 100, 100)  # Reshape input to [batch_size , channels, height, width]

        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1) # Flatten to 1D Array

        x = self.fcl_1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fcl_2(x)
        x = self.sigmoid(x)
        # print(x.shape)
        # print(torch.sigmoid((x.view(-1))))
        # print(x.view(-1))
        # print(x)
        return x.view(-1)

# https://www.youtube.com/watch?v=SDPeeX6LEnk (Reference)
def train(data_loader, model, loss_function, optimizer):
    model.train() # Setting to training mode
    train_loss = 0
    num_of_batches = len(data_loader)

    # (x, y) is (count, element) respectively
    for i, (input_data, target) in enumerate(data_loader):  # X is input, y is actual result.
        input_data, target = input_data.to('cpu').to(torch.float), target.to('cpu').to(torch.float)

        pred_y = model(input_data)

        # mse is "Mean Square Error"
        bse = loss_function(pred_y, target)
        # .item() is necessary because it was originally a tensor, so we need scalar
        print('loss item: ', bse.item())
        train_loss += bse.item()
        optimizer.zero_grad()
        bse.backward()
        optimizer.step() # Update parameters
        # Divide Mean square error
    print(f'Number of training batches: {num_of_batches}')
    train_bse = train_loss / num_of_batches
    print(train_bse**(1/2))

def test(data_loader, model, loss_function=None):
    model.eval()  # Setting to evaluation mode
    train_loss = 0

    for input_data, target in data_loader:
        input_data, target = input_data.to('cpu').to(torch.float), target.to('cpu').to(torch.float)
        pred_y = model(input_data)
        if loss_function is not None:
            train_loss += loss_function(pred_y, target).item()
        # print(f'Predicted value: {pred_y}\nActual Value: {target}')

    num_of_batches = len(data_loader)
    print(f'Number of testing batches: {num_of_batches}')
    test_bse = train_loss / num_of_batches
    print(test_bse ** (1/2))
    try:
        return pred_y
    except Exception:
        print("Could not return pred_y")


def run_training_program():
    # Data will have a name with an associated label
    data, labels = create_dataset_string()

    # Create DataLoader
    dataset = CustomDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True) # returns two arrays: array of 100x100 and labels

    # Loss function for binary classification
    model = Net().to('cpu')
    print(model)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # lr is learning rate

    if input("Do you wish to load old model? 'y' for yes.\n").lower() == 'y':
        load_model(model, optimizer)
        print("Model loaded.")

    for i in range(100):
        train(data_loader, model, loss_function, optimizer)
        test(data_loader, model, loss_function)
        print("EPOCH", i+1)

    ringtone() # Alert when training has finished

    if input("Do you want to save the current model? 'y' for save.\n").lower() == 'y':
        save_model(model, optimizer)
        print("Model saved.")

run_training_program()

# model = Net().to('cpu')
# print(model)
# loss_function = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # lr is learning rate




