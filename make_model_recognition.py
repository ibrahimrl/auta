


# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import cv2
cudnn.benchmark = True
plt.ion()   # interactive mode


def load_model(path):


    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, 63)

    model_ft.load_state_dict(torch.load(path,map_location=torch.device('cpu')))  # Change 'cpu' to 'cuda' if you're using GPU

def read_class_names(file_path):
    """
    Reads a text file and returns a list of lines, preserving the order.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        list: List of lines from the text file.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    
def forward(model, image, class_names):

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.3750, 0.3984, 0.4250], [0.2682, 0.2669, 0.2715])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.3750, 0.3984, 0.4250], [0.2682, 0.2669, 0.2715])
    ]),}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to Pillow image
    img = Image.fromarray(rgb_image)

    # Display the image
    images_so_far += 1

    # Preprocess the image and get the prediction
    img = data_transforms['val'](img).unsqueeze(0)  # Assuming transform is defined elsewhere
    # inputs = img.to(device)  # Assuming device is defined elsewhere
    outputs = model(img)
    _, preds = torch.max(outputs, 1)

    return class_names[preds.item()]