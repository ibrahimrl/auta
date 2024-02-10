

# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torch.nn.functional as F
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
    return model_ft
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

def forward( model, img, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.3750, 0.3984, 0.4250], [0.2682, 0.2669, 0.2715])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.3750, 0.3984, 0.4250], [0.2682, 0.2669, 0.2715])]),}

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # Convert to Pillow image
    img = Image.fromarray(rgb_image)

    transformed_img = data_transforms['val'](img)

    model.eval()
    transformed_img = transformed_img.to(device)
    y_pred = model(transformed_img.unsqueeze(0))
    y_prob = F.softmax(y_pred, dim = -1)
    top_pred = y_prob.argmax(1, keepdim = True)
    prob = round(y_prob.max().item(), 2)
    print( class_names[top_pred], prob)

    return class_names[top_pred]


# make_model = load_model('Weights/MakeModel_Rec.pt')
# car_names = read_class_names('Weights/Class_Names.txt')
# print(car_names)
# car_make_model = forward(make_model,cv2.imread('data/input_images/00:46:35_0d5253a5-8233-4073-aff7-06dfbd6186b6.jpg'),car_names)
# print(car_make_model)
