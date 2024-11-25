import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import numpy as np
import cv2

#to utilize apple silicon chips, i used 'mps'
#if you are not using it on an apple silicon machine, use 'cuda'
device = torch.device("mps" if torch.mps.is_available()  else "cpu")
print(f'Using device: {device}')

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) 
model = model.to(device)

state_dict = torch.load("models/catdog.pth", map_location=device,weights_only=True)
model.load_state_dict(state_dict)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
