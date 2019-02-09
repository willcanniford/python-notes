import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import utils


a = argparse.ArgumentParser(description='train.py')
a.add_argument('data_path', action="store", default="./flowers/")
a.add_argument('--save_dir', dest="save_path", action="store", default="checkpoint.pth")
a.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
a.add_argument('--hidden_layer', dest="hidden_layer", action="store", type=int, default=256)
a.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.001)
a.add_argument('--architecture', dest="architecture", action="store", default='densenet121', choices=['densenet121', 'vgg16'])
a.add_argument('--gpu', dest="gpu", action="store_true", default=True)


arguments = a.parse_args()
data_path = arguments.data_path
epochs = arguments.epochs
learning_rate = arguments.learning_rate
save_path = arguments.save_path
gpu = arguments.gpu
hidden_layer = arguments.hidden_layer

# Define the architecture of the model 
if type(arguments.architecture) == type(None):
    architecture = 'densenet121'
else:
    architecture = arguments.architecture 


# Load the flower category dictionary
cat_to_name = utils.load_category_json('cat_to_name.json')

# Load the data in using the data path provided
trainloader, validloader, testloader, train_data = utils.load_data(data_path)

# Build the model ready for training 
exec('model = models.{}(pretrained = True)'.format(architecture))
model.classifier = utils.classifier(model, hidden_layer, architecture)

# Define the loss criteria and optimiser for training
criterion = nn.NLLLoss()
optimiser = optim.Adam(model.classifier.parameters(), lr = learning_rate)

# Create a device using the terminal input 
device = torch.device("cuda" if gpu else "cpu")

# Train the model 
utils.train_network(epochs, trainloader, validloader, optimiser, device, criterion, model)

# Save the model information
utils.save_checkpoint(model, save_path, architecture, hidden_layer, train_data)
