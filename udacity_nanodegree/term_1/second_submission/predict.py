import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import utils

a = argparse.ArgumentParser(description='predict.py')
a.add_argument('image_path', action="store", default = "./flowers/test/1/image_06743.jpg")
a.add_argument('--checkpoint_path', dest="checkpoint_path", action="store", default='checkpoint.pth')
a.add_argument('--top_k', dest="top_k", action="store", type=int, default=5)
a.add_argument('--gpu', dest="gpu", action="store", default=True)
a.add_argument('--json_label_file', dest="json_label_file", action="store_true")

arguments = a.parse_args()
image_path = arguments.image_path
checkpoint_path = arguments.checkpoint_path
top_k = arguments.top_k
gpu = arguments.gpu
json_label_file = arguments.json_label_file

device = torch.device("cuda" if gpu else "cpu")

# Load and recreate the trained model using the load_checkpoint utils function
model = utils.load_checkpoint(checkpoint_path)

cat_to_name = utils.load_category_json('cat_to_name.json')

# Make a prediction to a given file path
probs, labels = utils.predict(image_path, model, cat_to_name, top_k, device, gpu)

print('Probabilites: {}'.format(probs))
print('Labels: {}'.format(labels))