from PIL import Image
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_category_json(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_data(path = './flowers'):
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    # Transforms for training includes random rotations, flips and normalisations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Transforms for validation
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Transforms for testing
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    # datasets = 
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    
    return trainloader, validloader, testloader, train_data


def classifier(model, hidden_layer_size, architecture):
    for param in model.parameters():
        param.requires_grad = False

    start_size = {"vgg16":25088,"densenet121":1024}
        
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(start_size[architecture], hidden_layer_size)
            self.fc2 = nn.Linear(hidden_layer_size, 102)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = F.log_softmax(self.fc2(x), dim=1)
            return x
    
    return Classifier()



def save_checkpoint(model, save_path, architecture, hidden_layer_size, train_data):
    """
    Save the checkpoint for the model 
    """
    model.class_to_idx = train_data.class_to_idx
    
    # Transfer the cpu for saving
    model.to('cpu')
    
    checkpoint = {'architecture':architecture,
                  'state_dict': model.state_dict(), 
                  'class_to_idx': train_data.class_to_idx, 
                  'hidden_layer_size':hidden_layer_size}

    torch.save(checkpoint, save_path)
    
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    architecture = checkpoint['architecture']
    hidden_layer_size = checkpoint['hidden_layer_size']
    
    # Declare the model prior to the exec function as it isn't capable of doing so 
    # model = 0
    # Recreate the model using the checkpoint saves
    # exec('model = models.{}(pretrained = True)'.format(architecture))
    
    if architecture == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        model = models.vgg16(pretrained = True)
    
    # Make sure that the model is the right shape and size 
    model.classifier = classifier(model, hidden_layer_size, architecture)

    # Load the categories dictionary and model state 
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model 


    
def train_network(epochs, trainloader, validloader, optimiser, device, criterion, model):
    steps = 0
    print_every = 32
    print('Training start\n')
    model.to(device)
    # Loop through each epoch for training 
    for epoch in range(epochs):
        running_loss = 0

        # Go through the training batches 
        for inputs_1, labels_1 in trainloader:
            # Update the steps progress
            steps += 1

            # Move input and label tensors to the default device so they are available 
            inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)

            optimiser.zero_grad()

            outputs = model.forward(inputs_1)
            loss = criterion(outputs, labels_1)

            # Update the gradients 
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            # Only print every n steps 
            if steps % print_every == 0:
                # Turn off the dropout for the validation phase so that all inputs are used 
                model.eval()

                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs_2, labels_2 in validloader:
                        optimiser.zero_grad()

                        # Move the validation examples to the relevant device 
                        inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
                        model.to(device)

                        # Get the outputs from the model 
                        outputs_2 = model.forward(inputs_2)

                        # Calculate the loss
                        batch_loss = criterion(outputs_2, labels_2)
                        validation_loss += batch_loss.item()

                        # Find the probabilities 
                        ps = torch.exp(outputs_2)

                        # Get the top result
                        top_p, top_class = ps.topk(1, dim=1)

                        # Calculate accuracy
                        equals = top_class == labels_2.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                    print(f"Epoch: {epoch+1}/{epochs} - "
                        f"Train loss: {running_loss/print_every:.4f} - "
                        f"Validation loss: {validation_loss/len(validloader):.4f} - "
                        f"Validation accuracy: {accuracy/len(validloader):.4f}")

                    # Set the model back to training mode with dropout included for training segment
                    model.train()
                    # Reset running_loss
                    running_loss = 0

    # Note the end of the network training
    print('\nTraining end')
                        
                        
                        

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    
    image_processing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = image_processing(image)
    
    # Must return a numpy array so that it can be transposed for plotting 
    return image_tensor.numpy()


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, cat_to_name, topk = 5, device = 'cpu', gpu = False):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Set the model to evaluate 
    model.eval()
    model.to(device)
    
    if gpu:
        processed_image = torch.from_numpy(process_image(image_path)).type(torch.cuda.FloatTensor)
    else:
        processed_image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor)
    
    # Add batch dimensioning as it is missing with single image
    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    processed_image.unsqueeze_(0)
    
    # Feed forward through the model 
    output = model.forward(processed_image)
    probs = torch.exp(output)
    
    # Find the top k labels and their classes 
    top_probs, top_labels = probs.topk(topk, dim = 1)
    
    # Reverse the index_to_class from the checkpoint so that we can obtain the proper classes 
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Simplify the output to make it easier to work with in future 
    # Have to copy the tensor to host memory before conversion to numpy 
    # Get the labels for the classes using the idx_to_classes obtained from the checkpoint 
    labels = [idx_to_class[i] for i in np.array(top_labels)[0]]
    probs = [i for i in top_probs.cpu().detach().numpy()[0]]
    
    final_labels = [cat_to_name[i] for i in labels]
    return probs, final_labels 