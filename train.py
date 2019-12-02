import time
import json
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np
import PIL

from PIL import Image
import classifier
from collections import OrderedDict
from classifier import Network

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

import os

densenet = models.densenet121(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
densenet161 = models.densenet161(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
vgg13 = models.vgg13(pretrained=True)
vgg19 = models.vgg19(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16, 'densenet': densenet, 'vgg13': vgg13, 'vgg16':  vgg16, 'vgg19': vgg19, 'densenet161': densenet161, 'densenet121': densenet121
         }

def enable_gpu(gpu_on):
    if not gpu_on:
        return False
    else:
        return True


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Positional argument: path to a folder
    parser.add_argument('data_dir', type=str,
                        help='path to the folder flowers')

    # Optional argument: the architecture
    parser.add_argument('--arch', type=str, default='densenet',
                        help='CNN model architecture to use for image classification')
    
    # Optional argument: the hidden_units
    parser.add_argument('--hidden_units', type=str,
                        help='The values of hidden units')
    # Optional argument: the directory where to save the trained model
    parser.add_argument('--save_dir', type=str, default='./',
                        help='directory where to save the trained model')
    # Optional argument: enable gpu mode
    parser.add_argument('--gpu', action='store_true',
                        help='Enable gpu mode')
    # Optional argument: the number of epochs
    parser.add_argument('--epochs', type=int, default=20,
                        help='The number of epochs')
    # Optional argument: the learning rate
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate')
    # Optional argument: the batch size
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size')
    # Optional argument: the output size
    parser.add_argument('--output_size', type=int, default=102,
                        help='The output size')

    in_args = parser.parse_args()
    
    print("Arguments 1: ", in_args.data_dir)

    return in_args

in_arg = get_input_args()
gpu_enabled = enable_gpu(in_arg.gpu)
data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
batch_size = in_arg.batch_size

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = data_transforms = data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
}

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'valid', 'test']}
images, labels = next(iter(dataloaders['train']))
images.size()

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
model = in_arg.arch

in_features = None
hidden_layers = None
hidden_units = in_arg.hidden_units
if hidden_units != None:
    hidden_layers = [int(s) for s in hidden_units.split(',')]
if model == 'densenet' or model == 'densenet121':
    in_features = 1024
    if hidden_layers == None:
        hidden_layers = [1000]
elif model == 'densenet161':
    in_features = 2208
    if hidden_layers == None:
        hidden_layers = [1000]
elif model == 'resnet' or model == 'resnet18':
    in_features = 512
    if hidden_layers == None:
        hidden_layers = [1000]
elif model == 'alexnet':
    in_features = 9216
    if hidden_layers == None:
        hidden_layers = [4096]
elif model == 'vgg' or model == 'vgg13' or model == 'vgg16' or model == 'vgg19':
    in_features = 25088
    if hidden_layers == None:
        hidden_layers = [4096]
else:
    print("Unknown model, please choose from the following: \ndensenet, densenet121, densenet161, vgg, resnet, alexnet, vgg13, vgg16, vgg19, alexnet")

model = models[model]

output_size = in_arg.output_size
classifier = None
if hidden_units == None:
    # Build and train your network
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, hidden_layers[0])),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
else:
    classifier = Network(input_size = in_features,
                     output_size = output_size,
                     hidden_layers = hidden_layers,
                     drop_p = 0.2)
print(classifier)


for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier
criterion = nn.NLLLoss()


# Only train the classifier parameters, feature parameters are frozen
lr = in_arg.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Adapted from https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/
def train_model(model, criterion, optimizer, sched, epochs=5):
    ''' Trains a neural network model.
    
    Parameters:
        model: the neural network model
        criterion: loss criterion (cost function)
        optimizer: optimizer to be used for updating the weights
        sched: the learning_rate scheduler
        epochs: number of epochs for training
        
   Returns:
            model - the trained neural network model
    '''
       
    since = time.time()

    most_acc_params = copy.deepcopy(model.state_dict())
    most_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > most_acc:
                most_acc = epoch_acc
                most_acc_params = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s:::::::::::::::::::::::::::::::::::::'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy value obtained: {:4f}::::::::::::::::::::::::::::::::::::::'.format(most_acc))

    #load the most accurate model weights
    model.load_state_dict(most_acc_params)
    
    return model

    
device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_enabled == True
                      else "cpu")

epochs = in_arg.epochs
sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)
model.to(device)
print('GPU STATUS ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::, gpu_enabled')
model = train_model(model, criterion, optimizer, sched, epochs)

# Do validation on the test set

model.eval()

accuracy = 0

for inputs, labels in dataloaders['test']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model.forward(inputs)
    
    # Class with the highest probability is our predicted class
    equality = (labels.data == outputs.max(1)[1])

    # Accuracy is number of correct predictions divided by all predictions
    accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    
print("Test Accuracy: {:.3f}:::::::::::::::::::::::::::;".format(accuracy/len(dataloaders['test'])))


filepath = in_arg.save_dir + 'checkpoint.pth'

model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'input_size': in_features,
              'output_size': output_size,
              'epochs': epochs,
              'batch_size': batch_size,
              'model': models[in_arg.arch],
              'classifier': classifier,
              'scheduler': sched,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
   
torch.save(checkpoint, filepath)