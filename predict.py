import argparse
import torch
import json
import matplotlib.pyplot as plt
from classifier import Network
from PIL import Image
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys


def load_checkpoint(filepath):
    """
    This function Loads the trained model.
    
    Parameters:
        filepath - the path were the model is saved.
        
    Returns:
        model - the pretrained model
    """
    
    # load checkpoint and rebuild the model
    print("Loading Pretrained Model:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    checkpoint = torch.load(filepath)
    checkpoint.keys()

    model = checkpoint['model']
    print(f"Model of Type {model} Sucessfully Loaded::::::::::::::::::::::::::::::::::::::::::::::: \n")
    
    #Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False
    
   
    classifier =  checkpoint['classifier']
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict']) 
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
        Parameters:
            image: the image to be processed
        
    Returns: 
        image - the processed image
    '''
    image = Image.open(image)
    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image


def predict(image_path, model, device, top_k=5, category_names=None):
    ''' Predicts the class (or classes) of an image using a pretrained neural network model.
    
    Parameters:
        image_path - the image path
        model - the pretrained model to be used to predict the class of the iamge
        device - the device ('cpu or 'cuda')
        topk - the top_k classes
        category_names - option for printing category names
    
    Returns: 
        probs, names, classes - the results of the prediction: predicted probablilities, category names and classes
        
    '''
    image = process_image(image_path)
    image = np.expand_dims(image, 0)
    
    image = torch.from_numpy(image)
    
    try:
        model.to(device).float()
    except AssertionError as e:
        print(e)
        sys.exit(1)
    
    model.eval()

    inputs = Variable(image).to(device)
    logits = model.forward(inputs)
    ps = F.softmax(logits,dim=1)
    top_k = ps.cpu().topk(top_k)

    probs, categories = (e.data.numpy().squeeze().tolist() for e in top_k)
  
    classes = []
    if type(categories) == int: # this case is not iterable
        classes.append(list(model.class_to_idx)[categories])
    else:
        for e in categories:
            classes.append(list(model.class_to_idx)[e])

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
       
        names = []
        
        for e in classes:
            names.append(cat_to_name[e])
    else:
        names = None
    return probs, names, classes


if __name__ == '__main__':
   
    
    parser = argparse.ArgumentParser()

    # Positional argument: path to a folder
    parser.add_argument('image_path', type=str,
                        help='path to the flower image')

    # Optional argument: the architecture
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth',
                        help='Path to the trained model checkpoint')
    
    # Optional argument: the hidden_units
    parser.add_argument('--top_k', type=int,
                        help='The K Most Probable Classes')
    # Optional argument: the directory where to save the trained model
    parser.add_argument('--category_names', type=str,
                        help='Explicit Predicted Flowers Classes Names (json input)')

    parser.add_argument('--gpu', action='store_true',
                        help='Enable gpu mode')
    
    in_arg = parser.parse_args()
    
   
    model = load_checkpoint(in_arg.checkpoint)
    
    if not in_arg.gpu:
        device = 'cpu'
    else:
        device = 'cuda'
    
    image_path = in_arg.image_path
    
    if not in_arg.top_k:
        top_k = 6
    else:
        top_k = in_arg.top_k 
        
    probs, names, classes = predict(image_path=image_path,
                                          model=model,
                                          device=device,
                                          top_k=top_k,
                                          category_names = in_arg.category_names)
                                          
    
    if not in_arg.category_names:
        print(f"Prediction: Flower of Class {classes} with {probs} probability")
    else:
        print(f"Prediction: Flower of Class {names} with {probs} probability")
        