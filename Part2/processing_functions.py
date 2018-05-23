# This Script handles all functions for image processing and input processing

import numpy as np
from PIL import Image
import glob, os
import torch
import argparse
import ast
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from os import listdir
import json

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()


    # Creates 4 command line arguments args.dir for path to checkpoint file,
    # args.arch which CNN model to use for classification, args.learning rate for the learning rate,
    # ars.hidden_units for the number of hidden layers and args.epochs for the number of epochs
    parser.add_argument('--dir', type=str, default='checkpoints/checkpoint.pt', 
                        help='path to folder for checkpoint saves')
    parser.add_argument('--arch', type=str, default='densenet121', 
                        help='chose model from densenet121, alexnet')
    parser.add_argument('--learning_rate', type=float, default='0.01',
                        help='put in learning rate(float)')
    parser.add_argument('--hidden_units', type=int, default='512',
                        help='put in the number of hidden units(integer)')
    parser.add_argument('--epochs', type=int, default='2',
                        help='put in the number of epochs(integer)')
    parser.add_argument('--gpu', type=str, default='gpu',
                         help='use gpu for training with gpu and '' for training with cpu')
    parser.add_argument('--topk', type=int, default='5',
                         help='insert topk classes(integer)')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                         help='insert a mapping of categories')
    
    # returns parsed argument collection
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array ''' 
    global img

    mean = [0.485, 0.456, 0.406] 
    stdv = [0.229, 0.224, 0.225] 
    img = Image.open(image)
    if img.size[0]>=img.size[1]: 
        img.thumbnail((10000,256)) 
    else: 
        img.thumbnail((256,10000))

    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img = img.crop(
        (
            half_the_width - 112,
            half_the_height - 112,
            half_the_width + 112,
            half_the_height + 112
        )
    )

    np_image = np.array(img)
    img = np_image/255
    img=(img-mean)/stdv

    img=img.transpose((2,0,1))
    
    return img  

def predict(image_path, model, topk, image_datasets_training, cat_to_name, cuda):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Feeding Image to the predict function
    processed_img = process_image(image_path)
    y = np.expand_dims(processed_img, axis=0)
    img = torch.from_numpy(y)
    
    if cuda == 'gpu':
        # Move model parameters to the GPU
        model.cuda()
    else:
        model.cpu()
    
    if cuda == 'gpu':
        img = torch.from_numpy(y).cuda()
    
    #arranging model.class_to_idx to get right classes
    xx = image_datasets_training.class_to_idx
    res = dict((v,k) for k,v in xx.items()) 
    model.class_to_idx = res

    inputs = Variable(img, volatile=True).float()
    if cuda == 'gpu':
        # Move input tensors to the GPU
        inputs = inputs.cuda()

    #predicting class from picture
    model = model.eval()
    output = model.forward(inputs)
    ps = torch.exp(output)
    top_probs, top_labels = ps.topk(topk)

    #getting classes from top_k to the class name with the right probability connection
    top_probs = top_probs.data.cpu().numpy().tolist()
    top_labels = top_labels.data.cpu().numpy().tolist()

    class_to_idx = []
    for i in range(topk):
        class_to_idx.append(model.class_to_idx[top_labels[0][i]])

    cat_to_name_topk = []
    for i in range(topk):
        cat_to_name_topk.append(cat_to_name[class_to_idx[i][0]])
    return (top_probs, cat_to_name_topk)

def printing_results(top_probs, cat_to_name_topk):
    print(top_probs)
    print(cat_to_name_topk)
    
    return None
                     
def catToName(cat_to_name_file):
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name