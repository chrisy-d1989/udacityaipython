#this script handles all functions neccessaray for the training part

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import glob, os
import argparse
from os import listdir
import json

def set_data_dir(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    return (train_dir,valid_dir,test_dir)
  
def  load_data(train_dir, valid_dir, test_dir):
  	# Setting Transformations
    data_transforms_training = transforms.Compose([transforms.RandomRotation(25),
                                              	   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomResizedCrop(224),
                                               	   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],                                               
                                                                      [0.229, 0.224, 0.225])])

    data_transforms_validation = transforms.Compose([transforms.RandomResizedCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                                          [0.229, 0.224, 0.225])])

    data_transforms_testing = transforms.Compose([transforms.RandomResizedCrop(224),
                                              	  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                                       [0.229, 0.224, 0.225])])

	# Load the datasets with ImageFolder
    image_datasets_training = datasets.ImageFolder(train_dir, transform=data_transforms_training)
    image_datasets_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
    image_datasets_testing  = datasets.ImageFolder(test_dir, transform=data_transforms_testing)

	# Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets_training, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(image_datasets_validation, batch_size=32)
    testloader = torch.utils.data.DataLoader(image_datasets_testing, batch_size=32) 
  
    return (trainloader,validationloader,testloader, image_datasets_training)

def load_model(model_from_input):
    # Loading resnet18/alexnet/densenet121 Model
    if model_from_input == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_from_input == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    return model

def define_classifier(model, hidden_layers, architecture):
  	# Freeze parameters so we don't backprop through them
    for param in model.parameters():
         param.requires_grad = False
	
    #Create Classifier for densenet121
    if architecture == "densenet121":
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, hidden_layers)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

        model.classifier = classifier
    
    else:
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, hidden_layers)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

        model.classifier = classifier

    return (classifier, param)
  
def training_network(model,classifier,trainloader,validationloader, cuda, epochs):

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        
        if cuda == 'gpu':
            # Move model parameters to the GPU
            model.cuda()
            
        else:
            model.cpu()
            
        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs), Variable(labels)
            steps+=1
            if cuda == 'gpu':
                # Move input and label tensors to the GPU
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                validation_loss = 0
                for ii, (inputs, labels) in enumerate(validationloader):

                    #images = images.resize_(images.size()[0], 784)
                    # Set volatile to True so we don't save the history
                    inputs = Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)
                    if cuda == 'gpu':
                        # Move input and label tensors to the GPU
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = model.forward(inputs)
                    validation_loss += criterion(output, labels).data[0]

                    ## Calculating the accuracy 
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()
               
                if ii==3:
                    break

    return (model, epochs, optimizer) 

def save_checkpoint(image_datasets_training, epochs, model, optimizer, filepath_checkpoint):
    #saving the trained model_state_dict
    torch.save({
          'epochs': epochs,
          'state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'model_class_to_idx': model.class_to_idx,
          },filepath_checkpoint)
  
    return(filepath_checkpoint)

def load_checkpoint(filepath_checkpoint, model):
    #loading the saved model
    print("=> loading checkpoint")
    checkpoint = torch.load(filepath_checkpoint, map_location=lambda storage, loc: storage)
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_class_to_idx']
    print("=> loaded checkpoint")
  
    return(epochs,model.load_state_dict, model.class_to_idx)