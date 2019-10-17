import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict

import time
import copy
from workspace_utils import active_session

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

arch_output = {'vgg16':25088,
              'resnet18':512,
              'densenet161' : 2208}

Train_mode = 'train'
Valid_mode = 'valid'
Test_mode = 'test'
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def use_cuda(gpu):
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('CUDA not availiable, will use CPU')
    else:
        device = torch.device('cpu')
        
    return device

def json_load(json_path):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def data_load(data_dir= 'flowers'):
    #directory path
    dirs = {'train': data_dir + '/train',
            'valid': data_dir + '/valid',
            'test': data_dir + '/test'}
    #transformation
    transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
              'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
              'test': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])}
    
    dataset = {x: datasets.ImageFolder(dirs[x], transform = transform[x]) for x in ['train','valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size = 64, shuffle = True) for x in ['train', 'valid', 'test']}
    
    return dataset, dataloaders

def build_model(structure = 'densenet161', hidden_unit = 4096, dropout = 0.1, learning_rate = 0.001, step = 0, gamma = 0, json = json):
    
    #model argument
    args = {'structure' : structure,
            'hidden_unit' : hidden_unit,
            'dropout' : dropout,
            'learning_rate' : learning_rate,
            'step' : step,
            'gamma' : gamma}
    
    #download pretrained model
    if structure == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif structure == 'resnet18':
        model = models.resnet18(pretrained = True)
    elif structure == 'densenet161':
        model = models.densenet161(pretrained = True)
        
    #freeze feature extract layer        
    for param in model.parameters():
        param.requires_grad = False     
        
    #build classifier
    classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(arch_output[structure], hidden_unit)),
                                    ('relu1', nn.ReLU()),
                                    ('drop1', nn.Dropout(p = dropout)),
                                    ('fc2', nn.Linear(hidden_unit, len(json))),
                                    ('output', nn.LogSoftmax(dim=1))]))
    
    if structure == 'resnet18':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)
    else:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
            
    #criterion, scheduler
    criterion = nn.NLLLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step, gamma = gamma)
    
    return model, criterion, optimizer, scheduler, args

def TrainValid(model, criterion, optimizer, scheduler, epochs, device, dataloaders):
    #training and validation
    print('Start to train......')
    model.to(device)
    
    #start time
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for e in range(epochs):
        for mode in [Train_mode, Valid_mode]:
            #set model mode
            if mode == Train_mode:
                print('\nmode: ', mode)
                scheduler.step()
                model.train()
                torch.set_grad_enabled(True)
                
            elif mode == Valid_mode:
                print('\nmode:', mode)
                model.eval()
                torch.set_grad_enabled(False)
                
            accuracy = 0                
            pass_count = 0
            running_loss = 0 
            
            for inputs, labels in dataloaders[mode]:
                pass_count += 1
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                #forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                #backward 
                if mode == Train_mode:
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if mode == Train_mode:
                print(f'Epoch: {e+1}/{epochs}...'
                      f'\nTraining loss: {running_loss/pass_count:.4f}'
                      f'\nTraining Accuracy: {accuracy/pass_count:.4f}')
            elif mode == Valid_mode:
                print(f'Validation loss: {running_loss/pass_count:.4f}'
                      f'\nValidation Accuracy: {accuracy/pass_count:.4f}'
                      '\n---------------------------------------------------')
                if accuracy > best_acc:
                    best_acc = accuracy/pass_count
                    best_model = copy.deepcopy(model.state_dict())
        
            
    print(f'\nTraining completed. Time used: {time.time()-start:.0f}sec')
    print(f'Best valid accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model)
    model.eval()
    return model

def model_testing(model, criterion, dataloader):
    model.eval()
    test_accuracy = 0
    test_loss = 0
    torch.no_grad()
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, labels)
        
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim =1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')  
    
def save_checkpoint(path, model, optimizer, args, dataset, json):
    checkpoint = {  'model_state_dict': model.state_dict(),
                    'class_to_idx' : dataset['train'].class_to_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'json': json}
    torch.save(checkpoint, path)
    
def load_checkpoint(checkpointpath):
    checkpoint = torch.load(checkpointpath, map_location=lambda storage, loc: storage)
        
    #build model
    model, criterion, optimizer, scheduler, args = build_model(structure = checkpoint['args']['structure'], 
                                                               hidden_unit = checkpoint['args']['hidden_unit'], 
                                                               dropout = checkpoint['args']['dropout'], 
                                                               learning_rate = checkpoint['args']['learning_rate'], 
                                                               step = checkpoint['args']['step'], 
                                                               gamma = checkpoint['args']['gamma'],
                                                               json = checkpoint['json'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, criterion, optimizer, scheduler, args

def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(imagepath)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    img = transform(img)
    return img

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(img.to(device))
        probs = torch.exp(output)
    return probs.topk(topk)
