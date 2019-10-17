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

import argparse
import flower_utils as fu

def arg_parser():
    parser = argparse.ArgumentParser(description ='Lets train a Neural Network')
    parser.add_argument('data_dir', action="store", default="./flowers", nargs='*')
    parser.add_argument('--savepath', action="store", default="./checkpoint.pth", help = 'directory to save checkpoint')
    parser.add_argument('--gpu', action="store_true", default = False)
    parser.add_argument('--jsonpath', action="store", default= "./cat_to_name.json", type = str, help = 'json for changing class index to name')
    parser.add_argument('--arch', action="store", default="densenet161", help = 'Choose a pretrain model(default = densenet161)\
                                                                                :vgg16, densenet161, resnet18', type = str)
    parser.add_argument('--hidden_unit', action="store", default= 4096, type = int, help = 'default = 4096')
    parser.add_argument('--dropout', action="store", default= 0.1, type = int, help = 'default = 0.1')
    parser.add_argument('--learning_rate', action="store", default= 0.001, type = int, help = 'default = 0.001')
    parser.add_argument('--step', action="store", default = 2, type = int, help = 'learning rate adjusts by every step epoch, default = 2')
    parser.add_argument('--gamma', action="store", default = 0.1, type = int, help = 'learning rate step down by a ratio, default =0.1')
    parser.add_argument('--epochs', action="store", default = 6, type = int, help = 'Train epochs, default = 6')

    pa = parser.parse_args()
    return pa

def main():
    Train_mode = 'train'
    Valid_mode = 'valid'
    Test_mode = 'test'
    
    pa = arg_parser()
    dataset, dataloaders = fu.data_load(data_dir = pa.data_dir)
    cat_to_name = fu.json_load(pa.jsonpath)
    model, criterion, optimizer, scheduler, args = fu.build_model(structure = pa.arch,
                                                                  hidden_unit = pa.hidden_unit,
                                                                  dropout = pa.dropout,
                                                                  learning_rate = pa.learning_rate,
                                                                  step = pa.step,
                                                                  gamma = pa.gamma,
                                                                  json = cat_to_name)
    
    device = fu.use_cuda(pa.gpu)
    print(f'Device :{device}')
    
    with active_session():
        model = fu.TrainValid(model, criterion, optimizer, scheduler, epochs = pa.epochs, device = device, dataloaders = dataloaders)
    
    fu.save_checkpoint(pa.savepath, model, optimizer, args, dataset= dataset, json = cat_to_name)

    print('Training and Saving finished!')
    
if __name__ == '__main__':
    main()
