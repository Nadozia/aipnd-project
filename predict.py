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
    parser = argparse.ArgumentParser(description ='Lets make some predictions')
    parser.add_argument('input_img', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
    parser.add_argument('load_model_checkpoint', default = './checkpoint.pth', nargs='*', action = "store", type = str)
    parser.add_argument('--jsonpath', action="store", default= "./cat_to_name.json", type = str, help = 'json for changing class index to name')
    parser.add_argument('--gpu', action="store_true", default = False) 
    parser.add_argument('--topk', action="store", default= 5, type = int)

    pa = parser.parse_args()
    return pa

def main():
    pa = arg_parser()
    #load model
    model, criterion, optimizer, scheduler, args = fu.load_checkpoint(pa.load_model_checkpoint)
    
    #use gpu/cpu
    device = fu.use_cuda(pa.gpu)
    print(f'Device :{device}')
    
    #make prediction
    probs, topk = fu.predict(pa.input_img, model, device, pa.topk)
    probs, topk = probs[0].tolist(), topk[0].tolist()
    
    #map index to name
    cat_to_name = fu.json_load(pa.jsonpath)
    
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
        
    cls = []
    cls_name =[]
    for c in topk:
        cls.append(idx_to_class[c])
    for c in cls:
        cls_name.append(cat_to_name[c])
    #print model information
    print(args)
    #print result
    for i in range(len(probs)):
        print(f'The image has a {probs[i]*100:.2f} percentage of being {cls_name[i]}.')
                       
if __name__ == '__main__':
    main()