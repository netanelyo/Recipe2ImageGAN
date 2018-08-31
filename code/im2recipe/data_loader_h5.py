from __future__ import print_function
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import h5py
 

# https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3
# https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/6
class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file, transform=None):
        super(dataset_h5, self).__init__()
 
        self.file = h5py.File(in_file, 'r')
        self.transform = transform
 
    def __getitem__(self, index):
        img = self.file['images'][index, ...]
        instructions = self.file['inst_emb'][index, ...]
        instLength = self.file['inst_length'][index, ...]
        ingredients = self.file['ingr_emb'][index,...]
        ingLength = self.file['ingr_lenght'][index,...]

        # converting to torch.Tensor
        instructionsTorch = torch.FloatTensor(instructions)
        ingredientsTorch = torch.LongTensor(ingredients)

        # rotate axis to (256,256,3)
        img = np.moveaxis(img,0,-1)
        # convert to PIL Image
        img = Image.fromarray(img, 'RGB')
        
        if self.transform is not None:
            img = self.transform(img)        
        
        #print(input.shape)
        return [img,instructionsTorch,instLength,ingredientsTorch,ingLength], [torch.ones(1),index,index]
 
    def __len__(self):
        return self.file['images'].shape[0]


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ])



'''
dataset = dataset_h5(opts.data_path+'/newData.h5',transform=transform)
assert dataset

print("starting")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                         shuffle=True, num_workers=opts.workers)
 
'''
 