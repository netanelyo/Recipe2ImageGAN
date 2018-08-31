'''MIT License

Copyright (c) 2016 hanzhanggit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg
import h5py
import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
from main import parse_args
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(imageIndex, imsize, file_name,transform=None, normalize=None):

    f = h5py.File(file_name,'r')
    images = f['images']
    img = images[imageIndex]
    # rotate axis to (256,256,3)
    img = np.moveaxis(img, 0, -1)
    # convert to PIL Image
    img = Image.fromarray(img, 'RGB')

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
        
    rec_id = f['recIDs'][imageIndex]
    img_id = f['imagesIDs'][imageIndex]

    return ret, rec_id, img_id


class TextDataset(data.Dataset):
    def __init__(self, data_dir, args, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            # we change to our normalization
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.target_transform = target_transform
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir

        # we define split_dir to be data/train
        split_dir = data_dir + '/' + split

        # self.filenames = data/train/data3162.h5
        self.filenames = split_dir + '/' + args.imagesFile
        self.embeddings = self.load_embedding(split_dir, embedding_type, args)
        # self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()


        self.iterator = self.prepair_training_pairs
        '''if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs'''

    '''def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox'''

    '''def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict'''

    def load_embedding(self, data_dir, embedding_type, args):

        embedding_filename = '/' + args.rec_embs_file
        with h5py.File(data_dir + embedding_filename, 'r') as f:
	    if cfg.TEXT.EMBEDDING_TYPE == "sem":
	        embeddings = f.get('rec_sem').value
            else:
 	        embeddings = f.get('rec_embs').value               
            print('embeddings: ', embeddings.shape)
        self.num_of_samples = embeddings.shape[0]
        return embeddings

    '''def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id'''

    '''def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames'''

    def prepair_training_pairs(self, index):

        embedding = self.embeddings[index]
        imgs, rec_id, im_id = get_imgs(index, self.imsize,
                        self.filenames, self.transform, normalize=self.norm)

        wrong_ix = random.choice(range(0, index) + range(index+1, self.num_of_samples))
        
        wrong_imgs, _, _ = get_imgs(wrong_ix, self.imsize,
                              self.filenames, self.transform, normalize=self.norm)
                              
         

        return imgs, wrong_imgs, embedding, rec_id, im_id, index  # captions
        

    def prepair_test_pairs(self, index):
        # captions = self.captions[key]
        embeddings = self.embeddings[index]
        imgs, rec_id, im_id = get_imgs(index, self.imsize,
                        self.filenames, self.transform, normalize=self.norm)

        #if self.target_transform is not None:
        #    embeddings = self.target_transform(embeddings)

        return imgs, embeddings, rec_id, im_id, index  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return self.num_of_samples
