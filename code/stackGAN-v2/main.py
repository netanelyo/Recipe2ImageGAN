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

from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


from miscc.config import cfg, cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_proGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--imagesFile', default='data3162.h5')
    parser.add_argument('--rec_embs_file', default='rec_embs_data24031.h5')
    parser.add_argument('--emb_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--mode', default='train', help="if eval then eval mode else train")
    parser.add_argument('--emb_type', default='', help="sem for semantic regularization")


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None and len(args.cfg_file)>0:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
        
    if args.netG != '':
        cfg.TRAIN.NET_G = args.netG
    
    if args.netD != '':
        cfg.TRAIN.NET_D = args.netD

    if args.mode == 'eval':
        cfg.TRAIN.FLAG = False
        print('-------------------------')
        print('-------------------------')
        print("EVALUATION MODE")
        print('-------------------------')
        print('-------------------------')
    else:
        print('-------------------------')
        print('-------------------------')
        print("TRAIN MODE")

        print("press ctrl+C NOW if you meant Evaluation mode")
        print('-------------------------')
        print('-------------------------')
    
    time.sleep(8)
        
    cfg.GAN.EMBEDDING_DIM = args.emb_dim
    
    cfg.TRAIN.BATCH_SIZE = args.batch_size

    cfg.TEXT.EMBEDDING_TYPE = args.emb_type

    if args.emb_type == 'sem':
        cfg.TEXT.DIMENSION = 1048

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if cfg.TRAIN.FLAG:
        output_dir = cfg.DATA_DIR+'/output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    else:
        output_dir = cfg.DATA_DIR+'/output/EVAL_%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    '''if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'birds':
            bshuffle = False
            split_dir = 'test' '''

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    from datasets import TextDataset
    dataset = TextDataset(cfg.DATA_DIR, args, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    print('num_gpu: ', num_gpu)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    if cfg.TRAIN.FLAG:
        from trainer import condGANTrainer as trainer
    else:
        from eval_trainer import condGANTrainer as trainer
    # disable cudnn fixed the mem leak problem
    #torch.backends.cudnn.enabled = False
    algo = trainer(output_dir, dataloader, imsize)

    start_t = time.time()
    algo.train()
    ''''if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)'''
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
    ''' Running time comparison for 10epoch with batch_size 24 on birds dataset
        T(1gpu) = 1.383 T(2gpus)
            - gpu 2: 2426.228544 -> 4min/epoch
            - gpu 2 & 3: 1754.12295008 -> 2.9min/epoch
            - gpu 3: 2514.02744293
    '''
