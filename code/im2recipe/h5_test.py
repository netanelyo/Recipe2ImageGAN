print("starting h5_test.py")
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader_h5 import dataset_h5 # our data_loader
import numpy as np
from trijoint import im2recipe
import pickle
from args import get_parser
import h5py

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)
if not opts.no_cuda:
        torch.cuda.manual_seed(opts.seed)

np.random.seed(opts.seed)

def main():
   
    model = im2recipe()
    #model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0,1,2,3])
    # barelo:
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0])
    # model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0,1])
    if not opts.no_cuda:
        model.cuda()

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    cosine_crit = nn.CosineEmbeddingLoss(0.1)
    if not opts.no_cuda:
        cosine_crit.cuda()
    # cosine_crit = nn.CosineEmbeddingLoss(0.1)
    if opts.semantic_reg:
        weights_class = torch.Tensor(opts.numClasses).fill_(1)
        weights_class[0] = 0 # the background class is set to 0, i.e. ignore
        # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
        class_crit = nn.CrossEntropyLoss(weight=weights_class)
        if not opts.no_cuda:
            class_crit.cuda()
        # class_crit = nn.CrossEntropyLoss(weight=weights_class)
        # we will use two different criteria
        criterion = [cosine_crit, class_crit]
    else:
        criterion = cosine_crit

    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    transform = transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ])
                                     
    dataset = dataset_h5(opts.data_path + '/' + opts.emb_h5_input_file + '.h5',transform=transform)
    assert dataset

    print("starting")


    # preparing test loader 
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=(not opts.no_cuda))
    print 'Test loader prepared.'

    # run test
    test(test_loader, model, criterion)

def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        print "batch_num=", i
        input_var = list() 
        for j in range(len(input)):
            v = torch.autograd.Variable(input[j], volatile=True)
            # barelo:
            input_var.append(v.cuda() if not opts.no_cuda else v)
        target_var = list()
        for j in range(len(target)-2): # we do not consider the last two objects of the list
            #barelo-delte:target[j] = target[j]
            v = torch.autograd.Variable(target[j], volatile=True)
            target_var.append(v.cuda() if not opts.no_cuda else v)

        # compute output
        '''print "in test.py:, len(input_var)=", len(input_var)
        print "input_var[0].shape=", input_var[0].shape
        import math'''

        #print "input_var[1:=", input_var[1:]
        output = model(input_var[0],input_var[1], torch.squeeze(input_var[2],1), input_var[3], torch.squeeze(input_var[4],1))
        #barelo:
        '''print "len(output), len(output[0])=", len(output), len(output[0])
        print "output[0][0].shape=", output[0][0].shape
        print "output[0][1].shape=", output[0][1].shape
        print "output[0][0]=", output[0][0]
        print "output[0][1]=", output[0][1]'''

   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i==0:
            #data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            data3 = output[3].data.cpu().numpy()
            #data2 = target[-2]
            #data3 = target[-1]
        else:
            #data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
            data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)
            #data2 = np.concatenate((data2,target[-2]),axis=0)
            data3 = np.concatenate((data3,output[3].data.cpu().numpy()),axis=0)


    with h5py.File(opts.data_path + '/recipe_embs/' + opts.rec_emb_h5 + '_' + opts.emb_h5_input_file + '.h5', 'w') as f:
        f.create_dataset("rec_embs",data=data1)
        f.create_dataset("rec_sem",data=data3)
    
    print "output file created"

    return 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()