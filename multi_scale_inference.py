# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy

import argparse
import numpy as np


import PIL

import torch.nn.functional as F




from core.networks import *

from core.WS_dataset import *

from tools.general.io_utils import *
from tools.general.time_utils import *


from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.augment_utils import *


def get_cam(ori_imageA,ori_imageB, scale):
        # preprocessing
        image = copy.deepcopy(ori_imageA)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).cuda()
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        imagesA = images

        image = copy.deepcopy(ori_imageB)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).cuda()
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        imagesB = images
        
        # inferenece
        _, features = model(imagesA,imagesB, with_cam=True)

        # postprocessing
        cams = F.relu(features)
        cams = cams[0] + cams[1].flip(-1)

        return cams

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--image_size', default=256, type=int)
###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)
parser.add_argument('--combination', default='cat', type=str)
###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)





if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s'%args.scales
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.5,0.5,0.5]
    imagenet_std = [0.5,0.5,0.5]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    dataset = WSCDDataSet_with_ID(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                 list_file=args.data_dir+'/list/train_label.txt',
                                 img_size=args.image_size,change_only= False)

    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier_Siamese(args.architecture, 1, args.mode, args.combination)

    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    


    with torch.no_grad():
        length = len(dataset)
        for step, (ori_imageA,ori_imageB, label, image_id) in enumerate(dataset):
            ori_w, ori_h = ori_imageA.size

            npy_path = pred_dir + str(image_id) + '.npy'
            if os.path.isfile(npy_path):
                continue
            
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

            cams_list = [get_cam(ori_imageA,ori_imageB, scale) for scale in scales]

            
            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]
            
            keys = torch.nonzero(label)[:, 0]
            
            
            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

            # save cams
            keys = np.pad(keys + 1, (1, 0), mode='constant')
            np.save(npy_path, {"keys": keys, "hr_cam": hr_cams.cpu().numpy()})
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()))
            sys.stdout.flush()
        print()
    
