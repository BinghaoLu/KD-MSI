# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>


import sys
import copy

import argparse
import numpy as np
import PIL
import imageio

import torch

import torch.nn.functional as F




from core.networks import *

from core.WS_dataset import *


from tools.general.io_utils import *
from tools.general.time_utils import *


from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnet50', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################


parser.add_argument('--tag', default='WHU_weakly_change_detection', type=str)

parser.add_argument('--domain', default='test', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)
parser.add_argument('--image_size',default=256,type=int)


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration

    pred_dir = create_directory('./experiments/predictions/{}/'.format(args.tag))
    
    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.5,0.5,0.5]#[0.485, 0.456, 0.406]
    imagenet_std = [0.5,0.5,0.5]#[0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    # meta_dic = read_json('./data/VOC_2012.json')

    dataset = WSCDDataSet_with_ID(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                 list_file=args.data_dir+f'/list/{args.domain}_label.txt',
                                 img_size=args.image_size,change_only= False)
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus_siamese(args.backbone, num_classes=1 + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=2)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=2)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(imagesA,imagesB, image_size):
        imagesA = imagesA
        imagesB = imagesB
        
        logits = model(imagesA,imagesB)
        logits = resize_for_tensors(logits, image_size)
        
        logits = logits[0] + logits[1].flip(-1)

        logits = logits.permute(1,2,0)
        return logits

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_imageA, ori_imageB, gt_mask, image_id) in enumerate(dataset):
            ori_w, ori_h = ori_imageA.size

            cams_list = []

            for scale in scales:
                imageA = copy.deepcopy(ori_imageA)
                imageB = copy.deepcopy(ori_imageB)

                imageA = imageA.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                imageB = imageB.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                imageA = normalize_fn(imageA)
                imageB = normalize_fn(imageB)

                imageA = imageA.transpose((2, 0, 1))
                imageB = imageB.transpose((2, 0, 1))


                imageA = torch.from_numpy(imageA).cuda()
                imageB = torch.from_numpy(imageB).cuda()

                flipped_imageA = imageA.flip(-1)
                flipped_imageB = imageB.flip(-1)
                
                imagesA = torch.stack([imageA, flipped_imageA])
                imagesB = torch.stack([imageB, flipped_imageB])

                cams = inference(imagesA,imagesB, (ori_h, ori_w))
                cams_list.append(cams)
            
            preds = torch.stack(cams_list, axis=0)
            preds = torch.sum(preds,dim=0)
            preds = F.softmax(preds, dim=-1).cpu().numpy()
            # print(preds.shape)
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                preds = crf_inference(np.asarray(ori_imageB), preds.transpose((2, 0, 1)), t=args.iteration)
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)

            pred_mask = pred_mask*255
            imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))