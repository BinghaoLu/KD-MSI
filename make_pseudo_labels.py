

import os
import sys

import argparse
import numpy as np

import imageio

import torch



from core.networks import *
from core.WS_dataset import *

from tools.general.io_utils import *
from tools.general.time_utils import *


# from tools.ai.log_utils import *
from tools.ai.demo_utils import *

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
parser.add_argument('--image_size', default=256, type=int)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='WHU_KD@train@scale=1.0,1.25,1.5', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--threshold', default=0.65, type=float)
parser.add_argument('--crf_iteration', default=0, type=int)


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    cam_dir = f'./experiments/predictions/{args.experiment_name}/'
    pred_dir = create_directory(f'./experiments/predictions/{args.experiment_name}@crf={args.crf_iteration}@255@threshold{args.threshold}/')

    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    dataset = WSCDDataSet_with_ID(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                 list_file=args.data_dir+'/list/train_label.txt',
                                 img_size=args.image_size,change_only= False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_imageA, ori_imageB, label, image_id) in enumerate(dataset):
            png_path = pred_dir + image_id + '.png'
            if os.path.isfile(png_path):
                continue
            
            ori_w, ori_h = ori_imageB.size
            predict_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()

            keys = predict_dict['keys']
            
            cams = predict_dict['hr_cam']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

            cams = np.argmax(cams, axis=0)
            if keys.shape[0]>1:

                if args.crf_iteration > 0:
                    cams = crf_inference_label(np.asarray(ori_imageB), cams, n_labels=keys.shape[0], t=args.crf_iteration)
            else:
                pass
            
            conf = keys[cams]*255
            imageio.imwrite(png_path, conf.astype(np.uint8))
            


            sys.stdout.write('\r# Make Pseudo Labels [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
            sys.stdout.flush()
        print()
    
