

import os
import sys
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.WS_dataset import *


from tools.general.io_utils import *
from tools.general.time_utils import *

from tools.ai.log_utils import *

from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *




def accuracy(pred, y):
    
    correct = sum(row.all().int().item() for row in (pred.ge(0) == y))
    n = y.shape[0]
    return correct / n



parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix
parser.add_argument('--teacher', default='abs', type=str)
parser.add_argument('--student', default='minus', type=str)
###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--max_epoch', default=20, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=256, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='WHU_KD', type=str)



if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    
    log_path = log_dir + f'{args.tag}.txt'
    model_path = model_dir + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    train_dataset = WSCDDataSet(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                 list_file=args.data_dir+'/list/train_label.txt',
                                 img_size=args.image_size)

    valid_dataset = WSCDDataSet_iou_evaluate(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                             mask_folder=args.data_dir+'/label',
                                 list_file=args.data_dir+'/list/train_label.txt',
                                 img_size=args.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration



    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))


    
    ###################################################################################
    # Network
    ###################################################################################

    
    model = Classifier_Siamese(args.architecture, 1, args.mode, args.teacher)
    model2 = Classifier_Siamese(args.architecture, 1, args.mode, args.student)

    
    param_groups = model.get_parameter_groups(print_fn=None)
    param_groups2 = model2.get_parameter_groups(print_fn=None)
    
    model = model.cuda()
    model.train()

    model2 = model2.cuda()
    model2.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)

    load_model_fn = lambda: load_model(model2, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model2, model_path, parallel=the_number_of_gpu > 1)
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.BCEWithLogitsLoss().cuda()

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},

        {'params': param_groups2[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups2[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups2[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups2[3], 'lr': 20*args.lr, 'weight_decay': 0},

    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'class_loss','accuracy'])

    best_acc1 = -1
    best_train_mIoU = -1
    thresholds = list(np.arange(0.10, 1, 0.05))
    
    def evaluate(loader):
        model2.eval()
        eval_timer.tik()

        valid_meter = Average_Meter(['val_loss','val_acc'])
        meter_dic = {th : Calculator_For_mIoU() for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (imageA, imageB, labels,gt_masks) in enumerate(loader):


                imageA = imageA.cuda()
                imageB = imageB.cuda()
                labels =  labels.cuda()

                logits, features = model2(imageA,imageB)
                
                loss = class_loss_fn(logits, labels).mean()
                acc = accuracy(logits, labels)

                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = torch.sigmoid(features)*mask

                

                valid_meter.add({'val_loss': loss.item(),'val_acc':acc})

                for batch_index in range(imageA.size()[0]):
                        # c, h, w -> h, w, c
                        cam = get_numpy_from_tensor(cams[batch_index]).transpose((1, 2, 0))


                        cam = cv2.resize(cam,(256,256),interpolation=cv2.INTER_NEAREST)
                        cam = cam.reshape(cam.shape[0],cam.shape[1],1)

                        gt_mask = get_numpy_from_tensor(gt_masks[batch_index])
                        
                        

                        h, w,c = cam.shape
                        gt_mask = cv2.resize(gt_mask, (h,w), interpolation=cv2.INTER_NEAREST)

                        

                        for th in thresholds:
                            bg = np.ones_like(cam[:, :, 0]) * th
                            pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)

                            meter_dic[th].add(pred_mask, gt_mask)

                    # break

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        model2.train()

        best_th = 0.0
        best_mIoU = 0.0

        for th in thresholds:
            mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
            if best_mIoU < mIoU_foreground:
                best_th = th
                best_mIoU = mIoU_foreground

        return valid_meter.get(clear=True), best_th, best_mIoU

    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)                                                          

    for iteration in range(max_iteration):
        imageA, imageB, labels = train_iterator.get()
        imageA, imageB, labels = imageA.cuda(), imageB.cuda(), labels.cuda()

        logits , features1= model(imageA,imageB)
        logits2 ,features2 = model2(imageA,imageB)

        cam = make_cam(features1)*labels.unsqueeze(2).unsqueeze(3)
        cam1 = cam.clone().detach()


        cam2 = F.sigmoid(features2)

        loss_kd = nn.MSELoss()(cam2,cam1)
        class_loss = class_loss_fn(logits, labels).mean()
        acc1= accuracy(logits, labels)
        loss = class_loss + 10*loss_kd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'class_loss' : class_loss.item(),
            'accuracy':acc1
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, class_loss,acc1 = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'class_loss' : class_loss,
                'acc1':acc1,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                acc1={acc1:.4f},\
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/class_loss', class_loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
            writer.add_scalar('Train/acc1', acc1, iteration)
        
        ################################################################################################
        # Evaluation
        ################################################################################################
        # if True:
        if (iteration + 1) % val_iteration == 0:
            (valid_loss,val_acc1),threshold, mIoU = evaluate(valid_loader)
            
            if best_acc1 == -1 or best_acc1 < val_acc1:
                best_acc1 = val_acc1

            if best_train_mIoU == -1 or best_train_mIoU < mIoU:
                best_train_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'valid_loss' : valid_loss,
                'best_acc1' : best_acc1,
                'val_acc1' : val_acc1,
                'train_mIoU':mIoU,
                'threshold':threshold,
                'best_train_mIoU':best_train_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)

            
            log_func('[i] \
                iteration={iteration:,}, \
                valid_loss={valid_loss:.4f}, \
                val_acc1={val_acc1:.4f}%,\
                best_acc1={best_acc1:.4f}%, \
                threshold={threshold:.2f}, \
                train_mIoU={train_mIoU:.2f}%, \
                best_train_mIoU={best_train_mIoU:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            
            writer.add_scalar('Evaluation/valid_loss', valid_loss, iteration)
            writer.add_scalar('Evaluation/valid_acc1', val_acc1, iteration)
            writer.add_scalar('Evaluation/best_acc1', best_acc1, iteration)
    

    writer.close()

    print(args.tag)