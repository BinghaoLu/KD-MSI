import os

import numpy as np
from PIL import Image
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--list_file", default='train.txt', type=str)
parser.add_argument("--threshold", default=None, type=float)
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--predict_folder', default='', type=str)

parser.add_argument('--mode', default='npy', type=str) # png
parser.add_argument('--max_th', default=0.90, type=float)

args = parser.parse_args()


if args.experiment_name =='':
    predict_folder = args.predict_folder
else:
    predict_folder = './experiments/predictions/{}/'.format(args.experiment_name)


gt_folder = args.data_dir + '/label'

list_file = args.data_dir + f'/list/{args.list_file}'


categories = ['background', 
    'change']
num_cls = len(categories)

def compare(start,step,TP,P,T, name_list):
    for idx in range(start,len(name_list),step):
        name = name_list[idx]
        # print(predict_folder + name + '.npy')

        if os.path.isfile(predict_folder + name + '.npy'):
            predict_dict = np.load(os.path.join(predict_folder, name + '.npy'), allow_pickle=True).item()
            
            if 'hr_cam' in predict_dict.keys():
                cams = predict_dict['hr_cam']

                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

            elif 'rw' in predict_dict.keys():
                cams = predict_dict['rw']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
            
            keys = predict_dict['keys']
            predict = keys[np.argmax(cams, axis=0)]
        else:
            predict = np.array(Image.open(predict_folder + name + '.png'))/255
        
        gt_file = os.path.join(gt_folder,'%s.png'%name)
        gt = np.array(Image.open(gt_file).convert('L'))
        ######
        # h,w = predict.shape
        # gt = gt.resize((h,w))
        #####
        gt = gt/255
        ######
        # h,w = predict.shape
        # gt = gt.resize((h,w))
        #####
        
        cal = gt<255

        mask = (predict==gt)# * cal

        for i in range(num_cls):
            P[i].acquire()
            P[i].value += np.sum((predict==i)*cal)
            P[i].release()
            T[i].acquire()
            T[i].value += np.sum((gt==i)*cal)
            T[i].release()
            TP[i].acquire()
            TP[i].value += np.sum((gt==i)*mask)
            TP[i].release()

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, num_cores=8):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    p_list = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=compare, args=(i, num_cores, TP, P, T, name_list))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    F1 = []
    for i in range(num_cls):
        iou = TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10)
        t_tp = T[i].value / (TP[i].value + 1e-10)
        p_tp = P[i].value / (TP[i].value + 1e-10)
        fp_all = (P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10)
        fn_all = (T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10)
        
        precision = TP[i].value / (TP[i].value + (P[i].value - TP[i].value) + 1e-10)
        recall = TP[i].value / (TP[i].value + (T[i].value - TP[i].value) + 1e-10)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        IoU.append(iou)
        T_TP.append(t_tp)
        P_TP.append(p_tp)
        FP_ALL.append(fp_all)
        FN_ALL.append(fn_all)
        F1.append(f1)



    overall_accuracy = sum([tp.value for tp in TP]) / sum([t.value for t in T ]) 

    mean_f1 = np.mean(F1)
    
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
    
    miou = np.mean(np.array(IoU))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    miou_foreground = np.mean(np.array(IoU)[1:])
    
    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    loglist['overall_accuracy'] = overall_accuracy * 100
    loglist['mean_f1'] = mean_f1 * 100
    
    return loglist


if __name__ == '__main__':

    # name_list = os.listdir(gt_folder)
    # name_list = [os.path.splitext(filename)[0] for filename in name_list]
    name_list=[]
    with open(list_file, 'r') as file:

        for line in file:

            filename = line.strip().replace('.png','')
            name_list.append(filename)


    if args.mode == 'png':
        loglist = do_python_eval(predict_folder, gt_folder, name_list, 2)
        print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f},IoU_foreground={:.3f}%, OA={:.3f}, f1={:.3f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all'],loglist['miou_foreground']*100,loglist['overall_accuracy'],loglist['mean_f1']))
    elif args.mode == 'rw':
        th_list = np.arange(0.05, args.max_th, 0.05).tolist()

        over_activation = 1.60
        under_activation = 0.60
        
        mIoU_list = []
        FP_list = []

        for th in th_list:
            args.threshold = th
            loglist = do_python_eval(predict_folder, gt_folder, name_list, 2)

            mIoU, FP = loglist['mIoU'], loglist['fp_all']

            print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(th, mIoU, FP))

            FP_list.append(FP)
            mIoU_list.append(mIoU)
        
        best_index = np.argmax(mIoU_list)
        best_th = th_list[best_index]
        best_mIoU = mIoU_list[best_index]
        best_FP = FP_list[best_index]

        over_FP = best_FP * over_activation
        under_FP = best_FP * under_activation

        print('Over FP : {:.4f}, Under FP : {:.4f}'.format(over_FP, under_FP))

        over_loss_list = [np.abs(FP - over_FP) for FP in FP_list]
        under_loss_list = [np.abs(FP - under_FP) for FP in FP_list]

        over_index = np.argmin(over_loss_list)
        over_th = th_list[over_index]
        over_mIoU = mIoU_list[over_index]
        over_FP = FP_list[over_index]

        under_index = np.argmin(under_loss_list)
        under_th = th_list[under_index]
        under_mIoU = mIoU_list[under_index]
        under_FP = FP_list[under_index]
        
        print('Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(best_th, best_mIoU, best_FP))
        print('Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th, over_mIoU, over_FP))
        print('Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th, under_mIoU, under_FP))
    else:
        if args.threshold is None:
            th_list = np.arange(0, args.max_th, 0.05).tolist()
            
            best_th = 0
            best_mIoU = 0

            for th in th_list:
                args.threshold = th
                loglist = do_python_eval(predict_folder, gt_folder, name_list, 2)
                # print('Th={:.2f}, mIoU={:.3f}%,IoU_foreground={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'],loglist['miou_foreground']*100, loglist['fp_all'], loglist['fn_all']))
                print('th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f},IoU_foreground={:.3f}%, OA={:.3f}, f1={:.3f}'.format(args.threshold,loglist['mIoU'], loglist['fp_all'], loglist['fn_all'],loglist['miou_foreground']*100,loglist['overall_accuracy'],loglist['mean_f1']))


                if loglist['mIoU'] > best_mIoU:
                    best_th = th
                    best_mIoU = loglist['mIoU']
            
            print('Best Th={:.2f}, mIoU={:.3f}%'.format(best_th, best_mIoU))
        else:
            loglist = do_python_eval(predict_folder, gt_folder, name_list, 21)
            # print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
            # print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f},IoU_foreground={:.3f}%, OA={:.3f}, f1={:.3f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all'],loglist['miou_foreground']*100,loglist['overall_accuracy'],loglist['mean_f1']))


