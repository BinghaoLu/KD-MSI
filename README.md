

## Prerequisite
```
imageio==2.9.0
numpy==1.21.2
opencv_python_headless==4.6.0.66
pandas==1.3.4
Pillow==10.1.0
pydensecrf==1.0rc2
torch==1.12.1+cu116
torchvision==0.11.2
```
## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt, test.txt`, each file records the image names (XXX.png) in the change detection dataset. It also contains `train_label.txt`, which is the image level label of training data, please run generate_cls_label.py to generate this list.

## Usage
### 1. Train siamese classification network with knowledge distillation
#### For WHU-CD teacher and student are set as minus and cat, respectively. For LEVIR-CD, abs, cat respectively. For DSIFN-CD, minus, cat respectively.
```bash
CUDA_VISIBLE_DEVICES=0 python train_classification_with_KD.py --data_dir $your_dir --tag WHU_KD_T_minus_S_cat --teacher minus --student abs
```

### 2. Apply Multi-scale Sigmoid Inference to refine Change Probability Map
```bash
CUDA_VISIBLE_DEVICES=0 python multi_scale_sigmoid_inference.py --data_dir $your_dir --tag WHU_KD_T_minus_S_cat --student_combination minus --scales 0.5,1.0,1.25,2.0
```

### 3. Evaluate Change Probability Map with different background threshold
```bash
python3 evaluate.py --list_file train.txt --predict_folder $predict_folder --mode npy --data_dir $your_dir
```

### 4. Generate pseudo masks with the optimal background threshold
```bash
python3 make_pseudo_labels.py --data_dir $your_dir --experiment_name WHU_KD_T_minus_S_cat@train@scale=0.5,1.0,1.25,2.0 --domain train --threshold 0.3
```
### 5. Train change detection model with pseudo labels
```bash
python3 train_change_detection.py --data_dir $your_dir --tag WHU_weakly_change_detection --label_name WHU_KD_T_minus_S_cat@train@scale=0.5,1.0,1.25,2.0@crf=0@255@threshold0.3
```

### 6. Inference change detection
```bash
python3 inference_change_detection.py --data_dir $your_dir --tag WHU_weakly_change_detection --scales 0.5,1.0,1.5,2.0
```



