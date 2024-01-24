import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from core.data_utils_for_CAM_generation import CDDataAugmentation
class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data

class WSCDDataSet(Dataset):
    
    def __init__(self, pre_img_folder=None, post_img_folder=None, list_file=None, 
                 img_size=256,to_tensor=True):
        
        self.pre_img_folder = pre_img_folder
        self.post_img_folder = post_img_folder
        self.list_file = list_file
        self.list_data = []

        with open(self.list_file, 'r') as file:

            for line in file:

                parts = line.strip().split(',')

                filename = parts[0]
                class_label = int(parts[-1])  # Assuming the class label is always the last element
                self.list_data.append((filename, class_label))
        self.length = len(self.list_data)

        
        self.img_size = img_size
        self.to_tensor = to_tensor
        
        self.augm = CDDataAugmentation(
            img_size=self.img_size
        )
        
    
    def __getitem__(self,idx):
        # print(self.pre_img_folder)
        # print(self.list_data)
        pre_img_path = os.path.join(self.pre_img_folder, self.list_data[idx][0])
        post_img_path = os.path.join(self.post_img_folder, self.list_data[idx][0])
        
        pre_img = np.array(Image.open(pre_img_path).convert('RGB'))
        post_img = np.array(Image.open(post_img_path).convert('RGB'))
        
        
        [pre_img, post_img] = self.augm.transform(imgs=[pre_img, post_img], labels=None,to_tensor=self.to_tensor)
        
        label = torch.tensor(self.list_data[idx][1]).unsqueeze(0).float()
        # print(label.size())
        
        return pre_img, post_img, label
    
    def __len__(self):
        return self.length
    

class WSCDDataSet_iou_evaluate(Dataset):
    
    def __init__(self, pre_img_folder=None, post_img_folder=None, mask_folder=None, list_file=None, 
                 img_size=256,to_tensor=True):
        
        self.pre_img_folder = pre_img_folder
        self.post_img_folder = post_img_folder
        self.list_file = list_file
        self.list_data = []

        with open(self.list_file, 'r') as file:

            for line in file:

                parts = line.strip().split(',')

                filename = parts[0]
                class_label = int(parts[-1])  # Assuming the class label is always the last element
                self.list_data.append((filename, class_label))
        self.length = len(self.list_data)


        self.mask_folder = mask_folder
        
        self.img_size = img_size
        self.to_tensor = to_tensor

        self.augm = CDDataAugmentation(img_size=self.img_size)
       
        
    
    def __getitem__(self,idx):
        
        pre_img_path = os.path.join(self.pre_img_folder, self.list_data[idx][0])
        post_img_path = os.path.join(self.post_img_folder, self.list_data[idx][0])
        mask_path = os.path.join(self.mask_folder, self.list_data[idx][0])
        base_name, ext = os.path.splitext(mask_path)
        mask_path = base_name + '.png'

        
        pre_img = np.array(Image.open(pre_img_path).convert('RGB'))
        post_img = np.array(Image.open(post_img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'),dtype=np.uint8)
        
        
        [pre_img, post_img] = self.augm.transform([pre_img, post_img], to_tensor=self.to_tensor)
        # pre_img = ToTensor()(pre_img)
        # post_img = ToTensor()(post_img)
        mask = ToTensor()(mask).long().squeeze(0)
        # print(mask.size()).squeeze(0)
        
        label = torch.tensor(self.list_data[idx][1]).unsqueeze(0).float()
        # print(mask)

        
        return pre_img, post_img, label, mask
    
    def __len__(self):
        return self.length
    
class WSCDDataSet_with_ID(Dataset):
    
    def __init__(self, pre_img_folder=None, post_img_folder=None, list_file=None, 
                 img_size=256,change_only=False):
        
        self.pre_img_folder = pre_img_folder
        self.post_img_folder = post_img_folder
        self.list_file = list_file
        self.change_only=change_only
        self.img_size = img_size
        self.list_data = []
        with open(self.list_file, 'r') as file:

            for line in file:

                parts = line.strip().split(',')

                filename = parts[0]
                class_label = int(parts[-1])  # Assuming the class label is always the last element
                if not change_only:
                    self.list_data.append((filename, class_label))
                else:
                    if class_label==1:
                        self.list_data.append((filename, class_label))

        self.length = len(self.list_data)
        print(self.length)
    
    def __getitem__(self,idx):


        pre_img_path = os.path.join(self.pre_img_folder, self.list_data[idx][0])
        post_img_path = os.path.join(self.post_img_folder, self.list_data[idx][0])

        pre_img = Image.open(pre_img_path).convert('RGB')
        post_img = Image.open(post_img_path).convert('RGB')
        id = self.list_data[idx][0][:-4]
        
        #[pre_img, post_img] = self.augm.transform([pre_img, post_img], to_tensor=self.to_tensor)
        
        label = torch.tensor(self.list_data[idx][1]).unsqueeze(0).float()
        
        return pre_img, post_img, label, id
    
    def __len__(self):

        return self.length

