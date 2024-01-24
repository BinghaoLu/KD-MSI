
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet
from .abc_modules import ABC_Model

from .deeplab_utils import ASPP, Decoder


from tools.ai.torch_utils import resize_for_tensors

#######################################################################
# Normalization
#######################################################################


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)
def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)




class Classifier_Siamese(Backbone):
    def __init__(self, model_name, num_classes=1, mode='fix',combination='cat'):
        super().__init__(model_name, num_classes, mode)
        self.combination = combination
        if self.combination =='cat':
            self.classifier = nn.Conv2d(4096, num_classes, 1, bias=False)
        else:
             self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)              
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward1(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        return x

    def forward(self,x,y, with_cam=False):
        if self.combination == 'cat':
            z= torch.cat((self.forward1(y), self.forward1(x)),dim=1)
        elif self.combination == 'minus':
            z= self.forward1(y)-self.forward1(x)
        elif self.combination == 'abs':
            z= torch.abs(self.forward1(y)-self.forward1(x))
        

        features = self.classifier(z)
        logits = self.global_average_pooling_2d(features)
        return logits, features






    
class DeepLabv3_Plus_siamese(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)
        
        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn
        
        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)
        
    def forward1(self, x, with_cam=False):


        x = self.stage1(x)
        x = self.stage2(x)
        x_low_level = x
        
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        return x,x_low_level
    
    def forward(self,x,y):
        inputs = x
        x,x_low_level = self.forward1(x)
        y, y_low_level = self.forward1(y)

        x = self.aspp(x-y)
        x = self.decoder(x, x_low_level-y_low_level)
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

        return x
