
import numpy as np



import torchvision.transforms.functional as TF

import torch


# def to_tensor_and_norm(imgs, labels):
#     # to tensor
#     imgs = [TF.to_tensor(img) for img in imgs]
#     labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
#               for img in labels]

#     imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             for img in imgs]
#     return imgs, labels




class CDDataAugmentation:

    def __init__(
            self,
            img_size,

    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False

    def transform(self, imgs, labels=None, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]
        if labels !=None:
            labels = [TF.to_pil_image(img) for img in labels]
            if len(labels) != 0:
                if labels[0].size != (self.img_size, self.img_size):
                    labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                            for img in labels]
        else:
            pass


            
        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            if labels !=None:
                labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                        for img in labels]
            else:
                pass
            
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    for img in imgs]
        if labels != None:
            return imgs, labels
        else:
            return imgs



