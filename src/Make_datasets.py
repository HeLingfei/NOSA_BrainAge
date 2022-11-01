import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torch.utils.data as Data
import warnings
import My_function as mf
warnings.filterwarnings('ignore')


class Mydataset(Data.Dataset):
    def __init__(self, datapath, label, with_augmentation=False):
        super(Mydataset,self).__init__()
        imgs = []
        for i,current_path in enumerate(datapath):
            imgs.append((current_path,round(label[i])))
        self.imgs = imgs
        self.with_augmentation = with_augmentation
        
    def __getitem__(self, index):
        path, label_ = self.imgs[index]
        image = nib.load(path)
        image = np.asarray(image.get_fdata())
        image = image.astype('float32')
        #image = image / 255
        if self.with_augmentation:
            image = mf.augmention(image)
        image = image.reshape(1,121,145,121)
        image = torch.Tensor(image)
        return image, label_
    
    def __len__(self):
        return len(self.imgs)



class Mydataset_mask(Data.Dataset):
    def __init__(self, datapath, label, mask):
        super(Mydataset_mask,self).__init__()
        imgs = []
        for i,current_path in enumerate(datapath):
            imgs.append((current_path,round(label[i])))
        self.imgs = imgs
        self.mask = mask
        
    def __getitem__(self, index):
        path, label_ = self.imgs[index]
        image = nib.load(path)
        image = np.asarray(image.get_fdata())
        image = image.astype('float32')
        #img = img / 255
        image = image * self.mask
        image = image.reshape(1,121,145,121)
        image = torch.Tensor(image)
        return image, label_
    
    def __len__(self):
        return len(self.imgs)

