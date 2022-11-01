import os
import torch
import torch.nn as nn
import numpy as np
from Make_datasets import Mydataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import glob
import math
import time
import logging
import warnings
import random
import My_function as mf
import nibabel as nib
from Network import My_Network
from collections import OrderedDict
#from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)

#划分训练集验证集训练
batch_size = 1
LR = 0.001
expdir = '/n02dat01/users/cchen/simple_ckpts'
datapath = '/n02dat01/users/cchen/HCP/hcp_test/*.nii'
labelpath = '/n02dat01/users/cchen/HCP/hcp_test/hcp_test.csv'

#将所有数据文件路径装入list中
train_path = sorted(glob.glob(datapath))

#加载标签
train_label = mf.Load_Label(labelpath)

train_data = Mydataset(train_path,train_label)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
n1 = len(train_loader)


model = My_Network()
#多gpu
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


softmax = nn.Softmax(dim=1)
Loss_func = nn.CrossEntropyLoss() 
Loss_func2 = nn.L1Loss()
#if torch.cuda.is_available():
#    softmax.cuda()
#    Loss_func.cuda()
#    Loss_func2.cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=0.001)
#optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.3)

epoch = 0
print('<-------------------Starting to train!-------------------->')
#恢复之前训练的模型
try:
    prior_model_path = sorted(glob.glob(os.path.join(expdir,'Epoch_*')),key=os.path.getmtime)

    if prior_model_path:
        current_model = prior_model_path.pop()
    state = torch.load(current_model)

    # 加载模型参数
    # 多gpu
    if torch.cuda.device_count() > 1:
        model.load_state_dict(state["model_state_dict"])
    # 单gpu
    else:
        state_dict = OrderedDict()
        for m, v in state["model_state_dict"].items():
            name = m
            if name[:7] == 'module.':  # loaded model is multiple GPUs but we will train it in single GPU!
                name = m[7:]  # remove 'module.'，从第7个key值字符取到最后一个字符，正好去掉了'module.'
                state_dict[name] = v
            else:
                state_dict[name] = v
        model.load_state_dict(state_dict)

    # 加载优化器和epoch信息
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state["epoch"]
    MAE = state["MAE"]
      
    print("Successfully restored the model state. Resuming training from Epoch {}".format(epoch + 1))

except Exception as e:
    print("No model to restore. Resuming training from Epoch 0. {}".format(e))


grad_sum = np.zeros((121,145,121));
for i,(data_t, label_t) in enumerate(train_loader):
    train_x = data_t
    train_y = torch.LongTensor(label_t)
    train_x, train_y = train_x.cuda(), train_y.cuda()
    train_x.requires_grad_(True)
    model.train()
    optimizer.zero_grad()
    pred_y = model(train_x)
    loss = Loss_func(pred_y,train_y)
    train_x.retain_grad()
    loss.backward()
    grad = train_x.grad.reshape(121,145,121)
    grad = np.array(grad.cpu())
    grad_sum += grad*10000
    optimizer.step()

grad_sum = abs(grad_sum)
#体素的梯度
atlas = nib.load('../data/BN_Atlas_246_combined_15mm.nii')
affine = atlas.affine
atlas = np.asarray(atlas.get_fdata(),dtype=np.float32)
result = grad_sum/len(train_path)
result1 = result
result = nib.Nifti1Image(result,affine)
nib.save(result,'./Lifespan_voxel_grad.nii')

#脑区的梯度，求脑区总体素的平均值
excel_res = np.zeros((246,2))
res = np.zeros((121,145,121))
for i in range(1,247):
    a = result1
    temp = (atlas == i)
    a = a * temp.astype(int)
    count = 0
    gradsum = 0
    for n in range(0,121):
        for j in range(0,145):
            for k in range(0,121):
                if(a[n][j][k]!=0):
                    count += 1
                    gradsum += a[n][j][k]
    excel_res[i-1][0] += gradsum/count

max = excel_res[0][0]
min = excel_res[0][0]
for j in range(1,246):
    if excel_res[j][0] > max:
        max = excel_res[j][0]
    if excel_res[j][0] < min:
        min = excel_res[j][0]
for k in range(0,246):
    excel_res[k][1] = (excel_res[k][0] - min) / (max - min)

for i in range(1,247):
    temp = (atlas == i)
    res += (temp.astype(int) * excel_res[i-1][1])

res = nib.Nifti1Image(res,affine)
nib.save(res,'./Lifespan_brain_region_grad_socre.nii')
writer = pd.ExcelWriter('./Lifespan_BNA_246_grad.xls',engine='xlsxwriter')
dataframe = pd.DataFrame(excel_res)
dataframe.to_excel(writer)
writer.save()