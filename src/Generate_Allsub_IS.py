import torch
import torch.nn as nn
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
import math
import glob
import time
import xlsxwriter
import My_function as mf
import nibabel as nib
from Network import My_Network
from Make_datasets import Mydataset,Mydataset_mask
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
torch.set_default_tensor_type(torch.FloatTensor)

# In[2]:
mf.set_random_seeds(42)


batch_size = 1
expdir = '/n02dat01/users/cchen/simple_ckpts2'
datapath = '/n02dat01/users/cchen/HCP/hcp_test/*.nii'
labelpath = '/n02dat01/users/cchen/HCP/hcp_test/hcp_test.csv'

test_path = sorted(glob.glob(datapath))
test_label = mf.Load_Label(labelpath)
test_data = Mydataset(test_path,test_label)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size)
num = len(test_loader)


start = time.time()
# Select the model
model = My_Network()

if torch.cuda.is_available():
    #torch.backends.cudnn.benchmark = True
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

print('<-------------------Starting to test!-------------------->')
#加载训练好的模型
try:
    prior_model_path = sorted(glob.glob(os.path.join(expdir,'Epoch_*')),key=os.path.getmtime)

    if prior_model_path:
        current_model_path = prior_model_path.pop()
    state = torch.load(current_model_path)

    #加载模型参数
    #多gpu
    if torch.cuda.device_count() > 1:
        model.load_state_dict(state["model_state_dict"])
    #单gpu
    else: 
        state_dict = OrderedDict()
        for k,v in state["model_state_dict"].items():
            name = k
            if name[:7] == 'module.': # loaded model is multiple GPUs but we will train it in single GPU!
                name = k[7:] # remove 'module.'，从第7个key值字符取到最后一个字符，正好去掉了'module.'
                state_dict[name] = v
            else:
                state_dict[name] = v
        model.load_state_dict(state_dict)
    print("Successfully restored the model state.")

except Exception as e:
    print("Failed to load the model...... {}".format(e))


#不删除脑区测试
Allsub_res = np.zeros((555,247))
pred_single_sub_age = []
mae,pred_single_sub_age = mf.evaluate(model, test_loader, True)
#保存每个被试的mae放到结果集第一列，用于后续生成每个被试各脑区的IS值。
for i in range(0,555):
    Allsub_res[i][0] = abs(pred_single_sub_age[i]-test_label[i])

#删除单个脑区测试
atlas = nib.load('../analytical_data/BN_Atlas_246_combined_15mm.nii')
atlas = np.asarray(atlas.get_fdata(),dtype=np.float32)

for i in range(1,247):
    print("Processing No.{} brain area......".format(i))
    mask = mf.make_single_mask(atlas,i)
    test_data = Mydataset_mask(test_path,test_label,mask)
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size)
    MAE, pred_single_sub_age = mf.evaluate(model,test_loader, True)
    for j in range(0, 555):
        Allsub_res[j][i] = abs(abs(pred_single_sub_age[j]-test_label[j])-Allsub_res[j][0]);

    del test_data,test_loader

#归一化求IS值
for i in range(0,555):
    max = Allsub_res[i][1]
    min = Allsub_res[i][1]
    for j in range(2,247):
        if Allsub_res[i][j] > max:
            max = Allsub_res[i][j]
        if Allsub_res[i][j] < min:
            min = Allsub_res[i][j]
    for k in range(1,247):
        Allsub_res[i][k] = (Allsub_res[i][k]-min)/(max-min)

try:
    writer = pd.ExcelWriter('./Liver_Allsub_IS.xlsx',engine='xlsxwriter')
    dataframe = pd.DataFrame(Allsub_res)
    dataframe.to_excel(writer)
    writer.save()
    print("Successfully save the result to ***.xlsx!!!")
except Exception as e:
     print("Failed to save the result...... {}".format(e))

