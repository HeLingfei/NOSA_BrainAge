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
from Network import My_Network
from collections import OrderedDict
#from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)


#划分训练集验证集训练

batch_size = 12
LR = 0.005
Epoch = 1000
MAE = 4.7
expdir = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge/results/simple_ckpts'
logdir = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge/results/train_logs'
datapath = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge/Train/*.nii'
labelpath = './label/Train.csv'
testdatapath = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge/Test/multisite_test/*.nii'
testlabelpath = './label/multi_center_test.csv'
test2datapath = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge/Test/hcp_test/*.nii'
test2labelpath = './label/hcp_test.csv'

#mf.set_random_seeds(6)
#创建实验目录以及日志目录
mf.create_exp_directory(expdir)
mf.create_exp_directory(logdir)

#将所有数据文件路径装入list中
train_path = sorted(glob.glob(datapath))
#test_path = sorted(glob.glob(testdatapath))
test2_path = sorted(glob.glob(test2datapath))

#加载标签
train_label = mf.Load_Label(labelpath)
#test_label = mf.Load_Label(testlabelpath)
test2_label = mf.Load_Label(test2labelpath)

#划分训练集验证集
#train_path, val_path, train_label, val_label = train_test_split(path, label, test_size=0.1, random_state=42)

#train_loader
train_data = Mydataset(train_path,train_label)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
n1 = len(train_loader)
#val_loader
#val_data = Mydataset(val_path,val_label)
#val_loader = DataLoader(dataset=val_data,batch_size=1)

#test_loader
#test_data = Mydataset(test_path,test_label)
#test_loader = DataLoader(dataset=test_data,batch_size=1)

test2_data = Mydataset(test2_path,test2_label)
test2_loader = DataLoader(dataset=test2_data,batch_size=1)

print("[The number of Training sample is:]-------------->[{}]".format(len(train_data)))
#print("[The number of validation sample is:]-------------->[{}]".format(len(val_data)))
print("[The number of Testing sample is:]-------------->[{}]".format(len(test2_data)))


#1.创建logger
logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
#将handlers清空防止输出两次信息
logger.handlers = []

#2.创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
logger.addHandler(ch)

#3.创建另一个handler，用于写入日志文件
fh = logging.FileHandler(os.path.join(logdir, "log.txt"))
logger.addHandler(fh)

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



#print("Current LR:",optimizer.param_groups[0]['lr'])
optimizer.param_groups[0]['lr']=0.001
#print("Current LR:",optimizer.param_groups[0]['lr'])


epoch_start = time.time()
while epoch < Epoch:
    epoch = epoch + 1
    scheduler.step()
    #running_loss = 0.0
    mae = 0.0
    for i,(data_t, label_t) in enumerate(train_loader):
        train_x = data_t
        train_y = torch.LongTensor(label_t)
        train_y1 = train_y.float()
        #train_y1 = train_y1.reshape(-1,1)
        train_y1 = train_y1.cuda()
        train_x, train_y = train_x.cuda(), train_y.cuda()

        model.train()
        optimizer.zero_grad()
        pred_y = model(train_x)
        loss = Loss_func(pred_y,train_y)
        loss.backward()
        optimizer.step()
        #running_loss += loss.item()

        #通过softmax输出各类概率，然后乘上各自代表的年龄类再累加
        result = softmax(pred_y)
        pred = []
        for j in range(len(train_x)):
            sum_ = 0
            for n in range(73):
                sum_ += result[j][n] * n
            pred.append(sum_)
        pred = torch.Tensor(pred)
        pred = pred.cuda()
        loss2 = Loss_func2(pred,train_y1)
        mae += loss2.item()
        
        if i % 10 == 9:
            logger.info('Epoch [{}/{}], Step [{}/{}], MAE: {:.4f}'.format(
                epoch,
                Epoch,
                i + 1,
                n1,
                mae / (i + 1))
            )
        del train_x, train_y, train_y1, pred_y, loss, result, pred, loss2

    current_loss1 = mae/n1
    logger.info('*' * 60)
    logger.info('[Epoch:%d]Average training mae is: %.4f' % (epoch, current_loss1))

    #model.eval()
    # 验证集测试
    #current_loss2 = mf.evaluate(model,val_loader)
    #logger.info('[Epoch:%d]Average validation mae is: %.4f' % (epoch, current_loss2))

    # 测试集测试
    #current_loss3 = mf.evaluate(model,test_loader)
    current_loss4 = mf.evaluate(model, test2_loader)
    logger.info('[Epoch:%d]Average(HCP) test mae is: %.4f' % (epoch, current_loss4))
    logger.info('*' * 60)


    if epoch%5==0 or (current_loss4 < MAE):
        save_name = os.path.join(expdir, 'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
        checkpoint = {"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(), "epoch":epoch,"MAE":current_loss4}
        torch.save(checkpoint, save_name)
        if MAE > current_loss4:
            MAE = current_loss4
    

epoch_finish = time.time() - epoch_start
logger.info("Training {} Epoch finished in {:.04f} seconds.".format(Epoch, epoch_finish))


