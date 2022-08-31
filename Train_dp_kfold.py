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
import My_function as mf
from Network import My_Network
from collections import OrderedDict
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)


#10折交叉验证方法
k = 10
batch_size = 12
LR = 0.01
Epoch = 500
MAE = 4.7

#设置各路径
expdir = '/n02dat01/users/cchen/simple_ckpts2'
logdir = './logdir'
datapath = '/n02dat01/users/cchen/Train/*.nii'
labelpath = './label/Train.csv'
testdatapath = '/n02dat01/users/cchen/HCP/hcp_test/*.nii'
testlabelpath = './label/hcp_test.csv'
test2datapath = '/n02dat01/users/cchen/Test/*.nii'
test2labelpath = './label/Test.csv'

#创建实验目录以及日志目录
mf.create_exp_directory(expdir)
mf.create_exp_directory(logdir)

#将所有数据文件路径装入list中
path = sorted(glob.glob(datapath))
test_path = sorted(glob.glob(testdatapath))
test2_path = sorted(glob.glob(test2datapath))

#加载标签
label = mf.Load_Label(labelpath)
test_label = mf.Load_Label(testlabelpath)
test2_label = mf.Load_Label(test2labelpath)

#实例化测试集数据
test_data = Mydataset(test_path,test_label)
test_loader = DataLoader(dataset=test_data,batch_size=1)
test2_data = Mydataset(test2_path,test2_label)
test2_loader = DataLoader(dataset=test2_data,batch_size=1)

num_val_samples = len(path)//k
print("The number of validation sets is：",num_val_samples)

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
optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=0.001)
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

optimizer.param_groups[0]['lr']=0.0001

epoch_start = time.time()
while epoch < Epoch:
    epoch = epoch + 1
    k_train_loss = 0.0
    k_valid_loss = 0.0
    k_mae = 0.0
    scheduler.step()
    #k折交叉验证
    for p in range(k):
        item = 0
        k_path = path
        print('processing fold：#',p+1)
        
        #通过对路径划分切片实现训练集验证集的划分
        k_valid_data_path = k_path[p*num_val_samples:(p+1)*num_val_samples]
        k_train_data_path = k_path[:p*num_val_samples] + k_path[(p+1)*num_val_samples:]
        #得到验证集标签与训练集标签
        k_valid_label = label[p*num_val_samples:(p+1)*num_val_samples]
        k_train_label = label[:p*num_val_samples]+label[(p+1)*num_val_samples:]
 
        #实例化数据集类
        train_data = Mydataset(k_train_data_path,k_train_label)
        valid_data = Mydataset(k_valid_data_path,k_valid_label)

        train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
        valid_loader = DataLoader(dataset=valid_data,batch_size=1)
        
        running_loss = 0.0
        mae = 0.0
        n1 = len(train_loader)

        for i,(data_t, label_t) in enumerate(train_loader):
            train_x = data_t
            train_y = torch.LongTensor(label_t)
            train_y1 = train_y.float()
            #train_y1 = train_y.reshape(-1, 1)
            train_y1 = train_y1.cuda()
            train_x, train_y = train_x.cuda(), train_y.cuda()

            model.train()
            optimizer.zero_grad()
            pred_y = model(train_x)
            loss = Loss_func(pred_y,train_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
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
                logger.info('[Epoch:%d, Fold:%d, Iter:%d]Training loss: %.4f' % (epoch, p + 1, i + 1,loss.item()))
                logger.info('[Epoch:%d, Fold:%d, Iter:%d]Training MAE: %.4f' % (epoch, p + 1, i + 1,loss2.item()))   
            
            
        current_loss1 = running_loss/n1
        logger.info('[Epoch:%d, Fold:%d]Average Training loss: %.4f' % (epoch, p + 1, current_loss1))
        logger.info('[Epoch:%d, Fold:%d]Average Training MAE: %.4f' % (epoch, p + 1, mae/n1))
        k_train_loss += current_loss1
        k_mae += mae/n1
        
        model.eval()
        #验证集验证
        current_loss2 = mf.evaluate(model, valid_loader)
        logger.info('[Epoch:%d, Fold:%d]Average Validation mae: %.4f' % (epoch, p + 1, current_loss2))
        k_valid_loss += current_loss2
        if p == k-1:
            logger.info('*'* 50)
            logger.info('[Epoch:%d]:' % (epoch))
            logger.info('10 Fold Average train loss is: %.4f' % (k_train_loss/k))
            logger.info('10 Fold Average train mae is: %.4f' % (k_mae/k))
            logger.info('10 Fold Average validation mae is: %.4f' % (k_valid_loss/k))
            

    # 测试集测试
    current_loss3 = mf.evaluate(model,test_loader)
    current_loss4 = mf.evaluate(model, test2_loader)
    logger.info('[HCP]Average test mae is: %.4f' % (current_loss3))
    logger.info('[Test2]Average test2 mae is: %.4f' % (current_loss4))
    logger.info('*'*50)


    if epoch%2==0 or (current_loss3 < MAE and current_loss4 < MAE):
        save_name = os.path.join(expdir, 'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
        checkpoint = {"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                     "epoch":epoch,"MAE":current_loss3}
        torch.save(checkpoint, save_name)
        if MAE > current_loss3 and current_loss3 <= current_loss4:
            MAE = current_loss3
        if MAE > current_loss4 and current_loss4 < current_loss3:
            MAE = current_loss4
epoch_finish = time.time() - epoch_start
logger.info("Train {} Epoch finished in {:.04f} seconds.".format(Epoch, epoch_finish))






