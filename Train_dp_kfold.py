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
from sklearn.model_selection import KFold
import uuid
from train import get_model, get_data_loader_by_indexes, save_model

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)

# 10折交叉验证方法
k = 10
batch_size = 12
LR = 0.005
Epoch = 1000
MAE = 100

current_id = uuid.uuid1()
base_data_dir = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge'
# base_data_dir = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\数据'

datapath = f'{base_data_dir}/Train/*.nii'
labelpath = f'{base_data_dir}/Train/Train.csv'

testdatapath = f'{base_data_dir}/Test/multisite_test/*.nii'
testlabelpath = f'{base_data_dir}/Test/multisite_test/Test.csv'

test2datapath = f'{base_data_dir}/Test/hcp_test/*.nii'
test2labelpath = f'{base_data_dir}/Test/hcp_test/hcp_test.csv'

base_results_dir = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge/results'
# base_results_dir = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73' \
#                    r'\Result'
expdir = f'{base_results_dir}/simple_ckpts'
logdir = f'{base_results_dir}/train_logs'

# 创建实验目录以及日志目录
mf.create_exp_directory(expdir)
mf.create_exp_directory(logdir)

# 将所有数据文件路径装入list中
path = sorted(glob.glob(datapath))
test_path = sorted(glob.glob(testdatapath))
test2_path = sorted(glob.glob(test2datapath))

# 加载标签
label = mf.Load_Label(labelpath)
test_label = mf.Load_Label(testlabelpath)
test2_label = mf.Load_Label(test2labelpath)

# 实例化测试集数据
test_data = Mydataset(test_path, test_label)
test_loader = DataLoader(dataset=test_data, batch_size=1)
test2_data = Mydataset(test2_path, test2_label)
test2_loader = DataLoader(dataset=test2_data, batch_size=1)

num_val_samples = len(path) // k
print("The number of validation sets is：", num_val_samples)

# 1.创建logger
logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
# 将handlers清空防止输出两次信息
logger.handlers = []

# 2.创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
logger.addHandler(ch)

# 3.创建另一个handler，用于写入日志文件
fh = logging.FileHandler(os.path.join(logdir, f'log_{current_id}.txt'))
logger.addHandler(fh)

softmax = nn.Softmax(dim=1)
Loss_func = nn.CrossEntropyLoss()
Loss_func2 = nn.L1Loss()

print(f'<-------------------Starting to train {current_id}-------------------->')
epoch_start = time.time()
kf = KFold(n_splits=10, shuffle=True)


validation_maes = []
hcp_test_maes = []
multisite_test_maes = []
for kf_index, (train_indexes, validate_indexes) in enumerate(kf.split(path)):
    model = get_model(My_Network)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    train_dataloader = get_data_loader_by_indexes(paths=path, labels=label, indexes=train_indexes, data_augment=True,
                                                  b_size=batch_size)
    validate_dataloader = get_data_loader_by_indexes(paths=path, labels=label, indexes=train_indexes,
                                                     data_augment=False, b_size=1)
    running_loss = 0.0
    mae = 0.0
    n1 = len(train_dataloader)

    for epoch in range(Epoch):
        epoch_loss = 0
        epoch_mae = 0
        for i, (data_t, label_t) in enumerate(train_dataloader):
            train_x = data_t
            train_y = torch.LongTensor(label_t)
            train_y1 = train_y.float()
            train_y1 = train_y1.cuda()
            train_x, train_y = train_x.cuda(), train_y.cuda()

            model.train()
            optimizer.zero_grad()
            pred_y = model(train_x)
            loss = Loss_func(pred_y, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # 通过softmax输出各类概率，然后乘上各自代表的年龄类再累加
            result = softmax(pred_y)
            pred = []
            for j in range(len(train_x)):
                sum_ = 0
                for n in range(73):
                    sum_ += result[j][n] * n
                pred.append(sum_)
            pred = torch.Tensor(pred)
            pred = pred.cuda()
            loss2 = Loss_func2(pred, train_y1)
            epoch_mae += loss2.item()

            if i % 10 == 9:
                logger.info('Training loss: %.4f' % loss.item())
                logger.info('Training MAE: %.4f' % loss2.item())

        logger.info('[Epoch:%d, Fold:%d, Iter:%d] Average Training loss: %.4f' % (epoch, kf_index + 1, i + 1,
                                                                                  epoch_loss / len(train_dataloader)))
        logger.info('[Epoch:%d, Fold:%d, Iter:%d] Average Training MAE: %.4f' % (epoch, kf_index + 1, i + 1,
                                                                                 epoch_mae / len(train_dataloader)))

    model.eval()
    # 验证集验证
    validate_mae = mf.evaluate(model, validate_dataloader)
    hcp_test_mae = mf.evaluate(model, test_loader)
    multisite_test_mae = mf.evaluate(model, test2_loader)

    logger.info('*' * 50)
    logger.info('[Fold:%d] Validation MAE: %.4f' % (epoch, kf_index + 1, validate_mae))
    logger.info('[HCP Test] MAE is: %.4f' % hcp_test_mae)
    logger.info('[Multisite Test] MAE is: %.4f' % multisite_test_mae)
    logger.info('*' * 50)

    validation_maes.append(validate_mae)
    hcp_test_maes.append(hcp_test_mae)
    multisite_test_maes.append(multisite_test_mae)


epoch_finish = time.time() - epoch_start
logger.info(f"Train {current_id}  finished in {'%.4f' % epoch_finish} seconds.")

logger.info('-' * 50)
logger.info('[Average Validation] MAE is: %.4f' % np.mean(validation_maes))
logger.info('[Average HCP Test] MAE is: %.4f' % np.mean(hcp_test_maes))
logger.info('[Average Multisite Test] MAE is: %.4f' % np.mean(multisite_test_maes))
logger.info('-' * 50)

# while epoch < Epoch:
#     epoch = epoch + 1
#     k_train_loss = 0.0
#     k_valid_loss = 0.0
#     k_mae = 0.0
#     scheduler.step()
#     # k折交叉验证
#     for p in range(k):
#         item = 0
#         k_path = path
#         print('processing fold：#', p + 1)
#
#         # 通过对路径划分切片实现训练集验证集的划分
#         k_valid_data_path = k_path[p * num_val_samples:(p + 1) * num_val_samples]
#         k_train_data_path = k_path[:p * num_val_samples] + k_path[(p + 1) * num_val_samples:]
#         # 得到验证集标签与训练集标签
#         k_valid_label = label[p * num_val_samples:(p + 1) * num_val_samples]
#         k_train_label = label[:p * num_val_samples] + label[(p + 1) * num_val_samples:]
#
#         # 实例化数据集类
#         train_data = Mydataset(k_train_data_path, k_train_label, with_augmentation=True)
#         valid_data = Mydataset(k_valid_data_path, k_valid_label)
#
#         train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#         valid_loader = DataLoader(dataset=valid_data, batch_size=1)
#
#         running_loss = 0.0
#         mae = 0.0
#         n1 = len(train_loader)
#
#         for i, (data_t, label_t) in enumerate(train_loader):
#             train_x = data_t
#             train_y = torch.LongTensor(label_t)
#             train_y1 = train_y.float()
#             # train_y1 = train_y.reshape(-1, 1)
#             train_y1 = train_y1.cuda()
#             train_x, train_y = train_x.cuda(), train_y.cuda()
#
#             model.train()
#             optimizer.zero_grad()
#             pred_y = model(train_x)
#             loss = Loss_func(pred_y, train_y)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#             # 通过softmax输出各类概率，然后乘上各自代表的年龄类再累加
#             result = softmax(pred_y)
#             pred = []
#             for j in range(len(train_x)):
#                 sum_ = 0
#                 for n in range(73):
#                     sum_ += result[j][n] * n
#                 pred.append(sum_)
#             pred = torch.Tensor(pred)
#             pred = pred.cuda()
#             loss2 = Loss_func2(pred, train_y1)
#             mae += loss2.item()
#
#             if i % 10 == 9:
#                 logger.info('[Epoch:%d, Fold:%d, Iter:%d]Training loss: %.4f' % (epoch, p + 1, i + 1, loss.item()))
#                 logger.info('[Epoch:%d, Fold:%d, Iter:%d]Training MAE: %.4f' % (epoch, p + 1, i + 1, loss2.item()))
#
#         current_loss1 = running_loss / n1
#         logger.info('[Epoch:%d, Fold:%d]Average Training loss: %.4f' % (epoch, p + 1, current_loss1))
#         logger.info('[Epoch:%d, Fold:%d]Average Training MAE: %.4f' % (epoch, p + 1, mae / n1))
#         k_train_loss += current_loss1
#         k_mae += mae / n1
#
#         model.eval()
#         # 验证集验证
#         current_loss2 = mf.evaluate(model, valid_loader)
#         logger.info('[Epoch:%d, Fold:%d]Average Validation mae: %.4f' % (epoch, p + 1, current_loss2))
#         k_valid_loss += current_loss2
#
#     avg_valid_mae = k_valid_loss / k
#     logger.info('-' * 50)
#     logger.info('10 Fold Average train loss is: %.4f' % (k_train_loss / k))
#     logger.info('10 Fold Average train mae is: %.4f' % (k_mae / k))
#     logger.info('10 Fold Average validation mae is: %.4f' % avg_valid_mae)
#     logger.info('-' * 50)
#
#     if avg_valid_mae < MAE:
#         logger.info('*' * 50)
#         logger.info('[Epoch:%d]:' % epoch)
#         # 保存在验证集上MAE最优的模型
#         MAE = avg_valid_mae
#         save_name = os.path.join(expdir, 'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
#         checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
#                       "epoch": epoch, "MAE": MAE}
#         torch.save(checkpoint, save_name)
#         # 每次保存时最优模型时顺便保存此时的测试指标
#         current_loss3 = mf.evaluate(model, test_loader)
#         current_loss4 = mf.evaluate(model, test2_loader)
#         logger.info('[HCP Test] MAE is: %.4f' % (current_loss3))
#         logger.info('[Multisite Test] MAE is: %.4f' % (current_loss4))
#         logger.info('*' * 50)
#     #
#     # if epoch % 2 == 0 or (current_loss3 < MAE and current_loss4 < MAE):
#     #     save_name = os.path.join(expdir, 'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
#     #     checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
#     #                   "epoch": epoch, "MAE": current_loss3}
#     #     torch.save(checkpoint, save_name)
#     #     if MAE > current_loss3 and current_loss3 <= current_loss4:
#     #         MAE = current_loss3
#     #     if MAE > current_loss4 and current_loss4 < current_loss3:
#     #         MAE = current_loss4
# epoch_finish = time.time() - epoch_start
# logger.info("Train {} Epoch finished in {:.04f} seconds.".format(Epoch, epoch_finish))
