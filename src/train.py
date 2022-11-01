import glob
import logging
import os
import time
import warnings

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
import uuid
import My_function as mf

from Make_datasets import Mydataset
from Network import My_Network

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)

# 10折交叉验证方法
k = 10
batch_size = 12
LR = 0.001
Epoch = 400
MAE = 8
# early_stop_epoch_num = 50

base_data_dir = '/HOME/scz0774/run/lfhe/analytical_data/SimpleBrainAge'

datapath = f'{base_data_dir}/Train/*.nii'
labelpath = f'{base_data_dir}/Train/Train.csv'

testdatapath = f'{base_data_dir}/Test/multisite_test/*.nii'
testlabelpath = f'{base_data_dir}/Test/multisite_test/Test.csv'

test2datapath = f'{base_data_dir}/Test/hcp_test/*.nii'
test2labelpath = f'{base_data_dir}/Test/hcp_test/hcp_test.csv'

base_results_dir = '/HOME/scz0774/run/lfhe/analytical_data/SimpleBrainAge/results'
# base_results_dir = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73\Result'
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

experiment_id = uuid.uuid1()


def get_data_loader_by_indexes(paths, labels, indexes, data_augment, b_size):
    paths = np.array(paths)
    labels = np.array(labels)
    dataset = Mydataset(paths[indexes], labels[indexes], with_augmentation=data_augment)
    return DataLoader(dataset=dataset, batch_size=b_size, shuffle=True)


def get_model(model_class):
    m = model_class()
    # 多gpu
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        m.cuda()
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
    return m


def save_model(m, extra):
    save_path = os.path.join(expdir, extra + f'_{experiment_id}_training_state.pth')
    torch.save(m.state_dict(), save_path)


hcp_test_dataloader = get_data_loader_by_indexes(paths=test2_path, labels=test2_label,
                                                 indexes=list(range(len(test2_path))), data_augment=False, b_size=1)
multisite_test_dataloader = get_data_loader_by_indexes(paths=test_path, labels=test_label,
                                                       indexes=list(range(len(test_path))), data_augment=False,
                                                       b_size=1)

# 1.创建logger
logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
# 将handlers清空防止输出两次信息
logger.handlers = []

# 2.创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
logger.addHandler(ch)

# 3.创建另一个handler，用于写入日志文件
fh = logging.FileHandler(os.path.join(logdir, f'log_{experiment_id}.txt'))
logger.addHandler(fh)

softmax = nn.Softmax(dim=1)
Loss_func = nn.CrossEntropyLoss()
Loss_func2 = nn.L1Loss()
kf = KFold(n_splits=k, shuffle=True)
logger.info('*' * 50)
logger.info(
    f'Starting Training\n{experiment_id}, {k}_fold\nbatch_size: {batch_size}\nlr: {LR}\nEpoch: {Epoch}')
logger.info('*' * 50)
for kf_index, (train_indexes, validate_indexes) in enumerate(kf.split(path)):
    model = get_model(My_Network)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    train_dataloader = get_data_loader_by_indexes(paths=path, labels=label, indexes=train_indexes, data_augment=True,
                                                  b_size=batch_size)
    validate_dataloader = get_data_loader_by_indexes(paths=path, labels=label, indexes=train_indexes,
                                                     data_augment=False, b_size=1)
    best_MAE = MAE
    for epoch in range(Epoch):
        epoch_loss = 0
        epoch_mae = 0
        model.train()
        for i, (data_t, label_t) in enumerate(train_dataloader):
            train_x = data_t
            train_y = torch.LongTensor(label_t)
            train_y1 = train_y.float().cuda()
            train_x, train_y = train_x.cuda(), train_y.cuda()
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
                logger.info(f'[Fold:{kf_index + 1}, Epoch:{epoch + 1}, Left_Iter:{len(train_dataloader) - (i + 1)}] '
                            f'Loss:{"%.4f" % loss.item()}, MAE:{"%.4f" % loss2.item()}')

        model.eval()
        logger.info('[Fold:%d, Epoch:%d] Average Training loss: %.4f' % (kf_index + 1, epoch + 1,
                                                                         epoch_loss / len(train_dataloader)))
        logger.info('[Fold:%d, Epoch:%d] Average Training MAE: %.4f' % (kf_index + 1, epoch + 1,
                                                                        epoch_mae / len(train_dataloader)))
        # 保存验证集验证最优的模型
        validate_mae = mf.evaluate(model, validate_dataloader)
        logger.info('[Fold: %d, Epoch: %d] Validation MAE: %.4f' % (kf_index + 1, epoch + 1, validate_mae))
        if validate_mae < best_MAE:
            save_model(model, extra=f'Fold_{kf_index + 1}_Epoch_{epoch + 1}')
            best_MAE = validate_mae
            early_stop_counter = 0
            # 记录此时的测试指标
            hcp_test_mae = mf.evaluate(model, hcp_test_dataloader)
            multisite_test_mae = mf.evaluate(model, multisite_test_dataloader)

            logger.info('*' * 25 + ' Better Model ' + '*' * 25)
            logger.info('[HCP Test] MAE is: %.4f' % hcp_test_mae)
            logger.info('[Multisite Test] MAE is: %.4f' % multisite_test_mae)
            logger.info('*' * 25 + ' Better Model ' + '*' * 25)
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= early_stop_epoch_num:
        #         logger.info('*' * 50)
        #         logger.info(f'Early Stopping: {kf_index + 1} Fold, {epoch + 1} Epoch')
        #         logger.info('*' * 50)
        #         break
