import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import math
# import matplotlib.pyplot as plt
import cv2
import random

torch.set_default_tensor_type(torch.FloatTensor)


def create_exp_directory(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Successfully Created Directory @ {}".format(dir_name))
        except:
            print("Directory Creation Failed - Check Path")
    else:
        print("Directory {} Exists".format(dir_name))


# 读取label
def Load_Label(path):
    y = pd.read_csv(path, encoding='GBK')
    label = y.values
    label = label[:, 2] - 8
    label = label.astype('long')
    label = label.tolist()

    return label


# In[4]:
def set_random_seeds(random_seed=42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def compute_r(x, y):
    x_hat = np.mean(x)
    y_hat = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(len(x)):
        x_xhat = x[i] - x_hat
        y_yhat = y[i] - y_hat
        SSR += (x_xhat * y_yhat)
        varX += x_xhat ** 2
        varY += y_yhat ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST


def evaluate(model, test_loader, tag=False):
    softmax = nn.Softmax(dim=1)
    loss_func = nn.L1Loss()
    model.eval()
    if torch.cuda.is_available():
        softmax.cuda()
        loss_func.cuda()
    with torch.no_grad():
        mae = 0.0
        pre_out = []
        for i, (data_v, label_v) in enumerate(test_loader):
            test_x = data_v
            test_y = label_v.float()
            # test_y = test_y.reshape(-1,1)
            test_x, test_y = test_x.cuda(), test_y.cuda()
            v_pred_y = model.forward(test_x)
            result = softmax(v_pred_y)
            # 通过softmax输出各类概率，然后乘上各自代表的年龄类再累加
            pred = []
            for j in range(len(test_x)):
                sum_ = 0
                for k in range(73):
                    sum_ += result[j][k] * k
                pred.append(sum_)

            pred = torch.Tensor(pred)
            pre_out.append(pred.reshape(1).item())
            pred = pred.cuda()
            v_loss = loss_func(test_y, pred)
            mae += v_loss
            del test_x, test_y, v_pred_y, result, pred, v_loss
    mae = mae / len(test_loader)
    if tag:
        return mae, pre_out
    else:
        return mae


# 单一脑区mask
def make_single_mask(atlas_img, roi):
    mask = (atlas_img == roi)
    mask = (mask.astype(int) == 0)

    return mask.astype(int)


# 多脑区mask，传入脑区编号列表
def make_multi_mask(atlas_img, roi_list):
    mask = np.zeros((atlas_img.shape))
    for roi in roi_list:
        temp = (atlas_img == roi)
        mask += temp.astype(int)
    mask = (mask == 0).astype(int)

    return mask


def translation(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img


def rotation(img, width, height, angle):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img = cv2.warpAffine(img, M, (width, height))
    return img


def augmention(image):
    # 数据增强，平移，旋转
    # i表示三个维度
    i = random.randint(0, 2)
    # choice表示是平移还是旋转
    choice = random.randint(0, 1)
    # x,y表示沿x,y平移的距离
    x = random.randint(-5, 5)
    y = random.randint(-5, 5)
    # angle表示旋转的角度
    angle = random.randint(-20, 20)

    if i == 0:
        for j in range(image.shape[i]):
            if choice == 0:
                img = image[j, :, :]
                image[j, :, :] = translation(img, x, y)
            else:
                img = image[j, :, :]
                image[j, :, :] = rotation(img, image.shape[2], image.shape[1], angle)
    elif i == 1:
        for j in range(image.shape[i]):
            if choice == 0:
                img = image[:, j, :]
                image[:, j, :] = translation(img, x, y)
            else:
                img = image[:, j, :]
                image[:, j, :] = rotation(img, image.shape[2], image.shape[0], angle)
    else:
        for j in range(image.shape[i]):
            if choice == 0:
                img = image[:, :, j]
                image[:, :, j] = translation(img, x, y)
            else:
                img = image[:, :, j]
                image[:, :, j] = rotation(img, image.shape[1], image.shape[0], angle)
    return image
