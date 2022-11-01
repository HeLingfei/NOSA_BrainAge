import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
import math
import glob
import time
import xlwt
import My_function as mf
import nibabel as nib
from Network import My_Network
from Make_datasets import Mydataset, Mydataset_mask
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict

torch.set_default_tensor_type(torch.FloatTensor)


def load_model(path):
    # 加载训练好的模型
    model = My_Network()
    try:
        # prior_model_path = sorted(glob.glob(os.path.join(expdir, 'Epoch_*')), key=os.path.getmtime)

        # if prior_model_path:
        #     current_model_path = prior_model_path.pop()
        # current_model_path = r'C:\Users\h4619\Desktop\simple_ckpts2\Epoch_253_training_state.pkl'
        state = torch.load(path)

        # 加载模型参数
        # 多gpu
        if torch.cuda.device_count() > 1:
            model.load_state_dict(state["model_state_dict"])
        # 单gpu
        else:
            state_dict = OrderedDict()
            for k, v in state["model_state_dict"].items():
                name = k
                if name[:7] == 'module.':  # loaded model is multiple GPUs but we will train it in single GPU!
                    name = k[7:]  # remove 'module.'，从第7个key值字符取到最后一个字符，正好去掉了'module.'
                    state_dict[name] = v
                else:
                    state_dict[name] = v
            model.load_state_dict(state_dict)
        print("Successfully restored the model state.")
        if torch.cuda.is_available():
            # torch.backends.cudnn.benchmark = True
            model.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
    except Exception as e:
        print("Failed to load the model...... {}".format(e))
        return None
    return model


def load_data(data_path, label_path, batch_size=1):
    test_path = sorted(glob.glob(data_path))
    test_label = mf.Load_Label(label_path)
    test_data = Mydataset(test_path, test_label)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)
    return test_loader, test_label

def test_model(model, data_path, label_path, print_result=True):
    batch_size = 1
    test_loader, test_label = load_data(data_path, label_path, batch_size=batch_size)
    num = len(test_loader)

    # start = time.time()
    # Select the model

    # print('<-------------------Starting to test!-------------------->')

    # 不删除脑区测试
    Allsub_res = np.zeros((num, 2))  # 行数是输入组别被试个数
    pred_single_sub_age = []
    mae, pred_single_sub_age = mf.evaluate(model, test_loader, True)
    r = mf.compute_r(pred_single_sub_age, test_label)
    result = {
        'MAE': float(mae.cpu().float()),
        'r': r,
        'R2': r ** 2
    }
    if print_result:
        print(result)
    return result


def test_model_by_two_test_datasets(model_path):
    model = load_model(model_path)
    datapath = '/n02dat01/users/cchen/HCP/hcp_test/*.nii'
    labelpath = '/n02dat01/users/cchen/HCP/hcp_test/hcp_test.csv'
    base_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73'
    expdir = base_path
    hcp_test_data_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73\HCP\hcp_test\*.nii'
    hcp_test_label_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73\HCP\hcp_test\hcp_test.csv'

    multisite_test_data_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\数据\多中心数据集\dataset\Test\*.nii'
    multisite_test_label_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\数据\多中心数据集\dataset\Test\Test.csv'

    print('HCP test:')
    test_model(model, data_path=hcp_test_data_path, label_path=hcp_test_label_path)
    print('Multisite test:')
    test_model(model, data_path=multisite_test_data_path, label_path=multisite_test_label_path)

# expdir = r'C:\Users\h4619\Desktop/simple_ckpts3/*.pkl'
# model_paths = sorted(glob.glob(expdir))
# for path in model_paths:
#     print()
#     print(os.path.basename(path))
#     test_model_by_two_test_datasets(path)


# 保存被试真实年龄和预测年龄，结果要+8（实际年龄从8岁开始，为了第0类代表0岁，我们将所有被试年龄都减去了8，因此要加回来）
# for i in range(0, num):
#     Allsub_res[i][0] = test_label[i] + 8
#     Allsub_res[i][1] = pred_single_sub_age[i] + 8
#
# try:
#     dataframe = pd.DataFrame(Allsub_res)
#     dataframe.to_excel(r'C:\Users\h4619\Desktop/Lifespan_pred_age.xls')
#     print("Successfully save the result to Result.xls!!!")
# except Exception as e:
#     print("Failed to save the result...... {}".format(e))

# 删除单个脑区测试，生成当前输入组别所有被试246个脑区的平均IS值
# Result = np.zeros((246,2))
# atlas = nib.load('./BN_Atlas_246_combined_15mm.nii')
# atlas = np.asarray(atlas.get_fdata(),dtype=np.float32)
# max_mae_diff = 0
# min_mae_diff = 100
# for i in range(1,247):
#     print("Processing No.{} brain area......".format(i))
#     mask = mf.make_single_mask(atlas,i)
#     test_data = Mydataset_mask(test_path,test_label,mask)
#     test_loader = DataLoader(dataset=test_data,batch_size=batch_size)
#     MAE = mf.evaluate(model,test_loader,False)
#     result = abs((MAE-mae).item())
#     Result[i-1][0] = result
#     if max_mae_diff < result:
#         max_mae_diff = result
#     if min_mae_diff > result:
#         min_mae_diff = result
#     del test_data,test_loader
#
# for k in range(1,247):
#     Result[k-1][1] = (Result[k-1][0] - min_mae_diff)/(max_mae_diff - min_mae_diff)
#
# try:
#     dataframe = pd.DataFrame(Result)
#     dataframe.to_excel('/n02dat01/users/cchen/simple_brain_age_73/Result/HCP_Lifespan_IS.xls')
#     print("Successfully save the result to Result.xls!!!")
# except Exception as e:
#      print("Failed to save the result...... {}".format(e))
