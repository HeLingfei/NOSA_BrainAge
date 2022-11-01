import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import torch
from sklearn.model_selection import KFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import pandas as pd
from patsy.highlevel import dmatrices
from Test_Mymodel_GroupIS import load_model, load_data
from torch.utils.data.dataloader import DataLoader
# class TestModule(nn.Module):
#     def __init__(self):
#         super(TestModule, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.relu(self.conv1(x))
#
# import ssl
# # ssl._create_default_https_context = ssl._create_unverified_context
# df = sm.datasets.get_rdataset("Guerry", "HistData", cache=True).data # type: pd.DataFrame
# df = df.dropna()
# print('数据加载完成')
# y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
# print('设计的数据矩阵转换完成')
# mod = sm.OLS(y, X)
# print('模型建立')
# res = mod.fit()
# print('模型拟合完成')
# print(res.summary())
# a = [1, 2, 3]
# print(item <= 3 for item in a))
test_data_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73\HCP\hcp_test\*.nii'
test_label_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\程序\simple_brain_age_73\HCP\hcp_test\hcp_test.csv'
model_path = '../Epoch_253_training_state.pkl'

batch_size = 1
model_path = load_model(model_path)
loader, labels = load_data(data_path=test_data_path,
                           label_path=test_label_path, batch_size=batch_size) # type: DataLoader
loader.
print(loader[0])

