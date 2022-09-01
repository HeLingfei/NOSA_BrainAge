import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from sklearn.model_selection import KFold

class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))


# a = torch.randn(1, 1, 32, 32)
# a = torch.tensor([1.,-2.,3.,4.], requires_grad=True)
# b = F.relu(a)
# b.retain_grad()
# c = b ** 3 - 5*b # type:torch.Tensor
# # a2 = np.array([[1,2,3], [1,1,2]])
# # a3 = np.array([[3,4], [1,1], [2,2]])
# d = torch.sum(c)
# d.backward()
#
# print(a.grad)
# print(b.grad)
# model = TestModule()
# b = model(a)
#
# w = torch.tensor([[[
#     [2., 2.],
#     [2., 2.]
# ]]])
# x = torch.tensor([[[
#     [1., 2., 3.],
#     [4., 5., 6.],
#     [7., 8., 9.]
# ]]], requires_grad=True)
# w = nn.Parameter(w)
# out = F.conv2d(x, w, stride=1, padding=0)
# out2 = torch.sum(out)
# out2.backward()
# print(x.grad.data)
# # conv = nn.Conv2d(1,1)
# # print(a.requires_grad)
# X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]])
# y = np.array(range(1, 11))
a = [1, 2, 3]
a.append(4)
print(np.mean(a))
# kf = KFold(n_splits=10, shuffle=True)
# for train_index, validate_index in kf.split(x):
#     print(train_index)
#     print(validate_index)