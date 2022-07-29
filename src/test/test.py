import os.path
from tqdm import tqdm
import numpy as np
import torch
from src.model.model import *
from src.process.data_load import data_set
import pickle
import nibabel as nib
from torch.nn.functional import one_hot
from einops import *
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment


# model  = Model(1,1)
#
# torch.save(model.state_dict(), os.path.join('..', 'checkpoints', 'model.pth'))
# x = torch.ones((4,2,2))
# #
# # y = int(x.add(torch.tensor(0.2)))
#
# # print(y)
#
# # print(torch.round(x))
# print(x * torch.Tensor([8,4,2,1]).reshape((4,1,1)))
# y = x * torch.Tensor([8,4,2,1]).reshape((4,1,1))
# print(torch.sum(y, dim=0))


def data_set_test():
    print('begin')
    data = data_set()
    shape = []
    path = os.path.join('..', 'checkpoints', 'test_data', 'shape')
    if os.path.exists(path):
        with open(path, 'r') as f:
            shape = pickle.load(f)
    else:
        for d, _ in tqdm(data):
            shape.append(d.shape[0] / d.shape[2])
        with open(path, 'wb') as f:
            pickle.dump(shape, f)
    print(np.mean(shape))
    print(np.var(shape))
    print(np.max(shape))
    print(np.min(shape))
    print()


def data_set_test2():
    print('begin')
    data = data_set()
    num = []
    path = os.path.join('..', 'checkpoints', 'test_data', 'shape2')
    if os.path.exists(path):
        with open(path, 'r') as f:
            num = pickle.load(f)
    else:
        for d, _ in tqdm(data):
            # num.append(np.unique(d))
            print(np.unique(d))
        with open(path, 'wb') as f:
            pickle.dump(num, f)

    print(np.unique(num))


def data_set_test3():
    print('begin')
    data = data_set()
    size = []
    path = os.path.join('..', 'checkpoints', 'test_data', 'shape3')
    if os.path.exists(path):
        with open(path, 'r') as f:
            size = pickle.load(f)
    else:
        for d, _ in tqdm(data):
            size.append(d.shape)
        with open(path, 'wb') as f:
            pickle.dump(size, f)

    print(np.unique(size))


def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    print((2. * intersection + smooth) / \
          (output.sum() + target.sum() + smooth))
    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def dice_loss(target, predictive, ep=1e-8):
    intersection = 2. * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = intersection / union
    return loss


def acc(model):
    m = model
    test_set = data_set(False)
    task1_test_data = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    acc = []
    with torch.no_grad():
        for i, j in tqdm(task1_test_data):
            i = i.cpu().float()
            j = j.cpu().numpy()
            j = torch.LongTensor(j)
            sep_class = 0
            # for index in range(16):
            #     if index != sep_class:
            #         j[j == index] = 0
            j = one_hot(j, 16)
            j = rearrange(j, 'b w h d c -> b c w h d')
            z = m(i)
            y = torch.argmax(z, dim=1)
            # for index in range(16):
            #     if index != sep_class:
            #         z[z == index] = 0
            acc.append(dice_coef(z, j))
        print(np.mean(acc))


import PIL

from PIL import Image

import SimpleITK as sitk

if __name__ == '__main__':
    m = Model(1, 16)
    m.load_state_dict(torch.load(os.path.join(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot.pth'))))
    m.cpu()
    acc(m)
