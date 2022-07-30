import os

import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import plot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.process.data_load import *
from src.model.model import *
from loss import *
from einops import *
from tqdm import tqdm
from torch.nn.functional import one_hot


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(pre_train_model, batch_size, criterion, device):
    model = pre_train_model.to(device)
    # 指定损失函数，可以是其他损失函数，根据训练要求决定
    # criterion = nn.CrossEntropyLoss()  # 交叉熵
    # 指定优化器，可以是其他
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 初始化 early_stopping 对象
    patience = 10  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True, path=os.path.join('..', 'checkpoints', 'auto_save',
                                                                             'model_onehot2.pth'))  # 关于 EarlyStopping 的代码可先看博客后面的内容

    # batch_size = 64  # 或其他，该参数属于超参，对于如何选择超参，你可以参考下我的上一篇博客
    n_epochs = 100  # 可以设置大一些，毕竟你是希望通过 early stopping 来结束模型训练

    # 建立训练数据的 DataLoader
    train_set = data_set()
    # 把dataset放到DataLoader中
    data_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

    test_data = data_set(False)
    # 把dataset放到DataLoader中
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

    # ----------------------------------------------------------------
    # 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
    train_loss = []
    valid_loss = []
    for epoch in range(1, n_epochs + 1):
        # ---------------------------------------------------
        model.train()  # 设置模型为训练模式

        # training
        parb = tqdm(data_loader)
        t_loss = []
        model = model.to(device)
        for batch, (data, y) in enumerate(parb):
            optimizer.zero_grad()
            # trans y to onehot
            y = torch.LongTensor(y.long())
            data, y = data.float().to(device), y.to(device)
            y = one_hot(y, 16)
            target = rearrange(y, 'b d w h c -> b c d w h')
            # training param
            output = model(data)  # 输出模型预测值
            loss = criterion(output, target.float())  # 计算损失
            loss.backward()  # 计算损失对于各个参数的梯度
            optimizer.step()  # 执行单步优化操作：更新参数
            parb.set_description('epoch:{}, loss:{}'.format(epoch, loss.item()))
            t_loss.append(loss.item())
        # ----------------------------------------------------
        train_loss.append(np.mean(t_loss))
        model.eval()  # 设置模型为评估/测试模式
        v_loss = []
        for test, y in tqdm(test_loader):
            model = model.cpu()
            y = torch.LongTensor(y.long())
            y = one_hot(y, 16)
            target = rearrange(y, 'b d w h c -> b c d w h')
            PRED = model(test.float())
            loss = criterion(PRED, target.float())
            v_loss.append(loss.item())
        valid_loss.append(np.mean(v_loss))
        early_stopping(np.mean(v_loss), model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
    len_train = len(train_loss)
    plt.plot([i for i in range(len_train)], train_loss, '-', label='train_loss')
    plt.plot([i for i in range(len_train)], valid_loss, '-', label='valid_loss')
    plt.savefig('loss line.png', bbox_inches='tight')
    # 获得 early stopping 时的模型参数
    return model


if __name__ == '__main__':
    model = UnetModel(1, 16, 6)
    model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot2.pth')))
    model = train(model, 1, DiceLoss(), torch.device('cuda'))
