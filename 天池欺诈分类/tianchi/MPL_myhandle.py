import datetime

import numpy as np
import pandas
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from download_data import download
import matplotlib.pyplot as plt
from analysis_data import normalize_num_type
import Custom_MSE


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k,net



#获取之前处理好的数据
data = pandas.read_csv('E:\数据挖掘\Pycharm_Code\Boston_house_price\天池欺诈分类\\tianchi\\all_features.csv')



# 加载原始数据
train = pd.read_csv('E:\数据挖掘\Pycharm_Code\Boston_house_price\天池欺诈分类\\train.csv')
train = train.drop([254])  # 这个case 没又保险之前就出现?有点奇怪 直接去掉
test = pd.read_csv('E:\数据挖掘\Pycharm_Code\Boston_house_price\天池欺诈分类\\test.csv')
labels=pd.read_csv('E:\数据挖掘\Pycharm_Code\Boston_house_price\天池欺诈分类\\tianchi\\train_labels.csv')
def cross_entropy(y_hat, y):
    # y_hat = torch.sigmoid(y_hat)
    # print('62',y_hat[range(len(y_hat))])
    # print('63',y)
    # print('cross_entropy 62', y_hat[range(len(y_hat)), y])
    # print('cross_entropy:',- torch.log(y_hat[range(len(y_hat)), y]))
    return - torch.log(y_hat[range(len(y_hat)), y])

#定义损失函数
# loss = cross_entropy
loss = nn.CrossEntropyLoss()
#获取训练的数据集和对应的特征
n_train = train.shape[0]

#去掉fraud这个列

train_labels = torch.tensor(labels['fraud'].values,dtype=torch.long)#训练集和验证机的标签

# data.drop(['fraud'], axis=1)#去掉结果标签，剩下的就是训练特征


train_features = torch.tensor(data[:n_train].values,dtype=torch.float32)#训练集我用K折训练，所以里面既有训练集合也有验证集合


test_features = torch.tensor(data[n_train:].values,dtype=torch.float32)#这是测试集合


#
print("train_features shape is :",train_features.shape)
print("test_features shape is :",test_features.shape)


def get_net():
    net = nn.Sequential(
        nn.Linear(197, 50),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(50, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        )
    # net.apply(init_weights)

    return net

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
def init_xavier_normal_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight,gain=1)


def train():
    #获取网络，设置权重
    net = get_net()
    net.apply(init_xavier_normal_weights)

    k=10
    num_epochs=100
    batch_size=64
    # trainer = torch.optim.SGD(net.parameters(), lr=0.001,)
    trainer = torch.optim.Adam(net.parameters(), lr=0.001)
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_features, train_labels)

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',i,'折训练结果')

        train_iter = d2l.load_array((X_train, y_train), batch_size) #要先转成dataset的对象
        test_iter = d2l.load_array((X_valid, y_valid), batch_size)

        l_sum, accuracy, y_num =  d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    return net

def predict(net):
    # preds = net(test_features).detach().numpy()

    net.eval()
    preds = net(test_features).argmax(axis=1).detach().numpy()
    # print(preds)

    test['fraud'] = pd.Series(preds.reshape(1, -1)[0])
    # test_data['SalePrice'] = np.expm1(test_data['SalePrice'])
    submission = pd.concat([test['policy_id'], test['fraud']], axis=1)
    submission.to_csv('天池欺诈结果.csv', index=False)

def main():
    net = train()
    predict(net)

if __name__ == '__main__':
    main()
