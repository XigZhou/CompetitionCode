import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from download_data import download
import matplotlib.pyplot as plt
from analysis_data import normalize_num_type
import Custom_MSE

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# train_data.to_csv("train_2024");
# test_data.to_csv("test_2024");


print(train_data.shape)
print(test_data.shape)


# 去掉 训练和测试数据第一列
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

# 丢掉缺失值很多的列

all_features = all_features.drop('PoolQC',axis=1)
all_features = all_features.drop('Fence',axis=1)
all_features = all_features.drop('MiscFeature',axis=1)
all_features = all_features.drop('Alley',axis=1)
all_features = all_features.drop('FireplaceQu',axis=1)
all_features = all_features.drop('LotFrontage',axis=1)

all_features = all_features.drop('GarageCond',axis=1)
all_features = all_features.drop('GarageType',axis=1)
all_features = all_features.drop('GarageYrBlt',axis=1)
all_features = all_features.drop('GarageFinish',axis=1)
all_features = all_features.drop('GarageQual',axis=1)

all_features = all_features.drop('BsmtExposure',axis=1)
all_features = all_features.drop('BsmtFinType2',axis=1)
all_features = all_features.drop('BsmtFinType1',axis=1)
all_features = all_features.drop('BsmtCond',axis=1)
all_features = all_features.drop('BsmtQual',axis=1)

all_features = all_features.drop('MasVnrArea',axis=1)
all_features = all_features.drop('MasVnrType',axis=1)

# all_features = all_features.drop(all_features.loc[all_features['Electrical'].isnull()].index)

# all_features.to_csv('all_feature.csv')
#去掉离群点很远的数据


# Some of the non-numeric predictors are stored as numbers; convert them into strings
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)

# 把数值正态化
normalize_num_type(all_features,0.5)


# one-hot encoding
all_features = pd.get_dummies(all_features,dummy_na=True)
print(train_data.shape)

all_features.to_csv('all_feature.csv')

n_train = train_data.shape[0]
n_train

train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
#拿出训练数据化成tensor对象
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float32)
#拿出训练数据化成tensor对象
# train_data['SalePrice']=np.log1p(train_data['SalePrice'])

train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
#把训练数据结果标签编程列向量

print(train_features.shape)
print(test_features.shape)

loss = nn.MSELoss()
in_features = train_features.shape[1]

print('>>>>>train_features='+str(in_features))

# 损失的计算，按照以前y-Y的计算，房价越高损失越大并不好，所以要计算相对误差
##net(fetures)----->输出的是，预测结果
import numpy as nm


# def log_rmse(net, features, labels):
#     clipped_preds = torch.clamp(net(features), 1, float('inf'))  # 预测出来的值如果因为梯度问题而变成无穷大或者无穷小
#     # 那就用torch.clamp函数把张量里面小于min的弄成等于min大于max的弄成等于max
#     # 上面这个式子的作用把小于1块钱的弄成1块钱，大的保持原样
#     rmse = torch.sqrt(loss(torch.log(net(clipped_preds)), torch.log(labels)))
#     #     nm.savetxt('C:\\Users\\50588\\Desktop\\all_77.csv', clipped_preds.detach().numpy(), fmt = '%d', delimiter = ',')
#
#     return rmse.item()

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





def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    print('train_iter type is : ',type(train_iter))
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls

def pred(net):
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # test_data['SalePrice'] = np.expm1(test_data['SalePrice'])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission22.csv', index=False)


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')

    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # test_data['SalePrice'] = np.expm1(test_data['SalePrice'])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission22.csv', index=False)

    print('0>>>w的L2范数：', net[0].weight.norm().item())
    # print('1>>>w的L2范数：', net[3].weight.norm().item())
    # print('2>>>w的L2范数：', net[6].weight.norm().item())
    # print('2>>>w的L2范数：', net[9].weight.norm().item())
    # print('3>>>w的L2范数：', net[4].weight)
    plt.show()

def get_net():
    net = nn.Sequential(nn.Linear(in_features,120),
                        nn.ReLU(),
                        # nn.Dropout(0.1),
                        #

                        # #
                        nn.Linear(120, 80),
                        nn.ReLU(),
                        # nn.Dropout(0.1),
                        # #
                        # nn.Linear(400,200),
                        # nn.ReLU(),
                        # nn.Dropout(0.1),
                        #
                        # nn.Linear(200, 100),
                        # nn.ReLU(),
                        # nn.Dropout(0.05),

                        nn.Linear(80, 20),
                        nn.ReLU(),
                        # nn.Dropout(0.05),

                        nn.Linear(20, 1),
                        )
    # net.apply(init_weights)

    return net
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight,gain=1)
print('Start>>>>>>>>>>>>>',datetime.datetime.now())
# train_and_pred(train_features, test_features, train_labels, test_data,
#                2000, 10, 10, 1)
# train_and_pred(train_features, test_features, train_labels, test_data,
#                1500, 0.01,10, 64)
print('End>>>>>>>>>>>>>',datetime.datetime.now())

# def train_and_pred(train_features, test_features, train_labels, test_data,
#                    num_epochs, lr, weight_decay, batch_size):

k, num_epochs, lr, weight_decay, batch_size = 10, 100, 0.1, 0, 64
train_l, valid_l ,net= k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

pred(net)

plt.show()