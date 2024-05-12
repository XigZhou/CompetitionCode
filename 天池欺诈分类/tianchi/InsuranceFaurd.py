import pandas as pd

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from analysis_data import normalize_num_type

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# 数据加载
train = pd.read_csv('E:\数据挖掘\Pycharm_Code\Boston_house_price\天池欺诈分类\\train.csv')
train = train.drop([254])  # 这个case 没又保险之前就出现?有点奇怪 直接去掉
print(train.shape)


test = pd.read_csv('E:\数据挖掘\Pycharm_Code\Boston_house_price\天池欺诈分类\\test.csv')
print(test.shape)

#合并数据一起处理
# data = pd.concat(train.iloc[:,:],test.iloc[:,:])
data = pd.concat((train,test))
print(data.shape)

#绘制热力图看字段相关性分析
corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
# train_data.corr()['fraud'].sort_values # 查看变量与fraud的相关系数
plt.show()

'''
START
处理跟日期年份相关的逻辑，做一些新的字段
'''
#policy_bind_date ： 保险绑定日期
#incident_date : 出险日期
#auto_year: 汽车购买的年份
#处理这两个日期，生成新的字段，加入训练然后抛弃这两列
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
data['incident_date'] = pd.to_datetime(data['incident_date'])


print(data['policy_bind_date'].min()) #1989-12-25 00:00:00
print(data['policy_bind_date'].max()) #2015-03-08 00:00:00
print(data['incident_date'].min())#2014-12-05 00:00:00
print(data['incident_date'].max())#2015-03-29 00:00:00
print('data[auto_year].dtype=',data['auto_year'])
#获得保险绑定日期和出险日的差额
data['incident_date_policy_bind_date_diff'] = (data['incident_date'] - data['policy_bind_date']).dt.days #差额为天数
data['incident_date_auto_year_diff_year'] = (data['incident_date'].dt.year - data['auto_year'])
print(data['incident_date_auto_year_diff_year'].head())
print(data['incident_date'].head())
print(data['auto_year'].head())
print('row 54 is :',data['incident_date_policy_bind_date_diff'].head())
#看一下成为客户时长跟出险差值，欺诈情况
dum1 = pd.concat([train['customer_months'], train['fraud']], axis=1)
dum1.plot.scatter(x='fraud', y='customer_months', alpha=0.3, ylim=(0,1000));
# plt.show()



#看一下成为客户时长跟出险差值，欺诈情况,没啥料到
train['incident_date'] = pd.to_datetime(train['incident_date'])
train['policy_bind_date'] = pd.to_datetime(train['policy_bind_date'])
train['incident_date_policy_bind_date_diff'] = (train['incident_date'] - train['policy_bind_date']).dt.days #差额为天数
print(train['incident_date_policy_bind_date_diff'].head())
print(train['fraud'].head())
dum1 = pd.concat([train['incident_date_policy_bind_date_diff'], train['fraud']], axis=1)
dum1.plot.scatter(x='fraud', y='incident_date_policy_bind_date_diff', alpha=0.3, ylim=(0,10000));
# plt.show()

#看看诈骗的人incident_date_policy_bind_date_diff的分布情况

print(train[['incident_date_policy_bind_date_diff','fraud','policy_id']].where(train['fraud'] ==1))
# train.to_csv('train1.csv')
# train[['incident_date_policy_bind_date_diff','fraud','policy_id']].where(train['fraud'] ==1).to_csv('train2.csv')

'''
END  处理跟日期年份相关的逻辑，做一些新的字段
'''



'''
START
有些变量需要分箱像
age年龄，
policy_deductable 保险扣除额 只有500 1000 2000
auto_year 买车年份，比较离散又很大不利于收敛，统一处理成 = 年份 - 其中最小年份，在分箱，三年还是五年随便试试了
'''

#年纪分箱
# data.loc[data['age'].between(19, 25, 'both'), 'age_box'] = 'A'
# data.loc[data['age'].between(25, 30, 'right'), 'age_box'] = 'B'
# data.loc[data['age'].between(30, 35, 'right'), 'age_box'] = 'C'
# data.loc[data['age'].between(35, 40, 'right'), 'age_box'] = 'D'
# data.loc[data['age'].between(40, 45, 'right'), 'age_box'] = 'E'
# data.loc[data['age'].between(45, 50, 'right'), 'age_box'] = 'F'
# data.loc[data['age'].between(50, 55, 'right'), 'age_box'] = 'H'
# data.loc[data['age'].between(55, 60, 'right'), 'age_box'] = 'I'
# data.loc[data['age'].between(60, 65, 'right'), 'age_box'] = 'J'

#保险扣除额 分箱
data.loc[data['policy_deductable'] == 500, 'policy_deductable_box'] = 'A'
data.loc[data['policy_deductable'] == 1000, 'policy_deductable_box'] = 'B'
data.loc[data['policy_deductable'] == 2000, 'policy_deductable_box'] = 'C'

#买车年份-最小
print('(data[policy_bind_date].min()).dt.year')
print(data['policy_bind_date'].min().year)
data['auto_year'] = data['auto_year'] - (data['policy_bind_date'].min()).year
#买车分箱,代表对车年纪做一个分箱
data.loc[data['auto_year'].between(0, 5, 'both'), 'auto_year_box'] = 'A'
data.loc[data['auto_year'].between(5, 10, 'right'), 'auto_year_box'] = 'B'
data.loc[data['auto_year'].between(10, 15, 'right'), 'auto_year_box'] = 'C'
data.loc[data['auto_year'].between(15, 20, 'right'), 'auto_year_box'] = 'D'

#umbrella_limit保险责任上线做个分箱，这个字段区分度还是很高的
data.loc[data['umbrella_limit'] == -1000000, 'umbrella_limit_box'] = 'A'
data.loc[data['umbrella_limit'] == 0, 'umbrella_limit_box'] = 'B'
data.loc[data['umbrella_limit'] == 2000000, 'umbrella_limit_box'] = 'C'
data.loc[data['umbrella_limit'] == 3000000,  'umbrella_limit_box'] = 'D'
data.loc[data['umbrella_limit'] == 4000000, 'umbrella_limit_box'] = 'E'
data.loc[data['umbrella_limit'] == 5000000, 'umbrella_limit_box'] = 'F'
data.loc[data['umbrella_limit'] == 6000000, 'umbrella_limit_box'] = 'G'
data.loc[data['umbrella_limit'] == 7000000,  'umbrella_limit_box'] = 'H'
data.loc[data['umbrella_limit'] == 8000000, 'umbrella_limit_box'] = 'I'
data.loc[data['umbrella_limit'] == 9000000, 'umbrella_limit_box'] = 'J'
data.loc[data['umbrella_limit'] == 10000000, 'umbrella_limit_box'] = 'K'



'''
END
有些变量需要分箱像
age年龄，
policy_deductable 保险扣除额 只有500 1000 2000
auto_year 买车年份，比较离散又很大不利于收敛，统一处理成 = 年份 - 其中最小年份，在分箱，三年还是五年随便试试了
'''

'''
START
数值型，先不做标准化看看效果------->结果数值好大，才一层网络爆炸了
'''


#开始做归一化处理
# 把数值正态化
def logs(data, log_features):
    m = data.shape[1]
    for l in log_features:
        print('feature hading is :',l)
        if l == 'capital-loss' :
            data[l] = -1*data[l] #损失是复数，log的时候取绝对值

        data = data.assign(newcol=pd.Series(np.log(1.01+data[l])).values) #----------->这里的数值不好太多小数了还是要看看之前的数据
        data.columns.values[m] = l + '_log'
        m += 1
    return data

log_features = ['customer_months','policy_annual_premium','insured_zip','capital-gains','capital-loss','incident_hour_of_the_day',
                 'number_of_vehicles_involved','property_claim','vehicle_claim','total_claim_amount',
                 'incident_date_policy_bind_date_diff']
#total_claim_amount 整体索赔跟vehicle_claim汽车索赔相关性太强，尝试去掉
all_features = logs(data, log_features)


'''
END
数值型，先不做标准化看看效果
'''


'''
Start  
One-hot编码
'''
#umbrella_limit,保险责任上限 太稀疏了先去掉看看
# data.drop(['umbrella_limit'], axis=1, inplace=True)
# all_features['umbrella_limit'].astype(str) #弄成one-hot看看看
all_features.drop(['umbrella_limit'], axis=1, inplace=True)

# one-hot encoding
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape)


'''
End  
One-hot编码
'''

'''
Start  
去掉两个日期,案件编号，还有上面二次处理的变量
'''
all_features.drop(['policy_bind_date'], axis=1, inplace=True)
all_features.drop(['incident_date'], axis=1, inplace=True)
all_features.drop(['policy_id'], axis=1, inplace=True)#案件标号去除就好

for l in log_features:
    all_features.drop([l], axis=1, inplace=True) #去掉上面二次处理的变量



#去掉分箱变量
all_features.drop(['age'], axis=1, inplace=True)
all_features.drop(['policy_deductable'], axis=1, inplace=True)
all_features.drop(['auto_year'], axis=1, inplace=True)#案件标号去除就好

train_labels = data['fraud']
train_labels.to_csv('train_labels.csv')
all_features.drop(['fraud'], axis=1, inplace=True)

'''去掉相关性太强的特征尝试看看'''
all_features.drop(['injury_claim'], axis=1, inplace=True)

'''
End  
去掉两个日期
'''

print("最终模型数据为：",all_features.shape)

all_features.to_csv('all_features.csv')
# plt.show()