#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:19:35 2018

@author: HDK

Random Forest for Stock Selection
"""

#%% 环境准备

import os
os.chdir('D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/03 Group Assignment/Data/demo')

import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
import matplotlib.pyplot as plt


from WindPy import w
# w.start()

#%% 参数设置

# 0. 样本参数
class para_sample:
    data_path = 'D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/03 Group Assignment/Data/demo'
    result_path = 'D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/03 Group Assignment/Data/result'
    section_list_sample = range(82, 153+1)    # 样本内截面期
    section_list_test = range(154, 230+1)    # 测试截面期
    select_percent = [0.3, 0.3]    # 标记样本比例
    cv_percent = 0.1    # 交叉验证样本比例
    seed = 42    # 随机种子

para_sample = para_sample()

# 1. 全行业代码表
ind_code_df = pd.read_csv('ind_code_dict.csv',dtype=str, index_col=0)
ind_code_dict =ind_code_df.to_dict()['industry_code']

# 2. 沪深 300 指数收益表
hs300_df = pd.read_csv('hs300.csv', index_col=0)
hs300_dict =hs300_df.to_dict()['hs300_return_t_plus']

# 3. 截面日期字典
def get_month_end(any_day):    # 定义函数：返回月末日期
    month_next = any_day.replace(day=28) + dt.timedelta(days=4)    # 跳到下一月
    return month_next - dt.timedelta(days=month_next.day)    #  倒回上月最后一日

section_dict = {}
section_num = -2
for year in range (1998,2017+1):    # 获取全截面日期字典
    for month in range(1, 12+1):
        month_end = str(get_month_end(dt.date(year,month,1)))
        section_dict[section_num]=month_end
        section_num = section_num + 1


#%% 1.1 数据获取
"""
主体数据来源于万得，补充数据用 python WindPy 包下载
股票池为全A股，因子列表参见附录

# 从万得下载因子数据
section_num_list_11 = list(range(230, 237+1))    # 设置样本空间边界

for section_num in section_num_list_11:   # 下载数据
    list_stock = w.wset("SectorConstituent","date=%s" % section_dict[section_num],"sector=全部A股" ).Data[1]
    section = w.wss(list_stock, "ipo_date,sec_status,close", "tradeDate=%s" % section_dict[section_num])    # 设置要下载的因子
    data = pd.DataFrame(section.Data, index = section.Fields, columns = section.Codes).T
    data.to_csv('data_1/%s_1.csv' % section_num)
"""

#%% 1.2 样本筛选和数据清洗
"""
样本筛选：
1) 剔除未上市和上市3个月以内的样本： 通过 "上市日期（ipo_date）" 判定
2) 剔除ST样本：通过 "股票状态（sec_status）" 判定
3) 剔除特定因子组值缺省的样本
因子筛选：
1) 剔除EPcut因子
2) 剔除三个情绪因子
数据清洗：
1) 各因子的0值处理(暂时缺数据，以0代替)
2) rating_change 评级变动的缺省值用0填充
3) 剔除市值取对数和股价取对数缺省的样本
"""

section_num_list_12 = range(82, 82+1)    # 设定取样范围

for section_num in section_num_list_12:
    data_0 = pd.read_csv('data_0/%s.csv' % section_num, index_col=0)
    data_1 = pd.read_csv('data_1/%s_1.csv' % section_num, index_col=0)
    data_2 = pd.concat([data_0, data_1], axis=1, join_axes=[data_0.index])     # 合并数据

    data_2['IPO_DATE'] = pd.to_datetime(data_2['IPO_DATE'])    # str to date
    data_2 = data_2[data_2['IPO_DATE'] < section_dict[section_num-3]]    # 剔除上市三月内股票
    data_2 = data_2[data_2['SEC_STATUS'].str.contains('L')]    # 剔除ST或退市股票
    data_2 = data_2.drop(['Epcut', 'IPO_DATE', 'SEC_STATUS', 'rating_average', 'rating_change', 'rating_targetprice'], axis=1)    # 剔除因子

    # 剔除市值因子缺省样本、 str to float
    data_2 = data_2.replace('#DIV/0!', np.nan)
    data_2 = data_2.dropna()
    data_2 = data_2.astype('float64')

    # 剔除特定因子组值缺省的样本
    data_2 = data_2[~ ((data_2['ROE_G_q'] == 0) & (data_2['ROE_q'] == 0) & (data_2['ROE_ttm'] == 0))]
    data_2 = data_2[~ ((data_2['ROA_q'] == 0) & (data_2['ROA_ttm'] == 0))]
    data_2 = data_2[~ ((data_2['grossprofitmargin_q'] == 0) & (data_2['grossprofitmargin_ttm'] == 0))]
    data_2 = data_2[~ ((data_2['operationcashflowratio_q'] == 0) & (data_2['operationcashflowratio_ttm'] == 0))]

    # 输出文件
    # data_2.to_csv('data_2/%s_2.csv' % section_num)

#%% 2.1 特征预处理
"""
1) 缺失值: 因数据获取困难，不按照原文做法（设为行业均值），改为直接删除因子暴露缺失的样本
2) 去极值: MdAD 法(Median Absolute Deviation), 参数沿用研报用的 5
3) 中性化: 市值与行业中性化
4) 序列化: 排序取序法
"""

# 定义函数：MdAD法去极值:
def filter_EV_MdAD(series, n):
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    return np.clip(series,min_range,max_range)


# 定义函数：标记行业哑变量：
def industry_dummy(df):

    # 创建当截面的行业代码字典
    ind_dict = {}
    for stk in list(df.index):
        ind_dict['%s' % stk] = ind_code_dict['%s' % stk]
    # 标记行业哑变量
    identity_array = np.identity(len(ind_dict))
    df = pd.DataFrame(identity_array, index=ind_dict.keys(), columns=ind_dict.values())
    df = df.groupby(df.columns, axis=1).sum()

    return df


# 特征预处理
section_num_list_21 = range(82, 208+1)    # 设定取样范围

for section_num in section_num_list_21:
    data_2 = pd.read_csv('data_2/%s_2.csv' % section_num, index_col=0, usecols=range(54+1))
    data_3 = pd.read_csv('data_2/%s_2.csv' % section_num, index_col=0)
    data_2['ln_capital_4ols'] = data_2['ln_capital']
    factor_list = list(data_2.columns)
    factor_list.remove('DP')    # 此因子0值过多，不适宜用中位数法去极值，故排除
    factor_list.remove('ln_capital_4ols')

    # 去极值
    for column in factor_list:
        data_2[column] = filter_EV_MdAD(data_2[column], 5)

    # 中性化和序列化
    factor_list.append('DP')
    data_2_ind = industry_dummy(data_2)
    data_2_ind_LnMtkCap = pd.concat([data_2['ln_capital_4ols'], data_2_ind], axis=1, join_axes=[data_2.index])

    for column in factor_list:
        # 中性化：
        neu_model = sm.OLS(data_2[column], data_2_ind_LnMtkCap).fit()
        data_2[column] = neu_model.resid
        # 序列化：
        data_2_1 = data_2.sort_values(by=column)
        data_2_1[column] = list(range(1, len(data_2)+1))
        data_3[column] = data_2_1[column] / len(data_2)

    # 输出文件
    data_3.to_csv('data_3/%s_3.csv' % section_num)


#%% 2.2 标签提取和标记
"""
1) 以下一个自然月的个股超额收益作为样本的标签（以沪深300指数为基准），正例为1，负例为0
2) 剔除截面本月或下一个月停牌的股票：通过下一月收益是否空值判断
"""

# 定义函数：将超额收益高的标记为 1，超额收益低的标记为 0
def label(df):
    df['return_bin'] = np.nan
    df = df.sort_values(by='return_t_plus', ascending=False)    # 排序
    n_stk_select = np.multiply(para_sample.select_percent, len(df))    # 设置切片数
    n_stk_select = np.around(n_stk_select).astype(int)    # 取整
    # 标记正负例
    df.iloc[0:n_stk_select[0], -1] = 1
    df.iloc[-n_stk_select[1]:, -1] = 0
    df = df.dropna(axis=0)
    return df

section_num_list_22 = range(82, 207+1)    # 设定取样范围

for section_num in section_num_list_22:
    # 读取本月、下月数据
    df_t       = pd.read_csv('data_3/%s_3.csv' % (section_num),   index_col=0, usecols=[0,55])
    df_t_plus  = pd.read_csv('data_3/%s_3.csv' % (section_num+1), index_col=0, usecols=[0,55])
    df_t_all   = pd.read_csv('data_3/%s_3.csv' % (section_num), index_col=0, usecols=range(54+1))

    # 合并数据框
    df_return = pd.concat([df_t, df_t_plus], axis=1, join_axes=[df_t.index])
    df_return.columns = ['CLOSE_t', 'CLOSE_t_plus']

    # 计算超额收益率
    hs300_return = hs300_dict[section_dict[section_num]]    # 获取当期的沪深300下月收益率
    df_return['return_t_plus'] = (np.log(df_return['CLOSE_t_plus'] / df_return['CLOSE_t'])) - hs300_return
    data_4 = pd.concat([df_t_all, df_return], axis=1, join_axes=[df_t_all.index])

    # 标记正负例
    data_4 = label(data_4)
    data_4 = data_4.drop(['CLOSE_t', 'CLOSE_t_plus', 'return_t_plus'], axis=1)    # 删除中间变量

    # 输出文件
    data_4.to_csv('data_4/%s_4.csv' % section_num)


#%% 3.1 样本集合成

for section_num in para_sample.section_list_sample:
    # 读取文件
    file_name = para_sample.data_path + '/data_4/' + str(section_num) + '_4.csv'
    data_i = pd.read_csv(file_name)
    # 修改股票列名、新增截面期
    data_i.rename(columns={data_i.columns[0]: 'stock'}, inplace=True)
    data_i.insert(1, 'section_num', section_num)
    # 合并样本集
    if section_num == para_sample.section_list_sample[0]:    # 第一个截面期
        data_sample = data_i
    else:    # 除第一个截面期外的
        data_sample = data_sample.append(data_i)


#%% 3.2 训练集和交叉验证集切分

data_sample = pd.read_csv('data_sample_20180906.csv', index_col=0)
X_sample = data_sample.loc[:, 'EP':'BIAS']
y_sample = data_sample.loc[:, 'return_bin']
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=para_sample.cv_percent, random_state=para_sample.seed)


#%% 4.1 模型建立

# 模型参数设置
class para_model:
    n_estimators = 150          # 树的数量  默认值：10
    max_features = None         # 分叉时的最大特征数 可选值：'sqrt' 'log2' int float
    max_depth = 100             # 最大层数  默认值：None
    min_samples_split = 2       # 每个中间节点的最小样本数  默认值: 2
    min_samples_leaf = 50       # 每个叶结点的最小样本数  默认值：1
    min_impurity_decrease = 0   # 分叉判断：最小的纯度减少值  默认值：0
    oob_score = None            # Out-of-bag 训练模式（交叉验证提高泛化性） 默认值：None
    n_jobs = 4                  # 同时运行的核数
    random_state = 1            # 随机种子
para_model = para_model()

# 模型初定义
RF_model_0 = RandomForestClassifier(oob_score=True,n_jobs=-1,random_state=1)
RF_model_0.fit(X_train, y_train)

# 输出预测准确率
y_hat_train = RF_model_0.predict(X_train)    # 估计偏倚性
print('训练集准确率：%f' % (sum(y_hat_train == y_train) / len(y_train)))
y_hat_cv = RF_model_0.predict(X_test)    # 估计泛化性
print('测试集准确率：%f' % (sum(y_hat_cv == y_test) / len(y_test)))


#%% 4.2 交叉验证调参

# 图像法选择参数的较优候选值（前五个参数）, 依据：AUC

n_estimators_list = list(range(50, 1000+1, 50))
AUC_list = []

for para_value in range(50, 1000+1, 50):
    RF_model_0.set_params(n_estimators = para_value)
    RF_model_0.fit(X_train, y_train)
    y_scores = RF_model_0.fit(X_train, y_train).predict(X_test)
    y_true = y_test.values
    AUC = metrics.roc_auc_score(y_true, y_scores)    # 计算 AUC
    AUC_list.append(AUC)

plot_n_estimators = plt.scatter(n_estimators_list, AUC_list)

# 参数表设定
parameter_table = [{'n_estimators':[300,200], 'max_depth':[200,250,300], 'min_samples_leaf':[50,100]}]
#
 # 模型定义
RF_model_para = RandomForestClassifier()

# GridSearchCV 定义
GSCV = GridSearchCV(estimator=RF_model_para, param_grid=parameter_table, scoring='roc_auc', n_jobs=-1)

# 模型训练
GSCV.fit(X_sample, y_sample)

# 输出最优参数组合
result_2 = GSCV.cv_results_
best_2 = GSCV.best_params_
#%% 4.3 分阶段滚动回测


#%% 备用代码

"""
# 从万得获取行业字典

def wind_ind_dict(df, date):
    # 从万得获取行业字典
    stock_list = df.index
    ind_category = w.wsd(','.join(stock_list), "industry_swcode", "%s" % date, "%s" % date, "industryType=1")
    wind_ind_dict = {k: v for (k, v) in zip(ind_category.Codes, ind_category.Data[0])}

    # 标记行业哑变量
    identity_array = np.identity(len(wind_ind_dict))
    df_ind = pd.DataFrame(identity_array, index=wind_ind_dict.keys(), columns=wind_ind_dict.values())
    df_ind = df_ind.groupby(df_ind.columns, axis=1).sum()

    return df_ind






"""


