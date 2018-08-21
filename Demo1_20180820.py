#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
@author Dekun He
SVM DEMO
"""

#%% 准备事项






#%% 导入库

import numpy as np
import pandas as pd
import sklearn as skl
import blpapi as bl


#%% 定义参数类

class Para:
    method = 'SVM'   # 选择机器学习模型，或 'LR' 'SGD（随机梯度下降模型）'
    month_in_sample = range(82, 153+1)   # 样本内区间为 82-153 共 72 个月末
    month_test = range(154, 230+1)    # 样本外区间为 154-230 共 77 个月末
    percent_select = [0.3, 0.3]    # 选用月超额收益排名前30%的股票作为正例， 后30%的作为反例
    percent_cv = 0.1    # 选择10%作为交叉验证集，剩余90%为训练集
    path_data = '.\\csv_demo\\'    # 数据文件路径
    path_results = '.\\results_demo\\'    # 结果文件路径
    seed = 42    # 随机种子
    svm_kernel = 'linear'    # svm 参数 （核）：核函数类型，linear 是线性核，poly是多项式核，sigmoid或者 rbf（高斯核）
    svm_c = 0.01    # svm 参数：惩罚系数 C
para = Para()    # 创建一个实例，名字为 para

#%% 数据标记
# 本模块定义一个函数：输入全部样本，选择超额收益最高和最低的部分样本，
# 分别标记为 1 和 0，再将未被标记的样本剔除，返回标记完成的样本

def label_data(data):
    data['return_bin'] = np.nan    # 初始化：新增一列名为 return_bin 的列，初始化赋值为，nan，用于随后记录label
    data = data.sort_values(by='return', ascending=False)    # 按照超额收益降序排序（nan会被排在最后）

    n_stock_select = np.multiply(para.percent_select, data.shape[0])
    n_stock_select = np.around(n_stock_select).astype(int)
    # 第一行：调用参数中的 p_s值，shape方法获取维数，[0]表示选取第一个值，故返回总行数（总股数），multiply为乘法，故结果是 [0.3 x n, 0.3 x n]，即前后股数
    # 第二行：对第一行结果进行取整，astype为格式转换方法

    data.iloc[0:n_stock_select[0], -1] = 1 定义标记1    # 切片：切出前面行作为好股，标记为1
    data.iloc[-n_stock_select[1]:, -1] = 0 定义标记0    # 切片：切出前面行作为坏股，标记为0
    data = data.dropna(axis=0)    # drop掉无标记的股票（删除整行）
    return data    # 输出经过标记的样本空间（DataFrame格式）

#%% 数据读取
# 将 csv 表格中的数据按月份顺序逐个导入内存，剔 除空值，并做标记取前后部分样本，
# 再将所有月份的数据拼接形成一个大的 DataFrame， 作为最终的样本内数据集。

for i_month in para.month_in_sample:
    # 读取数据
    file_name = para.path_data + str(i_month) + '.csv'    # 文件路径
    data_curr_month = pd.read_csv(file_name, header=0)    # 读取文件

    # 数据处理
    para.n_stock = data_curr_month.shape[0]    # 对 para 实例定义一个新的参数：股票数，将数据框的行数返回给它
    data_curr_month = data_curr_month.dropna(axis=0)     # 删除 nan 值
    data_curr_month = label_data(data_curr_month)    # 标记样本

    # 合并数据
    if i_month == para.month_in_sample[0]:    # 第一个月
        data_in_sample = data_curr_month    # 定义样本空间数据框并从第一个月开始填入数据
    else:
        data_in_sample = data_in_sample.append(data_curr_month)    # 除了第一个月以外，都从最右列后面开始填入数据














