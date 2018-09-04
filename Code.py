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
from statsmodels import robust

from WindPy import w
w.start()

#%% 1.1 数据获取
"""
主体数据来源于万得，补充数据用 python WindPy 包下载
股票池为全A股，因子列表参见附录
"""

# 定义函数返回月末日期
def get_month_end(any_day):
    month_next = any_day.replace(day=28) + dt.timedelta(days=4)    # 跳到下一月
    return month_next - dt.timedelta(days=month_next.day)    #  倒回上月最后一日

# 获取全截面字典（1998-2017年每月月末为一截面）
section_dict = {}
section_num = -2

for year in range (1998,2017+1):
    for month in range(1, 12+1):
        month_end = str(get_month_end(dt.date(year,month,1)))
        section_dict[section_num]=month_end
        section_num = section_num + 1

# 从万得下载因子数据
section_range = list(range(230, 237+1))    # 设置样本空间边界
"""
for section_num in section_range:   # 下载数据
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

section_num_list = range(82, 208+1)    # 设定取样范围

for section_num in section_num_list:
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


#%% 2.1 标签提取
"""
1) 以下一个自然月的个股超额收益作为样本的标签（以沪深300指数为基准）
2) 剔除截面本月或下一个月停牌的股票：通过下一月收益是否空值判断
"""

# 定义函数：计算本月收益及下月收益
section_num_list = range(83, 207+1)    # 设定取样范围

for section_num in section_num_list:
    # 读取上月、本月、下月数据
    df_t_minus = pd.read_csv('data_2/%s_2.csv' % (section_num-1), index_col=0, usecols=[0,55])
    df_t       = pd.read_csv('data_2/%s_2.csv' % (section_num),   index_col=0, usecols=[0,55])
    df_t_plus  = pd.read_csv('data_2/%s_2.csv' % (section_num+1), index_col=0, usecols=[0,55])
    df_t_all   = pd.read_csv('data_2/%s_2.csv' % (section_num), index_col=0)

    # 合并数据框
    df_return = pd.concat([df_t_minus, df_t, df_t_plus], axis=1, join_axes=[df_t.index])
    df_return.columns = ['CLOSE_t_minus', 'CLOSE_t', 'CLOSE_t_plus']

    # 计算收益率
    df_return['return_t'] = np.log(df_return['CLOSE_t'] / df_return['CLOSE_t_minus'])
    df_return['return_t_plus'] = np.log(df_return['CLOSE_t_plus'] / df_return['CLOSE_t'])
    data_3 = pd.concat([df_t_all, df_return], axis=1, join_axes=[df_t_all.index])

    # 剔除停牌样本和中间变量
    data_3 = data_3.dropna()
    data_3 = data_3.drop(['CLOSE', 'CLOSE_t_minus', 'CLOSE_t', 'CLOSE_t_plus'], axis=1)

    # 输出文件
    # data_3.to_csv('data_3/%s_3.csv' % section_num)

#%% 2.2 特征预处理
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


# 定义函数： 依据万得行业分类，标记行业哑变量：
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


# 特征预处理
section_num_list = range(85, 85+1)    # 设定取样范围

for section_num in section_num_list:
    data_3 = pd.read_csv('data_3/%s_3.csv' % section_num, index_col=0)
    data_4 = pd.read_csv('data_3/%s_3.csv' % section_num, index_col=0)
    data_3['ln_capital_4ols'] = data_3['ln_capital']
    factor_list = list(data_3.columns)
    factor_list.remove('DP')
    factor_list.remove('ln_capital_4ols')
    factor_list.remove('return_t')
    factor_list.remove('return_t_plus')

    # 去极值
    for column in factor_list:
        data_3[column] = filter_EV_MdAD(data_3[column], 5)

    # 中性化和序列化
    factor_list.append('DP')
    data_3_ind = wind_ind_dict(data_3, date=section_dict[section_num])
    data_3_ind_LnMtkCap = pd.concat([data_3['ln_capital_4ols'], data_3_ind], axis=1, join_axes=[data_3.index])

    for column in factor_list:
        # 中性化：
        neu_model = sm.OLS(data_3[column], data_3_ind_LnMtkCap).fit()
        data_3[column] = neu_model.resid
        # 序列化：
        data_3_1 = data_3.sort_values(by=column)
        data_3_1[column] = list(range(1, len(data_3)+1))
        data_4[column] = data_3_1[column] / len(data_3)

    # 输出文件
    data_4.to_csv('data_4/%s_4.csv' % section_num)










