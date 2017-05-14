# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:12:12 2017

@author: ASY

使用时间序列方法对模型进行拟合

首先对全量数据进行拟合

为了消除周期性，得到平稳序列，可以考虑对序列使用差分的方法，即ARIMA模型，但是实际尝试的使用7
的差分后，结果仍然不平稳（平均数，方差随时间大致不变），因此这里通过预处理来做，通过时间衰减法来处理x(d) = alpha*x(d-7) + X(d)

再通过差分寻找平稳
定阶看自相关图感觉很玄，这里使用BIC最小的p和q值
"""

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.arima_model import ARIMA
import dataPreprocess as dP
import numpy as np
import math


def test():
    #'''
    #使用指数衰减方法来做
    #'''
    #ewma = test.ewm(span=7).mean()
    
    test2 = test.diff(7)
    test2.dropna(inplace=True)
    
    '''
    使用ACF图对MA部分定阶，由于自相关系数拖尾，偏自相关系数也拖尾，因此使用ARMI模型，
    由于偏自相关图从二阶开始在置信区间内，因此AR部分p = 1,
    自相关图从二阶开始控制在置信区间内，因此MA部分q = 1
    '''
    plot_acf(test2).show
    plot_pacf(test2).show
    '''
    建立ARIMA模型,BIC 500多。。。
    '''
    model = ARIMA(test2,(1,0,1)).fit()
    print(model.summary2())
    '''
    使用python的forcast方法
    传入step = 14 预测下面14个时间步，返回值tuple第一个为预测值
    '''
    res = model.forecast(steps = 7)[0]
    '''
    trainingSet和validationSet基本呈现稳定的7天周期性，后续在前面加入指数衰减，另外剔除trainingSet异常点效果会更好，对数值进行平滑处理
    指数衰减同时也是对结果做的一次平滑处理
    '''
    trainingSet, validationSet = data.getUserPay()
    '''
    先按照shop_id, datetime做聚合再count
    '''
    return res

def predictGroupByDate(param = [0,2,0]):
    '''
    第一步先预测出一个平均值，对于那些历史数据不足的情况，就是用平均预测值替代
    或者采取另一套方案，按照历史比例分配各个商家的流量
    '''
    res = []
    for i in range(7):
        model = ARIMA(trainingSet[i].astype(float),(param[0],param[1],param[2])).fit()   # 不知道为什么这个参数BIC，AIC都比较低，后续加个自己寻找参数的程序, 自动调参
        #    print(model.summary2())
        res.append(model.forecast(steps = 2)[0])
            
        eval = 0
        for i in range(len(res)):
            eval += (abs(res[i] - validationSet[i].values)/abs(res[i] + validationSet[i].values)).mean()
            
        eval /= 7  #误差为0.082，未分商店的情况下

    #for i in range(7):
    #    trainingSet[i],validationSet[i] = trainingSet[i].reset_index(),validationSet[i].reset_index()
    #    trainingSet[i],validationSet[i] = trainingSet[i].groupby(['shop_id','time_stamp']).size(), trainingSet[i].groupby(['shop_id','time_stamp']).size()
    #    
    shopTrainingSet = []
    res_forcast = []
    for i in range(7):
        shopTrainingSet_perWeekday = []
        res_forcast_perWeekday = []
    
        for j in range(2000):
            dataSet = trainingSet[i][trainingSet[i]['shop_id'] == j]
            shopTrainingSet_perWeekday.append(dataSet)
    #        model = ARIMA(dataSet.astype(float), (0,2,0)).fit()
    #        res_forcast_perWeekday.append(model.forcast(steps = 2)[0])  #这里怎么把时间放进结果里面
    #        
        shopTrainingSet.append(shopTrainingSet_perWeekday)
        res_forcast.append(res_forcast_perWeekday)
        
#def predictGroupByShop(dataSet, param = [1,2,1]):
#    '''
#    对每个shop的所有weekday，使用历史均值产生随后两周的值，保留dateindex
#    '''
#    res = []
#    for i in range(2000):
#        res_perShop = []
#        for j in range(7):
#            model = ARIMA(dataSet[i][j].astype(float),(param[0],param[1],param[2])).fit()
#            res_perShop.append(model.forecast(steps = 2)[0])
#        res.append(res_perShop)
#            
#    return res

def getEval(predictRes, validationSet):
    
    evalRes = 0
    for i in range(2000):
        
        validationSet[i] = dataSetFilled(validationSet[i], '10/18/2016', '10/31/2016')
        
        if (predictRes[i] + validationSet[i]).sum() == 0:
            underDiv = 1
        else:
            underDiv = predictRes[i] + validationSet[i]
        temp =  abs(predictRes[i] - validationSet[i])/underDiv
        
        if math.isnan(temp.mean()):
            print(predictRes[i], validationSet[i], i)
        evalRes += temp.mean()
        
    evalRes = evalRes/2000
    
    return evalRes
    
def similayShopFlow(shop_df, shopID, order = 1):
    '''
    输入一个shopid,根据shop属性表中经过预处理的特征返回与它最相似的shop对应的流量均值
    '''
    dist = [np.sqrt(np.sum(np.square(shop_df.iloc[shopID] - shop_df.iloc[i]))) for i in range(2000)]
    
    minShopID = dist.index(sorted(dist)[order]) #最小的是它自己，取第二小的元素

    return minShopID

def dataSetFilled(dataSetPerShop, start_date = '9/1/2016', end_date = '10/31/2016'):
    '''
    输入每个shop的trainingSet，对数据进行插值和补零扩展到同一时间段
    '''
    fullFillSet = pd.DataFrame(dataSetPerShop.resample('D').ffill())  # 先按天插值，抹掉中间为0的项               

#    dateLength = len(fullFillSet)  #使用插值后的长度做平均除法的分母
    dates = pd.date_range(start_date, end_date)        
    dates_df = pd.DataFrame(dates)
    dates_df[1] = 0
    dates_df = dates_df.set_index(dates_df[0])
    dates_fill = pd.DataFrame(dates_df[1])

    dataSetFill = pd.merge(dates_fill, fullFillSet, left_index = True, right_index = True, how = 'left').fillna(0)[0]

    return dataSetFill
    
    
def predictByShop(trainingSet, shopInfo):
    '''
    获取2000个shop的按周划分的数据，
    格式：shop1每个周一数据，shop1每个周二数据。。。。shop2每个周一数据
    有部分商店在trainingSet中没有数据，只在validationSet中有数据，这部分流量使用与这个shop最相似的shop的流量均值预测
    '''
    res = []
    for i in range(2000):        
        dataSetFill = dataSetFilled(trainingSet[i])        
        for j in range(14):
            '''
            如果训练数据都为0，那么另行处理（使用这个区域的平均数估计）
            如果训练数据截尾，说明这家商店不继续营业，后续流量估计都为0
            其他情况，使用平均数预测
            '''                    
            if dataSetFill.mean() == 0:
                '''
                对应训练数据集中完全没有任何历史流量记录的情况，此时使用与该shop属性向量余弦距离
                最小的shop的历史均值数据填充该shop的流量预测结果
                '''
                minIndex = 1
                while True:
                    simShop = similayShopFlow(shopInfo,i,minIndex)  # 找到最近的shopID
                    minIndex += 1
                    if trainingSet[simShop].sum() > 0:
                        break
                    
                newPredict = dataSetFilled(trainingSet[simShop])[j-45:j-15].sum()/30
                            
            elif dataSetFill[-20:-15].mean() == 0:
                newPredict = 0
            else:
                # 使用近30天均值预测---这里面如果商店开业时间比较靠后的话结果也不合理，应该用有值的部分做平均
                '''
                滚动预测，怎么滚待处理
                '''
                newPredict = dataSetFill[j-45:j-15].sum()/30
            
            dataSetFill[-(14-j)] = newPredict
            
        res.append(dataSetFill['2016-10-18':])
        
    return res
    
if __name__ == '__main__':
    args = dP.Param()
    data = dP.DataGenerator(args)

    trainingSet, validationSet = data.getUserPay()
    shopInfo = data.getShopInfo()
    predictRes = predictByShop(trainingSet, shopInfo)
    
    print(getEval(predictRes, validationSet))
    
    
'''
将周期性加入进去，所有周一，所有周二，在这个module就可以修改，然后输出结果
'''


