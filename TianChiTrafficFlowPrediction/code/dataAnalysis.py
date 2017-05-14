# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:03:55 2017

天池大数据，流量预测，数据分析部分

@author: ASY
"""

import pandas as pd

class DataAnalysis():
    '''
    数据统计及分析
    '''
    def __init__(self, data):
        self.userView = data.getUserView()
        self.userPay = data.getUserPay()
    
    def historyFlowCount(self):
        '''
        历史流量统计,从10月9日开始三周成周期分布，且每周流量基本保持一致
        '''
        hisUserPay = self.userPay.groupby('time_stamp').size()
        hisUserPay.index = pd.to_datetime(hisUserPay.index)
        hisUserPay.resample('D',fill_method='ffill') # 按天插值
        hisUserPay.plot(kind = 'bar', color = 'g', alpha = 0.3)
        return hisUserPay
        
    def windowSlide(self, dataSet):
        '''
        为了消除周期性的影响，将原数据中每个时间点的流量转换为该时间点的流量和它前面六个
        时间点流量的加和,通过分析表明，通过一次处理后效果仍然不好，结果并不平稳，该方案暂时不使用
        方差还可以
        '''
        res = pd.rolling_sum(dataSet, 7)
        
        return res

#    def inverseWindowSlide(self, dataSet):
    
    def weeklyExtract(self, dataSet):
        '''
        把每周相同的天的流量抽取出来单独进行预测,输入dataSet是getGroupCount处理后的数据
        '''
        extractRes = []
        for i in range(7):
            extractRes.append(dataSet[dataSet.index.weekday == i])
            
        return extractRes
        
    def timeStamp2days(self,timeString, dateTruncate = '2016-10-01'):
        '''
        将原始timeStamp格式字符串时间转换成天的计算，这里从第一天，dateTruncate算作0
        '''
        timeString = str(timeString)
        timeList = time.strptime(timeString, "%Y-%m-%d %H:%M:%S")
        
        #这里加个正则表达式判断字符串形式
        y,m,d = timeList[0:3]
        thisDate =  datetime.datetime(y,m,d) 
        
        startTime = time.strptime(dateTruncate, "%Y-%m-%d")
        startDate = datetime.datetime(startTime[0],startTime[1],startTime[2])
        
        return (thisDate - startDate).days

    def sampleValidationDiv(self, dataSet):
        '''
        输入一个dataframe,将原数据集通过按照一定比率采样的方法，分成trainingSet和
        validationSet,因为是预测问题所以这里不使用sample的方式，方法作废
        '''
        validationSet = dataSet.sample(frac = 0.1)
        tmp = pd.merge(dataSet, validationSet, left_index = True, right_index = True, how = 'left')
        tmp = tmp.fillna('t')
        tmp = tmp[tmp.iloc[:, -1] == 't']
        tmp = tmp.iloc[:,:3]
        tmp.columns = ['user_id','shop_id','time_stamp']
        trainingSet = tmp
        
        return validationSet, trainingSet