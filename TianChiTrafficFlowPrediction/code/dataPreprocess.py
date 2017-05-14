# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:06:27 2017

@author: ASY

天池大数据竞赛 流量预测  整体分数据预处理，模型部分，评估部分三块

这里为IO及预处理部分

baseline方法仅使用十月份的数据进行预测
"""

import pandas as pd
import time, datetime
import dateutil
import os
from sklearn.preprocessing import OneHotEncoder


class Param():
    
    datafiles = r'D:\Competition\TianChiTrafficFlowPrediction'
    dateTruncate = '2016-09-01'
    dataColNames = ['user_id', 'shop_id', 'time_stamp']
    
    
class DataGenerator():
    '''
    用于读取并生成各类数据
    '''    
    def __init__(self, args):
        
        datafiles = args.datafiles
        dateTruncate = args.dateTruncate
        dataColNames = args.dataColNames
        
        
        self.raw_shop_info = pd.read_csv(datafiles + '\dataset\dataset\shop_info.txt', names = ['shop_id','city_name','location_id','per_pay',\
                                                                                   'score','comment_cnt','shop_level','cate_1_name',\
                                                                                   'cate_2_name','cate_3_name'])
        print('count of Shops is', len(self.raw_shop_info))
        if os.path.exists(datafiles + r'\dataset\dataset\shopInfo.csv') == False:
            shopInfo = self.shopFeature(self.raw_shop_info)
            shopInfo.to_csv(datafiles + r'\dataset\dataset\shopInfo.csv',  index = False)
        
        if(os.path.exists(datafiles + r'\dataset\dataset\userView_truncated.csv')):
            self.raw_user_view = pd.read_csv(datafiles + r'\dataset\dataset\userView_truncated.csv', names = dataColNames)
            print('count of userView is', len(self.raw_user_view))
#            self.divideByShopID(datafiles, self.raw_user_view)
       
            self.raw_user_pay = pd.read_csv(datafiles + r'\dataset\dataset\userPay_truncated.csv', names = dataColNames)
            print('count of userPay is', len(self.raw_user_pay))
            
            self.dataDiv = self.divideByShopID(datafiles, self.raw_user_pay)
        
        else:
            self.dateTruc(dateTruncate, dataColNames, datafiles + r'\dataset\dataset\user_view.txt',datafiles + r'\dataset\dataset\userView_truncated.csv')
            self.dateTruc(dateTruncate, dataColNames, datafiles + r'\dataset\dataset\user_pay.txt',datafiles + r'\dataset\dataset\userPay_truncated.csv')
            
    def dateTruc(self, dateTruncate, dataColNames, inputStr, outputStr):
        '''
        所有日期的数据不需要全部使用，dateTruncate为训练数据时间的起点
        '''
        raw_df = pd.read_csv(inputStr, names = dataColNames)
        user_df = raw_df[raw_df['time_stamp'] > dateTruncate]
        user_df.loc[:,'time_stamp'] = user_df.loc[:,'time_stamp'].apply(self.timeStamp2date) 
#        user_df = user_df.assign(lag = user_df.loc[:,'time_stamp'].apply(self.timeStamp2date))
        user_df.to_csv(outputStr, index=False, header=False)        
        
    def shopFeature(self, shopInfo):
        '''
        将商店属性信息整理成向量的形式，为后面计算相似度使用
        '''
        shopInfo['location_id'] = shopInfo['location_id'].astype(str)
        shopInfo = pd.get_dummies(shopInfo)
        shopInfo = shopInfo.fillna(0)
        # 先不归一化处理试试
        shopInfo['per_pay'] = (shopInfo['per_pay'] - shopInfo['per_pay'].min())/(shopInfo['per_pay'].max() - shopInfo['per_pay'].min())
        return shopInfo
        
    def divideByShopID(self, datafiles, dataSet):
        '''
        将原始数据按照shopid分成2000个文件
        '''
        dataSetDivByShop = []
        for i in range(2000):
            dataDiv = dataSet[dataSet['shop_id'] == i+1]
            dataDiv.to_csv(datafiles + '\dataset\\' + str(i+1) + '.csv', index = False, header = False)           
            dataSetDivByShop.append(dataDiv)
            
        return dataSetDivByShop
            
    def timeStamp2date(self, timeString):
        '''
        将带时间的日期截断为年月日的日期标准格式
        '''
        timeList = time.strptime(timeString, "%Y-%m-%d %H:%M:%S")
        y,m,d = timeList[0:3]
        
        return datetime.datetime(y,m,d)       
        
    def str2Date(self, dataSet):
        '''
        dataframe从磁盘读入为string形式，处理为时间序列
        '''
        dataSet['time_stamp'] = dataSet['time_stamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        # dataSet['time_stamp'] = dataSet['time_stamp'].apply(dateutil.parser.parse)
        dataSet = dataSet.set_index('time_stamp')
#        groupData = dataSet.groupby(dataSet.index).size()
        
        return dataSet
        
    def getGroupCountByShop(self, dataSet):
        '''
        按时间和shopid双索引，其余与getGroupCountByDate相同
        '''
#        dataSet['time_stamp'] = dataSet['time_stamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        # dataSet['time_stamp'] = dataSet['time_stamp'].apply(dateutil.parser.parse)
#        dataSet = dataSet.set_index('time_stamp')
        groupData = dataSet.groupby([dataSet['shop_id'], dataSet['time_stamp']]).size()
        
        return groupData
        
    def validationDiv(self, dataSet):
        '''
        取最后两周数据做validationSet，之前的数据做trainingSet，
        这里说明下原因是倒数第三周的数据跟后两周的数据趋势比较像。。。不一定会合理但是可以试
        
        输入数据为getGroupCount处理后的dataSet
        '''
        validationSet = dataSet['2016-10-18':]
        trainingSet = dataSet[:'2016-10-17']
        
        return trainingSet, validationSet        
        
    def getShopInfo(self):
        '''
        将shop属性处理为向量形式，用于后续计算相似度
        '''
        shopInfo = pd.read_csv(r'D:\Competition\TianChiTrafficFlowPrediction\dataset\dataset\shopInfo.csv')
        return shopInfo
        
    def getUserView(self):
        '''
        用户浏览数据暂时没有使用
        '''
        return self.raw_user_view
        
    def getUserPay(self):
        '''
        返回数据,先按date聚合划分trainingSet和validationSet，在加shop聚合用于生成预测数据
        '''
        user_pay = self.dataDiv
        extraData_train, extraData_vali = [],[]

        for i in range(2000):
            user_pay2Date = self.str2Date(user_pay[i]).reset_index().groupby('time_stamp').size()
            trainingSet,validationSet = self.validationDiv(user_pay2Date)
    
            extraData_train.append(trainingSet)
            extraData_vali.append(validationSet)
        
        #最后对shop进行聚合
#        trainingSet, validationSet = self.getGroupCountByShop(trainingSet), self.getGroupCountByShop(validationSet)
#        
        return extraData_train, extraData_vali        

#        
        
if __name__ == '__main__':
    
    args = Param()
    data = DataGenerator(args)
#    extraData_train, extraData_vali = data.getUserPay()
    
#    dataAnaly = DataAnalysis(data)
#    test = dataAnaly.historyFlowCount()







