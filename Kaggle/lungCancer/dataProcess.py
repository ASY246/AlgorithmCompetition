# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:47:07 2017

@author: ASY

肺癌预测，数据预处理
"""

import lungIO
import numpy as np
import pandas as pd

'''
后续加入归一化处理，白化,图像增强等处理
validationset
'''
def oneHotEncode(dataFrame, columns):
    '''
    use dummies to encode,输入为对应列的dataframe和修改后的one-hot编码列名
    '''
    resdf = pd.get_dummies(dataFrame)
    resdf.columns = columns
    
    return resdf
    
def preProcess_Mean():
    '''
    平均化每个人的图片信息的方法做预处理
    '''
    preProcessFeature = {}
    rawDict = lungIO.load_sampleData() #使用全量数据时修改函数
    labels = lungIO.load_labels()
    
    for userID in rawDict.keys():
        pictMean = sum(rawDict[userID])/len(rawDict[userID])
        pictMeanFlat = np.array(pictMean.flatten()).flatten().reshape(512*512)
        preProcessFeature[userID] = pictMeanFlat

    featureData,labelData = preProcessFeature,labels  # 导入全量数据,feature
    features = pd.DataFrame(pd.Series(featureData),columns = ['features'])
#    frames = [features, labelData]  #两块数据index均为userID，做join操作
    mergeFrame = pd.merge(features, labelData, left_index = True, right_index = True, how = 'left').fillna('Nan')  # 以索引为键值做左连接
    # 将train和test拼接然后加上index然后随机取点
#    testSet = concateFrame.loc[concateFrame['cancer'] == 'Nan']

    '''split data to train set ,validation set and test set'''
    testSet = mergeFrame.loc[mergeFrame['cancer'] == 'Nan']

    dataSetWithLabel = mergeFrame.loc[mergeFrame['cancer'] != 'Nan'] #全部有标签数据
    trainSet = dataSetWithLabel.sample(frac = 0.5, replace = False)  #划分trainSet和validation，小数据按4：1
    validationSet = dataSetWithLabel[pd.isnull((dataSetWithLabel - trainSet)['cancer'])]  # 空值的为validation
                                     
    trainSetOneHotLabel = oneHotEncode(trainSet['cancer'], ['label1','label2'])
    validationSetOneHotLabel = oneHotEncode(validationSet['cancer'],['label1','label2'])
 
    
    trainSetOneHotLabel['features'] = trainSet['features']
    validationSetOneHotLabel['features'] = validationSet['features']


    
#    validationSet = dataSetWithLabel[(~dataSetWithLabel['features'].isin(trainSet['features']))&(~dataSetWithLabel['cancer'].isin(trainSet[features]))]
    
    
    return trainSetOneHotLabel,validationSetOneHotLabel,testSet
    
def getFeaturesLabels(featureSet,labelSet):
    '''
    从上个函数的dataframe格式的featureSet中分离出可以送入tensor的numpy array类型的features
    '''
    featuresArray = np.array(list(featureSet.values), dtype=np.float)
    labelsArray = np.array(list(labelSet.values), dtype=int).reshape(featureSet.shape[0],-1)  # get_dummies这种方式存在问题，如果原数据只有一种，那么变化后不会成为二维
    
    return featuresArray, labelsArray
    

def miniBatch(trainSet,batchSize = 10):
    '''
    从全量数据中,选取batchSize大小的数据
    '''    
    '''train_features = np.array(list(trainSet['features'].values), dtype=np.float)  #类型转换
    validation_features = np.array(list(validationSet['features'].values), dtype=np.float)
    test_features = np.array(list(testSet['features'].values), dtype=np.float)
    
    train_labels = np.array(list(pd.get_dummies(trainSet['cancer']).values), dtype=int).reshape(train_features.shape[0],-1)
    validation_labels = np.array(list(pd.get_dummies(validationSet['cancer']).values), dtype=int).reshape(train_features.shape[0],-1)'''

    
#    batchRandom = np.random.randint(0, batchSize, size = batchSize)
#    batch = trainSet.reset_index()
#    batch.columns=['userID','features','label'] #reindex(range(len(trainSet))).take(batchRandom)
#    batch = batch.take(batchRandom)
    batch_train = trainSet.sample(n = batchSize,replace=False)  #dataframe的sample方法，用来有放回和无放回随机抽样
#    batch_features = np.array(list(batch_train['features'].values), dtype=np.float)  #类型转换
#    batch_labels = np.array(list(pd.get_dummies(batch_train['cancer']).values), dtype=int).reshape(10,-1)
    batch_features, batch_labels = getFeaturesLabels(batch_train['features'],pd.concat([batch_train['label1'],batch_train['label2']],axis=1))
    
    return batch_features, batch_labels
    
if __name__ == '__main__':
    preProcess_Mean()