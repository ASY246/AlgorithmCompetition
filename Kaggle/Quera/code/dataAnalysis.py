# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:31:32 2017

@author: ASY

kaggle Quera问题判断重复竞赛

baseline:每个词的one-hot向量加和形式，每个句子的向量长度等于vocabulary长度
"""

import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import Tf


trainFile = r'D:\Competition\Kaggle\Quera\data\train.csv\train.csv'
testFile = r'D:\Competition\Kaggle\Quera\data\test.csv\test.csv'

train_Ori = pd.read_csv(trainFile, index_col = 'id')
trainSet = pd.DataFrame()


def splitWords(sentence):
    '''
    使用nltk对原数据进行分词处理
    '''
    try:
        words = nltk.word_tokenize(sentence)
    except:
        words = []

    return words
    
trainSet['q1'] = train_Ori['question1'].apply(lambda x: splitWords(x))
trainSet['q2'] = train_Ori['question2'].apply(lambda x: splitWords(x))
trainSet['label'] = train_Ori['is_duplicate']

vocabularySet = set()

for questionList in (trainSet['q1'] + trainSet['q2']):
    for word in questionList:
        vocabularySet.add(word)

vocabularyList = list(vocabularySet)
index2wordDict = {index: word for index, word in enumerate(vocabularyList)}
word2indexDict = {word: index for index, word in enumerate(vocabularyList)}
outOfVocabulary = len(index2wordDict) #词典外的词标记

trainFeature = trainSet['q1'].apply(lambda question:[word2indexDict[word] for word in question]) + trainSet['q2'].apply(lambda question:[word2indexDict[word] for word in question])

def feature2Index(NumericFeature):
    '''
    将一行数字转化为one-hot编码加和的格式
    '''
    vector = np.zeros([139401])
    for numeric in NumericFeature:
        vector[numeric] += 1
    return vector

trainVector = trainFeature.head(10000).apply(lambda x: feature2Index(x))  # 这个太难存了，所以果然tf-idf转换是需要稀疏存储的。






