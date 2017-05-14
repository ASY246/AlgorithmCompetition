# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:21:45 2016

kaggle上面树叶分类的竞赛，使用softmax完成
整体的数据传输思路就是用传统的readlines把数据集传输到列表中，然后列表转换为矩阵np.matrix，再传入到tensorflow做运算

@author: ASY


加个标准化试试，BN
dropout对应处理nan的情形
"""

import pandas as pd
import os
#from pandas import Series, DataFrame

trainFile = 'D:/Competition/Kaggle/LeafClassify/train.csv/train.csv'

if os.path.exists('D:/Competition/Kaggle/LeafClassify/train.csv/trainWithoutName.csv') == 0:
    train_df = pd.read_csv(trainFile)#df的默认读入数据的方法把trainFile的列名自动读入进去，如果需要自己设置可以header= None, names =等
    train_df.to_csv('D:/Competition/Kaggle/LeafClassify/train.csv/trainWithoutName.csv', index = False, header = False)

#def multilayer_perceptron(x, weights, biases):
#    layer_1 = tf.add(tf.matmul)
#这里的作用只是把行 列名去掉,有关pandas和tensorflow的对接

#下面使用原始方法读入数据。。。。

with open('D:/Competition/Kaggle/LeafClassify/train.csv/trainWithoutName.csv') as trainFile:
    rawTrain = trainFile.readlines()

print('the amount of trainSamples:', len(rawTrain))

labelSet = set()
for line in rawTrain:
    labelSet.add(line.split(',')[1])

labelSet = sorted(list(set(labelSet)))
print('the amount of labelSet', len(labelSet))

label_indices = dict((c, i) for i, c in enumerate(labelSet))
indices_char = dict((i, c) for i, c in enumerate(labelSet))

trainFeature_list = []
trainLabel_list = []

#验证集
vali_rate = 0.2
vali_Feature = []
vali_Label = []

import numpy as np

for line in rawTrain:
    if np.random.random() > 0.2:        
        lineList = []
        for index in range(len(line.split(','))):
            if index == 1:#文件中第二列为类别
                trainLabel_list.append(int(label_indices[line.split(',')[1]]))
            elif index >= 2:
                lineList.append(float(line.split(',')[index]))
                
        trainFeature_list.append(lineList)
    else:
        lineList = []
        for index in range(len(line.split(','))):
            if index == 1:#文件中第二列为类别
                vali_Label.append(int(label_indices[line.split(',')[1]]))
            elif index >= 2:
                lineList.append(float(line.split(',')[index]))
                
        vali_Feature.append(lineList)        
    
#把label转化为one-hot向量

###################################这里将列表特征转化为numpy的矩阵形式，在随后的步骤中就可以使用矩阵乘法
#trainLabel_list = np.mat(trainLabel_list).I

#训练集
trainLabel_mat = np.zeros([len(trainFeature_list),99])
trainFeature_mat = np.mat(trainFeature_list)
#验证集
vali_Label_mat = np.zeros([len(vali_Feature),99])
vali_Feature_mat = np.mat(vali_Feature)

for index in range(len(trainLabel_mat)):
    trainLabel_mat[index][trainLabel_list[index]]=1

#matrix是ndarray的子类，所以前面ndarray的优点都保留，另外matrix全部都是二维的，并且加入了一些更符合直觉的函数
#mat.I表示逆矩阵，乘法表示的是矩阵相乘的结果

#上tensorflow

import tensorflow as tf

#NN Parameters
#设置两个隐层，每层节点个数为256
n_hidden = 10

x = tf.placeholder(tf.float32,[None, 192])#向量维数为192
#字典式写参数的方法
weights = {
           'h':tf.Variable(tf.random_normal([192,n_hidden])),
#           'h2':tf.Variable(tf.random_normal([n_hidden,n_hidden])),
           'out':tf.Variable(tf.random_normal([n_hidden,99]))
}
biases = {
    'b':tf.Variable(tf.random_normal([n_hidden])),
#    'b2':tf.Variable(tf.random_normal([n_hidden])),
    'out':tf.Variable(tf.random_normal([99]))
}

layer = tf.add(tf.matmul(x,weights['h']), biases['b'])
layer = tf.nn.relu(layer)

out_layer = tf.nn.softmax(tf.matmul(layer, weights['out']) + biases['out'])

label_correct = tf.placeholder(tf.float32, [None, 99])#建立label_correct
#交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_correct * tf.log(out_layer), reduction_indices=[1]))
#softmax是凸函数使用局部梯度下降
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

x_test = tf.placeholder(tf.float32,[None, 192])#测试数据特征输入
x_vali = tf.placeholder(tf.float32,[None, 192])
#导入测试数据
with open('D:/Competition/Kaggle/LeafClassify/test.csv/test.csv') as testFile:
    rawTest = testFile.readlines()
testFeature = []

for line in rawTest:
    lineList = []
    for index in range(len(line.split(','))):
        if index >= 1:
            lineList.append(float(line.split(',')[index]))
    testFeature.append(lineList)
    
testFeature = np.mat(testFeature)

batch_size = 100

def select_batch(batch_size, allFeature, allLabel):
#实现从序列中不放回随机抽样，用于mini_batch
    allFeatureDict = dict((i,c) for i, c in enumerate(allFeature))
    allLabelDict = dict((i,c) for i, c in enumerate(allLabel))
    batchFeature = []
    batchLabel = []
    for i in range(50):#这个循环有问题，把需要的元素拿掉了
        rand_index = int(np.random.choice(len(allFeatureDict),1)[0]) #有放回抽取其中一个元素
        batchFeature.append(allFeatureDict[rand_index])
        batchLabel.append(allLabelDict[rand_index])
        del allFeatureDict[rand_index]
        del allLabelDict[rand_index]#################字典，这里面key为1，2，3去掉了一个，就需要重新排列了
        allFeatureDict = dict((i,c) for i, c in enumerate(allFeatureDict.values()))
        allLabelDict = dict((i,c) for i, c in enumerate(allLabelDict.values()))
        
    batchLabel_oneHot = np.zeros([len(batchLabel),99])
    for index in range(len(batchLabel)):
        batchLabel_oneHot[index][batchLabel[index]] = 1
        
    return batchFeature, batchLabel_oneHot

###为什么第一次出来就是nan！！？？？？
with tf.Session() as sess:
    sess.run(init)
    
    #加入minibatch机制
    for epoch in range(200):#迭代1000次   
        ave_cost = 0
#        out_test = sess.run([out_layer], feed_dict={x:trainFeature})         
        total_batch = int(len(trainFeature_list)/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            batchFeature, batchLabel = select_batch(batch_size, trainFeature_list, trainLabel_list)
            #随机取出batch_size个样本点
            _, CE_res = sess.run([optimizer,cross_entropy], feed_dict={x:np.mat(batchFeature), label_correct:np.mat(batchLabel)})#feed的数据为python的原本数据，不是tf转化的常量
        if epoch% 100 == 0:
            print("epoch=" , epoch, "CE_res=", CE_res)         
#        print("W=",sess.run(weights['h1']),'\n',"b=",sess.run(biases['b1']),'\n')
    print("Optimization Finished\n")
#    print("CE_res=",CE_res,'\n')0
    #输出训练集精度
    correct_prediction_train = tf.equal(tf.argmax(label_correct,1), tf.argmax(out_layer,1))
    
    accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))
    
    print("Accuracy_train",accuracy_train.eval({x:trainFeature_mat,label_correct:trainLabel_mat}))
    #输出验证集精度
    
    y_test_layer = tf.add(tf.matmul(x_vali,weights['h']), biases['b'])
    y_test_outlayer = tf.nn.softmax(tf.matmul(y_test_layer, weights['out']) + biases['out'])
    
    #这里面使用什么方式把tf的object拿出来,eval
    y_vali_value = y_test_outlayer.eval({x_vali:vali_Feature})
    
    correct_prediction_vali = tf.equal(tf.argmax(label_correct,1), tf.argmax(y_vali_value,1))
    accuracy_vali = tf.reduce_mean(tf.cast(correct_prediction_vali, tf.float32))
    
    print("Accuracy_validation", accuracy_vali.eval({label_correct:vali_Label_mat}))
    
    
    #对于这类多分类问题没有必要计算准确率，直接计算交叉熵就可以
#    y_test_value = y_test_outlayer.eval({x_test:testFeature})
#test_res = np.zeros([594,99])
#for index in range(len(y_test_out)):
#    test_res[index][np.argmax(y_test_out[index])] = 1
    



    

























    

