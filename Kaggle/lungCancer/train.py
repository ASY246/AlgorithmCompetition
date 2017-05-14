# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:56:31 2017

@author: ASY

NavieCNN模型训练部分 根据tf Deep MNIST for Experts
"""

import dataProcess
import tensorflow as tf
import pandas as pd
import numpy as np


#def getTestSet():
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 标准差为0.1正态分布，超过2倍标准差的去除
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 因为使用relu激活函数，为了防止神经元为0，初始化为一个稍微小的正数
    return tf.Variable(initial)
    
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding = 'SAME')  # 步长为1，padding补0使输出与输入维度相同
    
def max_pool(x,dim):
    return tf.nn.max_pool(x, ksize=[1,dim,dim,1], strides=[1,dim,dim,1],padding='SAME')
    
def naiveCNN():
    '''
    tf上最简单的卷积神经网络
    '''
    
    x = tf.placeholder(tf.float32, shape=[None, 512*512])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    
    W_conv1 = weight_variable([32,32,1,64])   # 前两个为卷积核维度，第三个为输入数据的channel，最后一个为输出的channel，因此是对于每个5*5的卷积核，生成32个特征
    b_conv1 = bias_variable([64])
    
    x_image = tf.reshape(x, [-1, 512, 512, 1])  # 先将x处理为4维张量，最后一维为channel
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
    h_pool1 = max_pool(h_conv1,4)
    
    W_conv2 = weight_variable([32,32,64,128])  # 第二层卷积，由32个channel生成64个channel，每个5*5的卷积生成2个特征
    b_conv2 = bias_variable([128])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2,4)
    
    W_fc1 = weight_variable([32*32*128, 1024])  # 因为之前两次池化，卷积加padding不改变尺寸，因此现在图像的尺寸为除两次池化之后的结果，然后64个channel，featureMap的格式，隐藏层使用1024个节点
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*128])  # reshape里面放-1表示自动推断该维度
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    '''Dropout--reduce overfitting'''
    keep_prob = tf.placeholder(tf.float32)  # 在训练的时候打开dropout，输出测试数据的时候关闭dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    '''Readout Layer'''
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    trainSet, validationSet, testSet = dataProcess.preProcess_Mean() #在python shell中读入数据
    
#    trainFeatures, trainLabels = dataProcess.getFeaturesLabels(trainSet['features'], trainSet['cancer'])
    
    validationFeatures, validationLabels = \
    dataProcess.getFeaturesLabels(validationSet['features'],pd.concat([validationSet['label1'],validationSet['label2']],axis = 1))
    
#    trainFeatures, trainLabels = \
#    dataProcess.getFeaturesLabels(trainSet['features'],pd.concat([trainSet['label1'],trainSet['label2']],axis = 1))

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  
        for i in range(20000):
            batchFeatures, batchLabels = dataProcess.miniBatch(trainSet,5) #每次训练输入10张图片
            if i % 1 == 0:
                train_accuracy = accuracy.eval(feed_dict = {x:validationFeatures,y_:validationLabels,keep_prob:1.0})  #直接输入series看看是否有效
                print("step %d, training accuracy %g"%(i, train_accuracy))
                
            train_step.run(feed_dict={x:batchFeatures,y_:batchLabels,keep_prob:0.5}) #计算反向传播时设为0.5
            
        print("test accuracy %g"%accuracy.eval())
    

if __name__ == '__main__':
    naiveCNN()







