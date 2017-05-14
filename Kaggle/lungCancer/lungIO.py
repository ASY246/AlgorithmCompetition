# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 21:14:21 2017

@author: ASY

kaggle 肺癌预测竞赛，使用tf和卷积图像处理的实践
"""
import os
import os.path
import pandas as pd
import dicom  #用来解析dcm文件
import numpy as np
#import pylab

#数据IO

def load_labels(file = 'D:\Competition\Kaggle\lungCancer\stage1_labels.csv\stage1_labels.csv'):
    #导入标签，一共1398条数据,training set和validation set一共1398
    labels = pd.read_csv(file, index_col = 'id')  #pandas读取数据，将id这一列作为索引，后续可以按照字典的形式调用特定userID的label
    
    return labels

def load_sampleData(path = 'D:\Competition\Kaggle\lungCancer\sample_images'):
    #导入样例数据，这里需要解析DICOM格式文件
    lstFilesDCM = []
    #统计所有.dcm文件
    for dirName, subdirList, fileList in os.walk(path):   
        '''os.walk返回一个三元tupple(dirpath,dirnames,filenames) 第一个为起始路径，为一个string，第二个为起始路径下
        所有的文件夹list，第三个为起始路径下的非目录文件的名字，这些名字不包含路径信息，加路径要使用os.path.join(dirpath,name)'''
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName, filename))
                
    # Get ref file
    rawData = {}
    for fileName in lstFilesDCM:
        RefDs = dicom.read_file(fileName)
        userID = RefDs.PatientID  # 原始数据，一个文件夹对应一个病人，一个病人有很多图像
        rawPixelArray = np.mat(RefDs.pixel_array)  #获取512*512未经归一化和白话等处理的图片向量
        if userID not in rawData: #使用in和not in判断速度比has_key快
            rawData[userID] = [rawPixelArray]
        else:
            rawData[userID].append(rawPixelArray)
        
        # 画图
        '''pylab.imshow(RefDs.pixel_array, cmap = pylab.cm.bone)
        pylab.show()'''
        
    return rawData
    # Load dimensions based on the number of rows, columns, and slices(along the z axis)    
#    ConstPixelDims = (int(RefDs, Rows), int(RefDs, Columns), len(lstFilesDCM))
#    sampleData = pd.read_csv(file)

def load_AllData(path = 'D:\Competition\Kaggle\lungCancer\stage1'):
    #导入样例数据，这里需要解析DICOM格式文件
    lstFilesDCM = []
    #统计所有.dcm文件
    for dirName, subdirList, fileList in os.walk(path):   
        '''os.walk返回一个三元tupple(dirpath,dirnames,filenames) 第一个为起始路径，为一个string，第二个为起始路径下
        所有的文件夹list，第三个为起始路径下的非目录文件的名字，这些名字不包含路径信息，加路径要使用os.path.join(dirpath,name)'''
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName, filename))
                
    # Get ref file
    rawData = {}
    for fileName in lstFilesDCM[:200]:
        RefDs = dicom.read_file(fileName)
        userID = RefDs.PatientID  # 原始数据，一个文件夹对应一个病人，一个病人有很多图像

        rawPixelArray = np.mat(RefDs.pixel_array)  #获取512*512未经归一化和白话等处理的图片向量,并转化为矩阵格式
        if userID not in rawData: #使用in和not in判断速度比has_key快
            rawData[userID] = [rawPixelArray]
        else:
            rawData[userID].append(rawPixelArray)
        
        # 画图
        '''pylab.imshow(RefDs.pixel_array, cmap = pylab.cm.bone)
        pylab.show()'''
        
    return rawData