#!/usr/bin/env python
# coding=utf-8
from numpy import *
from numpy import linalg as la
def loadExData():
    return [[1,1,1,0,0],
           [2,2,2,0,0],
           [5,5,5,0,0],
           [1,1,0,2,2],
           [0,0,0,3,3],
           [0,0,0,1,1]]

# 利用不同的方法计算相似度
def eulidSim(inA, inB):
    return 1.0/(1.0+la.norm(inA - inB))#根据欧式距离计算相似度

def pearsSim(inA, inB):
    if len(inA)<3:
        return 1.0
    else:
        return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
    num   = float(inA.T*inB)            #向量inA和向量inB点乘,得cos分子
    denom = la.norm(inA)*la.norm(inB)   #向量inA,inB各自范式相乘，得cos分母
    return 0.5+0.5*(num/denom)          #从-1到+1归一化到[0,1]

myMat = mat(loadExData())
print(eulidSim(myMat[:,0], myMat[:,4])) #第一行和第五行利用欧式距离计算相似度
print(eulidSim(myMat[:,0], myMat[:,0])) #第一行和第一行欧式距离计算相似度
print(cosSim(myMat[:,0], myMat[:,4]))   #第一行和第五行利用cos距离计算相似度
print(cosSim(myMat[:,0], myMat[:,0]))   #第一行和第一行利用cos距离计算相似度
print(pearsSim(myMat[:,0], myMat[:,4])) #第一行和第五行利用皮尔逊距离计算相似度
print(pearsSim(myMat[:,0], myMat[:,0])) #第一行和第一行利用皮尔逊距离计算相似度

print(myMat)