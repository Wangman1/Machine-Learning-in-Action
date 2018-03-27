
from numpy import *
import matplotlib.pyplot as plt
# 加载数据
def loadDataSet(filename,delim = '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [map(float,line) for line in stringArr]
    return mat(dataArr)

def pca(dataMat,topN=999999):
    # 形成样本矩阵，样本中心化
    meanVals= mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    # 计算样本矩阵的协方差矩阵
    covMat = cov(meanRemoved,rowvar=0)
    #  对协方差矩阵进行特征值分解，选取最大的 p 个特征值对应的特征向量组成投影矩阵
    eigVals,eigVects =  linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topN+1):-1]
    redEigVects = eigVects[:,eigValInd]
    # 对原始样本矩阵进行投影，得到降维后的新样本矩阵
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T)+meanVals
    return lowDDataMat,reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

if __name__=='__main__':

    #加载数据
    dataMat = replaceNanWithMean()
    #去除均值
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    #计算协方差
    covMat = cov(meanRemoved, rowvar=0)

    #特征值分析
    eigVals, eigVects = linalg.eig(mat(covMat))
    print (eigVals)
