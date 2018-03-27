from numpy import *
import matplotlib.pyplot as plt
# 加载数据
def loadDataSet(filename,delim = '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [map(float,line) for line in stringArr]
    print(mean(mat(dataArr)))
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


if __name__=='__main__':
    dataMat = mat(loadtxt('testSet.txt'))
    lowMat,reconMat = pca(dataMat,1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()

