import matplotlib.pylab as plt
import numpy as np

"""
函数说明：加载数据

"""
def loadDataSet():
    # 创建数据列表
    dataMat=[]
    # 创建标签列表
    labelMat=[]
    # 打开文件
    fr=open('testSet.txt')
    # 逐行读取
    for line in fr.readlines():
        # 去回车，放入列表
        lineArr=line.strip().split()
        # 添加数据
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        # 添加标签
        labelMat.append(int(lineArr[2]))
    # 关闭文件
    fr.close()
    return dataMat,labelMat

"""
函数说明：sigmoid函数

"""
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


"""
函数说明：梯度上升算法

"""
def gradAscent(dataMatIn,classLabels):
    #转化成numpy的mat
    dataMatrix=np.mat(dataMatIn)
    # 转化成numpy的mat，并进行转置
    labelMat=np.mat(classLabels).transpose()
    #返回dataMatrix的大小，m为行，n为列
    m,n=np.shape(dataMatrix)
    #移动步长，也就是学习速率，控制参数更新的速度
    alpha=0.001
    #最大迭代次数
    maxCycles=500
    weights=np.ones((n,1))

    for k in range(maxCycles):
        #梯度上升矢量化公式
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    #将矩阵转化为数组，并返回权重参数
    return weights.getA()

if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    print(gradAscent(dataMat,labelMat))

