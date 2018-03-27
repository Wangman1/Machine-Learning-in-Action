import matplotlib.pylab as plt
import numpy as np
import random
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

"""
函数说明：绘制数据集

"""
def plotBestFit(weights):
    #加载数据集
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    #数据个数
    n=np.shape(dataMat)[0]
    #正样本
    xcord1=[]
    ycord1=[]
    #负样本
    xcord2=[]
    ycord2=[]
    #根据数据集标签进行分类
    for i in range(n):
        #1为正样本
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        #0为负样本
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    x=np.arange(-3.0,3.0,0.1)

    y=(-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x,y)

    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

"""
函数说明：改进的随机梯度上升算法

"""
def stocGradAscent1(dataMatix,classLabels,numIter=500):
    m,n=np.shape(dataMatix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            #降低alpha的大小
            alpha=4/(1.0+j+i)+0.01
            #随机选取样本
            randIndex=int(random.uniform(0,len(dataIndex)))
            #选择随机选取的样本，计算h
            h=sigmoid(sum(dataMatix[randIndex]*weights))
            #计算误差
            error=classLabels[randIndex]-h
            #更新回归系数
            weights=weights+alpha*error*dataMatix[randIndex]
            #删除已经使用的样本
            del(dataIndex[randIndex])
    return weights

if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    weights=gradAscent(dataMat,labelMat)
    plotBestFit(weights)

