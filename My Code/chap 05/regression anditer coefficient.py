import matplotlib.pylab as plt
import numpy as np
import random
from matplotlib.font_manager import FontProperties

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
    alpha=0.01
    #最大迭代次数
    maxCycles=500
    weights=np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        #梯度上升矢量化公式
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    #将矩阵转化为数组，并返回权重参数
    return weights.getA(),weights_array

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
def stocGradAscent1(dataMatix,classLabels,numIter=150):
    m,n=np.shape(dataMatix)
    weights=np.ones(n)
    weights_array=np.array([])
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
            #添加回归系数到数组中
            weights_array=np.append(weights_array,weights,axis=0)
            #删除已经使用的样本
            del(dataIndex[randIndex])
    #改变维度
    weights_array=weights_array.reshape(numIter*m,n)
    #返回
    return weights,weights_array

"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-30
"""
def plotWeights(weights_array1,weights_array2):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1,weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2,weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)

