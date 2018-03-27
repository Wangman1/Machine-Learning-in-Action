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
函数说明：绘制数据集
"""
def plotDataSet():
    #加载数据集
    dataMat,labelMat=loadDataSet()
    #转化成numpy的array
    dataArr=np.array(dataMat)
    #数据个数
    n=np.shape(dataMat)[0]
    #正样本
    xcord1=[];ycord1=[]
    #负样本
    xcord2=[];ycord2=[]
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
    fig=plt.figure()
    #添加subplot
    ax=fig.add_subplot(111)
    #绘制正样本
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=.5)
    #绘制负样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)

    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__=='__main__':
    plotDataSet()




