import numpy as np
import matplotlib.pylab as plt

def loadDataSet(fileName):
    """
    加载数据
    :param fileName: 文件名
    :return:
        xArr：x数据集
        yArr：y数据集
    """
    numFeat=len(open(fileName).readline().split('\t'))-1
    xArr=[]
    yArr=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr,yArr

def plotDataSet():
    """
    绘制数据集
    :return:
    """
    xArr,yArr=loadDataSet('ex0.txt')
    #数据个数
    n=len(xArr)
    #样本点
    xcord=[]
    ycord=[]
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #绘制样本点
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__=='__main__':
    plotDataSet()





