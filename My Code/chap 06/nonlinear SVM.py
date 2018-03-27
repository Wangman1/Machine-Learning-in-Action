import matplotlib.pylab as plt
import numpy as np

def loadDataSet(fileName):
    """

    :param fileName:
    :return: dataMat, labelMat
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return dataMat, labelMat


def showDataSet(dataMat,labelMat):
    """
    数据可视化

    :param dataMat: 数据矩阵
    :param labelMat: 数据标签
    :return: 无
    """
    #正样本
    data_plus=[]
    #负样本
    data_minus=[]

    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus)
    data_minus_np=np.array(data_minus)
    #正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1])
    #负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1])

    plt.show()

if __name__=='__main__':
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    showDataSet(dataArr,labelArr)


