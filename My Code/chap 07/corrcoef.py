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

def standRegres(xArr,yArr):
    """
    计算回归系数w
    :param xArr: x数据集
    :param yArr: y数据集
    :return: w：回归系数
    """
    #np.mat 将序列转化为二维数组
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    #np.linalg.inv()：矩阵求逆
    #np.linalg.det()：矩阵求行列式（标量）
    #如果行列式为0，则为奇异矩阵，不能求逆
    if np.linalg.det(xTx)==0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    #回归系数
    # .I为求逆
    ws=xTx.I*(xMat.T*yMat)
    return ws

if __name__=='__main__':
    #加载数据集
    xArr,yArr=loadDataSet('ex0.txt')
    #计算回归系数
    ws=standRegres(xArr,yArr)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    yHat=xMat*ws
    #np.corrcoef 获得相关系数矩阵
    print(np.corrcoef(yHat.T,yMat))


