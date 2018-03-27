from numpy import *
from numpy import linalg as la

def printMat(inMat,thresh=0.8):
    """
    打印矩阵
    :param inMat:输入数据集
    :param thresh:阈值
    :return:
    """
    #由于矩阵包含了浮点数，因此必须定义深色和浅色，通过阈值来界定
    #元素大于阈值，打印1，小于阈值，打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1),
            else:
                print(0),
        print('')

def imgCompress(numSV=3,thresh=0.8):
    """
    实现图像的压缩，允许基于任意给定的奇异值数目来重构图像
    :param numSV:
    :param thresh:
    :return:
    """
    #构建列表
    myl=[]
    #打开文本文件，从文件中以数值方式读入字符
    for line in open("0_5.txt").readlines():
        newRow=[]
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat=mat(myl)

    print("******original matrix**********")
    #使用SVD对图像进行SVD分解和重构
    printMat(myMat,thresh)
    U,Sigma,VT=la.svd(myMat)
    #建立一个全0矩阵
    SigRecon=mat(zeros((numSV,numSV)))
    #将奇异值填充到对角线
    for k in range(numSV):
        SigRecon[k,k]=Sigma[k]
    #重构矩阵
    reconMat=U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("***reconstructed matrix using %d singular values***" %numSV)
    printMat(reconMat,thresh)

if __name__=='__main__':
    imgCompress(2)
