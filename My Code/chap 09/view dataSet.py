#
# import matplotlib.pyplot as plt
# import numpy as np
#
# def loadDataSet(fileName):
#     """
#     函数说明:加载数据
#     Parameters:
#         fileName - 文件名
#     Returns:
#         dataMat - 数据矩阵
#
#     """
#     dataMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         fltLine = list(map(float, curLine))                    #转化为float类型
#         dataMat.append(fltLine)
#     return dataMat
#
# def plotDataSet(filename):
#     """
#     函数说明:绘制数据集
#     Parameters:
#         filename - 文件名
#     Returns:
#         无
#
#     """
#     dataMat = loadDataSet(filename)                                        #加载数据集
#     n = len(dataMat)                                                    #数据个数
#     xcord = []; ycord = []                                                #样本点
#     for i in range(n):
#         xcord.append(dataMat[i][0]); ycord.append(dataMat[i][1])        #样本点
#     fig = plt.figure()
#     ax = fig.add_subplot(111)                                            #添加subplot
#     ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)                #绘制样本点
#     plt.title('DataSet')                                                #绘制title
#     plt.xlabel('X')
#     plt.show()
#
# if __name__ == '__main__':
#     filename = 'ex00.txt'
#     plotDataSet(filename)


import MySQLdb
