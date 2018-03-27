import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC


def img2Vector(filename):
    """
    将32*32的二进制图像转换为1*1024的向量
    :param filename: 文件名
    :return: 返回的二进制图像的1*1024向量
    """
    # 创建1*1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1*1024向量
    return returnVect


def handwritingClassTest():
    """
    手写数字分类测试
    :return: 无
    """
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('E:/python/machine learning in action/My Code/chap 06/trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwlabels中
        hwLabels.append(classNumber)
        # 将每个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2Vector('E:/python/machine learning in action/My Code/chap 06/trainingDigits/%s' % (fileNameStr))

    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测技术
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2Vector(
            'E:/python/machine learning in action/My Code/chap 06/testDigits/%s' % (fileNameStr))
        # 获得预测结果
        classfierResult = clf.predict(vectorUnderTest)
        print("分类返回结果为 %d \t 真实结果为%d " % (classfierResult, classNumber))

        if (classfierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据 \n 错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
