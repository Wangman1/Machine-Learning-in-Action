import numpy as np
import operator

def createDataSet():
    """
    创建数据集
    :return:
    """
    group=np.array([[1,101],[5,89],[108,0],[108,5]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels

def classify0(inX,dataSet,labels,k):
    """
    简单文本分类
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(1)
    distance=sqDistances**0.5

    sortedDistIndices=distance.argsort()
    classCount={}

    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__=='__main__':
    group,labels=createDataSet()
    test=[1,20]
    test_class=classify0(test,group,labels,2)
    print(test_class)



