import numpy as np
import operator

"""
函数说明：KNN算法，文本分类

Parameters：
    inX：用于分类的数据（测试集）
    dataSet：用于训练的数据（训练集）
    labels：分类标签
    k：kNN算法参数，选择距离最小的k个点
returns：
    分类结果
modify:
    2018-03-08
"""
def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 0], [108, 5]])
    labels = ['爱情片','爱情片', '动作片', '动作片']
    return group, labels

def classify0(inX,dataSet,labels,k):
    #numpy函数shape[0]返回dataSet行数
    dataSetSize=dataSet.shape[0]
    #在列向量方向上重复inX一次（横向），在行向量方向上重复inX共dataSetSize次（纵向）
    #numpy.tile([0,0],(1,1))#在列方向上重复[0,0]1次，行1次  >>>array([[0, 0]])
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    #二维特征相减后平方
    sqDiffMat=diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances=sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances=sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    #x.argsort():将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y
    sortedDistIndices=distances.argsort()
    #定一个记录类别次数的字典
    classCount={}
    #range(3):生成[0,1,2]
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel=labels[sortedDistIndices[i]]
        # 计算类别次数
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 只需迭代的取前k个样本点的labels(即标签)，并计算该标签出现的次数，
        # 这里还用到了dict.get(key, default=None)函数，key就是dict中的键voteIlabel，
        # 如果不存在则返回一个0并存入dict，如果存在则读取当前值并+1；
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #key=operator.items(1)根据字典的值进行排序
    #key=operator.items(0)根据字典的键值进行排序
    #reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)