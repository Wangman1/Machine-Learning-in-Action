import numpy as np
from functools import reduce
"""
函数说明：创建实验样本

Parameters：
    无
Returns：
    postingList：实验样本切分的词条
    classVec：类别标签向量
Modify：
    2018-03-14

"""


def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


"""
函数说明：更加vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters：
    vocabList：createVocabList返回的列表
    inputSet：切分的词条列表
Returns：
    returnVec：文档向量，词集模型
Modify：
    2018-03-14

"""


def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        # 如果词条存在于词汇表中，则置1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    # 返回向量文档
    return returnVec


""""
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters：
    dataSet：整理的样本数据集
Returns：
    vocabSet：返回不重复的词条列表，也就是词汇表
Modify：
    2018-03-14

"""


def createVocabList(dataSet):
    # 创建一个空的不重复列表
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


"""
函数说明：朴素贝叶斯分类器训练函数

Parameters：
    trainMatrix：训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory：训练类别标签向量，即loadDataSet返回的classVec
Returns：
    p0Vect：侮辱类的条件概率数组
    p1Vect：非侮辱类的条件概率数组
    pAbusive：文档属于侮辱类的概率
Modify：
    2018-03-14

"""


def trainNB0(trainMtrix, trainCategory):
    # 计算训练的文档数目
    numTrainDocs = len(trainMtrix)
    # 计算每篇文章的词条数
    numWords = len(trainMtrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建numpy.ones数组，词条初始化为1，拉普拉斯平滑
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)
    # 分母初始化为2.0，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:
            p1Num += trainMtrix[i]
            p1Denom += sum(trainMtrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMtrix[i]
            p0Denom += sum(trainMtrix[i])
    # 相除
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p1Denom)
    # 返回属于侮辱类的条件概率
    return p0Vect, p1Vect, pAbusive

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classifyaaa：待分类的词条数组
    p0Vec：侮辱类的条件概率数组
    p1Vec：非侮辱类的条件概率数组
    pClass1：文档属于侮辱类的概率
Returns:
    0 ：属于非侮辱类
    1 ：属于侮辱类
Modify:
    2018-03-14
"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #对应元素相乘,logA*B=logA+logB，所以要加上np.log(pClass1)
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)

    if p1>p0:
        return 1
    else:
        return 0

"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
Modify:
    2018-03-14
"""
def testingNB():
    #创建实验样本
    listOPosts,listClasses=loadDataSet()
    #创建词汇表
    myVocabList=createVocabList((listOPosts))

    trainMat=[]
    for postinDoc in listOPosts:
        #将实验样本向量化
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    #训练朴素贝叶斯分类器
    p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
    #测试样本1
    testEntry=['love','my','dalmation']
    #测试样本向量化
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    #执行分类并打印分类结果
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    # 执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')
    #测试样本2
    testEntry=['stupid','garbage']

    #测试样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    # 执行分类并打印分类结果
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
        # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')

if __name__ == '__main__':
    testingNB()





