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
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

"""
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters：
    vocabList：createVocabList返回的列表
    inputSet：切分的词条列表
Returns：
    returnVec：文档向量，词集模型
Modify：
    2018-03-14

"""
def setOfWords2Vec(vocabList,inputSet):
    #创建一个其中所含元素都为0的向量
    returnVec=[0]*len(vocabList)
    #遍历每个词条
    for word in inputSet:
        #如果词条存在于词汇表中，则置1
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print("the word: %s is not in my Vocabulary!" % word)
    #返回向量文档
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
    #创建一个空的不重复列表
    vocabSet=set([])
    for document in dataSet:
        #取并集
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

if __name__=='__main__':
    postingList,classVec=loadDataSet()
    print('postingList:\n',postingList)
    myVocabList=createVocabList(postingList)
    print('myVocabList:\n',myVocabList)
    trainMat=[]
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print('trainMat:\n',trainMat)

