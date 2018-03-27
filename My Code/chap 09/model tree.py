from numpy import *
import matplotlib.pylab as plt

def loadDataSet(fileName):
	"""
	函数说明:加载数据
	Parameters:
	    fileName - 文件名
	Returns:
		dataMat - 数据矩阵
	Website:
	    http://www.cuijiahua.com/
	Modify:
	    2017-12-09
	"""
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float, curLine))					#转化为float类型
		dataMat.append(fltLine)
	return dataMat

#根据ws计算公式，求出子树对应的ws矩阵
#helper function used in two places
def linearSolve(dataSet):
    m,n = shape(dataSet)
    #注意，x矩阵第一列全部为1,
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

#create linear model and return coeficients
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws
#误差计算
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
	"""
	函数说明:树构建函数
	Parameters:
	    dataSet - 数据集合
	    leafType - 建立叶结点的函数
	    errType - 误差计算函数
	    ops - 包含树构建所有其他参数的元组
	Returns:
		retTree - 构建的回归树
	Website:
	    http://www.cuijiahua.com/
	Modify:
	    2017-12-12
	"""
	#选择最佳切分特征和特征值
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	#r如果没有特征,则返回特征值
	if feat == None: return val
	#回归树
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	#分成左数据集和右数据集
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	#创建左子树和右子树
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree


if __name__=='__main__':
    myDat = loadDataSet('exp2.txt')
    myMat2 = mat(myDat)
    myTree = createTree(myMat2, modelLeaf, modelErr)
    print(myTree)