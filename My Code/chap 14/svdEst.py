from numpy import *
from numpy import linalg as la

def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 利用不同的方法计算相似度
def eulidSim(inA, inB):
    return 1.0/(1.0+la.norm(inA - inB))#根据欧式距离计算相似度

def pearsSim(inA, inB):
    if len(inA)<3:
        return 1.0
    else:
        return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
    num   = float(inA.T*inB)            #向量inA和向量inB点乘,得cos分子
    denom = la.norm(inA)*la.norm(inB)   #向量inA,inB各自范式相乘，得cos分母
    return 0.5+0.5*(num/denom)          #从-1到+1归一化到[0,1]


def standEst(dataMat,user,simMeas,item):
    """
    计算在给定相似度计算方法的前提下，用户对物品的估计评分值
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似性度量方法
    :param item: 物品编号
    :return:
    """
    #数据中行为用于，列为物品，n即为物品数目
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    #用户的第j个物品
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:
            continue
        #寻找两个用户都评级的物品
        overLap=nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]

        if len(overLap)==0:
            similarity=0
        else:
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])

        simTotal+=similarity
        ratSimTotal+=simTotal*userRating

    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    """
    推荐引擎，会调用standEst()函数
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param N:前N个未评级物品预测评分值
    :param simMeas:
    :param estMethod:
    :return:
    """
    #寻找未评级的物品，nonzeros()[1]返回参数的某些为0的列的编号，dataMat中用户对某个商品的评价为0的列
    #矩阵名.A：将矩阵转化为array数组类型
    #nonzeros(a)：返回数组a中不为0的元素的下标
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:
        return 'you rated everything!'
    itemScores=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda  jj:jj[1],reverse=True)[:N]

#======================基于SVD的评分估计==========================
def SVDEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #化为对角阵，或者用linalg.diag()函数可破
    xformedItems = dataMat.T*U[:,:4]*Sig4.I#构造转换后的物品
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j, :].T)
        print("the %d and %d similarity is: %f" %(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal ==0 :
        return 0
    else:
        return ratSimTotal/simTotal

myMat = mat(loadExData2())
print(recommend(myMat, 1, estMethod = SVDEst))
print(recommend(myMat, 1, estMethod = SVDEst, simMeas = pearsSim))
