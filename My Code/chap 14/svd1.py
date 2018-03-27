from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1,1,1,0,0],
           [2,2,2,0,0],
           [5,5,5,0,0],
           [1,1,0,2,2],
           [0,0,0,3,3],
           [0,0,0,1,1]]

Data=loadExData()
U,Sigma,VT=la.svd(Data)


#由于Sigma是以向量的形式存储，故需要将其转为矩阵，
sig3=mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
# 也可以用下面的方法，将行向量转化为矩阵，并且将值放在对角线上，取前面三行三列
# Sig3=diag(Sigma)[:3,:3]
print(Sigma)
#重构原始矩阵
print(U[:,:3]*sig3*VT[:3,:])