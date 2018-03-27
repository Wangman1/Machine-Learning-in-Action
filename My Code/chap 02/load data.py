from numpy import *
import operator  # 导入运算符模块

# 创建数据集和标签的函数
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


if __name__ == '__main__':
    group, labels = createDataSet()

    print(group)
    print(labels)
