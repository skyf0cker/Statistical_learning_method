from EMAlgorithm import EmAlgorithm
import numpy as np


def create_data(mu0, sigma0, mu1, sigma1, alpha0, alpha1):
    '''
    初始化数据集
    这里通过服从高斯分布的随机函数来伪造数据集
    :param mu0: 高斯0的均值
    :param sigma0: 高斯0的方差
    :param mu1: 高斯1的均值
    :param sigma1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
    '''
    #定义数据集长度为1000
    length = 1000

    #初始化第一个高斯分布，生成数据，数据长度为length * alpha系数，以此来
    #满足alpha的作用
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
    #第二个高斯分布的数据
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))

    #初始化总数据集
    #两个高斯分布的数据混合后会放在该数据集中返回
    dataSet = []
    #将第一个数据集的内容添加进去
    dataSet.extend(data0)
    #添加第二个数据集的数据
    dataSet.extend(data1)

    #返回伪造好的数据集
    return dataSet


data = create_data(2, 2, 4, 2, 0.6, 0.4)
e = EmAlgorithm(data, 2)
e.train()
# a = e.compute_gama()
# e.update()