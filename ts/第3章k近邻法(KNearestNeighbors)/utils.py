import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GeneraterData(object):
    def __init__(self):
        pass
    def mul_cla(self, rad_seed=1, cla=2, num=50):
        """
            rad_seed 是 numpy.random.seed 如果 这个值保持一致，生成的随机数一样
            cla 是 有几类的意思，默认返回两类
            start 是随机数开始范围
            end 是随机数字结束范围，但是不包括
            返回都是整数
        """
        np.random.seed(rad_seed)
        data = {}
        for i in range(cla):
            point_x1 = np.random.randint(0 * i, 100 * i, num)
            point_x2 = np.random.randint(0 * i, 100 * i, num)
            if not data['x1']:
                data['flag'] = np.ones((1, num)) * (i + 1) 
                data['x1'] = point_x1
                data['x2'] = point_x2
            else:
                data['x1'] = np.hstack((data['x1'], point_x1))
                data['x2'] = np.hstack((data['x2'], point_x2))
        
        return data
    
    def ser_point(self, rad_seed=1, num=100, max_number=200, min_number=0):
        np.random.seed(rad_seed)
        
        point_x1 = np.random.randint(min_number, max_number, num)
        point_x2 = np.random.randint(min_number, max_number, num)

        data = [[x, y] for x, y in zip(point_x1, point_x2)]

        return data


class DrawTool(object):
    def __init__(self):
        pass


def main():
    pass

if __name__ == "__main__":
    main()
