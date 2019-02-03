#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-02 14:19:55
# @Author  : Vophan Lee (vophanlee@gmail.com)
# @Link    : https://www.jianshu.com/u/3e6114e983ad

from sklearn.datasets import make_classification
import numpy as np
import math


class decision_tree(object):
    """
    one step to access my dream
    """

    def __init__(self):
        super(decision_tree, self).__init__()
        self.features = 5
        self.samples = 100
        self.data = make_classification(n_samples=self.samples, n_features=self.features, n_classes=2)
        self.data_0 = []
        self.data_1 = []
        for i in enumerate(self.data[1]):
            if i[1] == 0:
                self.data_0.append(self.data[0][i[0]])
            else:
                self.data_1.append(self.data[0][i[0]])
        self.means_1 = np.mean(self.data_1, axis=0, keepdims=True).reshape(self.features)
        self.std_1 = np.std(self.data_1, axis=0, keepdims=True, ddof=1).reshape(self.features)
        self.means_0 = np.mean(self.data_0, axis=0, keepdims=True).reshape(self.features)
        self.std_0 = np.std(self.data_0, axis=0, keepdims=True, ddof=1).reshape(self.features)
        self.std = [self.std_0, self.std_1]
        self.means = [self.means_0, self.means_1]
        self.empirical_entropy = self.cal_emp_entropy()

    def cal_emp_entropy(self):
        all_p = []
        data_list = [self.data_0, self.data_1]
        for i in range(2):
            p = 1
            for dim in range(self.features):
                for data in data_list[i]:
                    # print(self.std[data_list.index(all_data)][dim])
                    p *= (1 / (pow(2 * math.pi, 0.5) * self.std[i][dim])) * pow(math.e, -(pow((data[dim] - self.means[i][dim]), 2) / 2 * pow(self.std[i][dim], 2)))
            all_p.append(p)

        entropy = 0
        for p in all_p:
            entropy += -p * math.log2(p)
        return entropy
