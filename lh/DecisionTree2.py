#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-03 15:17:08
# @Author  : Vophan Lee (vophanlee@gmail.com)
# @Link    : https://www.jianshu.com/u/3e6114e983ad

from sklearn.datasets import make_classification
import numpy as np
import math


class Decision_Tree(object):
    """
    this is a class to build the decision tree
    """

    feature_list = []
    gain_list = []
    dim_list = []
    index = 0

    def __init__(self):
        super(Decision_Tree, self).__init__()
        self.features = 5
        self.samples = 100
        self.data = make_classification(
            n_samples=self.samples, n_features=self.features, n_classes=2)
        self.empirical_entropy = self.cal_emp_entropy(self.data)

    def cal_emp_entropy(self, data):
        """
        calculate the empirical entropy
        """
        data_0 = []
        data_1 = []
        for i in enumerate(data[1]):
            if i[1] == 0:
                data_0.append(data[0][i[0]])
            else:
                data_1.append(data[0][i[0]])
        entropy = 0
        for data_ in [data_0, data_1]:
            entropy += - \
                (len(data_) / len(data[0])) * \
                 math.log2(len(data_) / len(data[0]))
        return entropy

    def div_point(self, dim_data):
        """
        decide the divided point of each feature,here we sopposed that dim_data is a continuous dataset
        dim_data: tuple
        """
        def dichotomy(dim_data):
            div_points = np.zeros((1, self.samples)).reshape(self.samples)
            for i in enumerate(dim_data):
                if i[0] == len(dim_data) - 1:
                    break
                div_points[i[0]] = (dim_data[i[0] + 1] + i[1]) / 2
            return div_points
        dim_data = list(dim_data)
        dim_data = np.array(dim_data)
        dim_data = dim_data[:, dim_data[0].argsort()]
        dim_data = tuple(dim_data)
        div_points = dichotomy(dim_data[1])
        information_gain_list = []
        for i in div_points:
            div_index = list(div_points).index(i) + 1
            front = dim_data[1][:div_index]
            behind = dim_data[1][div_index:]
            front_flag = dim_data[0][:div_index]
            behind_flag = dim_data[0][div_index:]
            front_data = (front, front_flag)
            behind_data = (behind, behind_flag)
            if len(front_data[0]) == 1 or ((front_data[1] == front_data[1][::-1]).all() and len(front_data[0]) != len(dim_data[0]) / 2):
                behind_entropy = self.cal_emp_entropy(behind_data)
                information_gain = self.empirical_entropy - \
                    (behind_entropy * (len(behind) / len(dim_data[0])))
                information_gain_list.append(information_gain)
            elif len(behind_data[0]) == 1 or ((behind_data[1] == behind_data[1][::-1]).all() and len(front_data[0]) != len(dim_data[0]) / 2):
                front_entropy = self.cal_emp_entropy(front_data)
                information_gain = self.empirical_entropy - \
                    (front_entropy * (len(front) / len(dim_data[0])))
                information_gain_list.append(information_gain)
            elif (front_data[1] == front_data[1][::-1]).all() and len(front_data[0]) == len(dim_data[0]) / 2:

                return -1, div_points[int(len(dim_data[0]) / 2 - 1)]
            else:
                front_entropy = self.cal_emp_entropy(front_data)
                behind_entropy = self.cal_emp_entropy(behind_data)
                information_gain = self.empirical_entropy - (front_entropy * (len(front) / len(
                    dim_data[0])) + behind_entropy * (len(behind) / len(dim_data[0])))
                information_gain_list.append(information_gain)
        max_information_gain = max(information_gain_list)
        return max_information_gain, div_points[information_gain_list.index(max_information_gain)]

    def compare_features(self):
        """
        here we choose a maximium information gain among all features
        """
        gain_list_tmp = []
        point_list = []
        for i in range(self.features):
            information_gain, div_point = self.div_point((self.data[1], self.data[0].transpose()[i]))
            gain_list_tmp.append(information_gain)
            point_list.append(div_point)
        com_matrix = np.array([
            gain_list_tmp,
            point_list,
            range(self.features)
        ])
        com_matrix = com_matrix[:, com_matrix[0].argsort()]
        Decision_Tree.feature_list = list(com_matrix[1])
        Decision_Tree.gain_list = list(com_matrix[0])
        Decision_Tree.dim_list = list(com_matrix[2])

    def planet_tree(self, data):
        """
        here is the process of planeting the tree
        data: without flag
        """
        feature = Decision_Tree.feature_list[Decision_Tree.index]
        dim = Decision_Tree.dim_list[Decision_Tree.index]
        Decision_Tree.index += 1
        if Decision_Tree.gain_list[Decision_Tree.feature_list.index(feature)] == -1 or Decision_Tree.index >= len(Decision_Tree.feature_list) - 1:
            return tree_node([x for x in data.transpose()[int(dim)] if x < feature],
                             [x for x in data.transpose()[int(dim)] if x > feature],
                             feature)
        else:
            return tree_node(self.planet_tree([x for x in data[0] if x < feature]),self.planet_tree([x for x in data[0] if x > feature]), feature)

class tree_node(object):
    """
    this is the node of the decision tree
    """

    def __init__(self, left, right, data):
        self.left=left
        self.right=right
        self.data=data
