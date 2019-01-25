import numpy as np
import matplotlib.pyplot as plt
import math
from stack import stack

class knn:

    def __init__(self):
        self.data = self.data_produce()
        self.tree = self.create_branch(self.data, 0)

    def data_produce(self, num=6):
        x = np.random.randint(0, 50, num)
        y = np.random.randint(0, 50, num)

        data = []
        for i in range(num):
            tmp = [x[i], y[i]]
            data.append(tmp)
        return data

    def draw_figure(self):
        x = []
        y = []
        for i in self.data:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y)
        plt.show()

    def create_branch(self, data, axia):
        tree_branch = {}
        if len(data) != 0 and len(data) != 1:
            sorted_data = self.sort(data, axia)
            if axia == 0:
                axia = 1
            else:
                axia = 0
            length = len(data)
            index = length // 2
            tree_branch['data'] = sorted_data[index]
            tree_branch['left'] = self.create_branch(sorted_data[:index], axia)
            tree_branch['right'] = self.create_branch(sorted_data[index+1:], axia)
        elif len(data) == 1:
            tree_branch['data'] = data[0]

        return tree_branch

    def search(self, point, axia):
        """
        利用kd-tree进行最近邻搜索
        思路：
        首先，我们通过切分超平面找到近似最近点，
        然后，我们去切分超平面的另一侧去看看有没有更近的点（通过到切分超平面的距离与到近似最近点的距离的比较）
        一直回溯上去
        """
        path_stack = stack()
        def find_fake_near(tree, point, axia):
            if 'left' not in tree.keys() and 'right' not in tree.keys():
                return tree
            else:
                if point[axia] < tree['data'][axia]:
                    if axia == 0:
                        axia =1
                    else:
                        axia = 0
                    path_stack.push(tree)
                    return find_fake_near(tree['left'], point, axia)

                else:
                    if axia == 0:
                        axia = 1 
                    else:
                        axia = 0
                    path_stack.push(tree)
                    return find_fake_near(tree['right'], point, axia)

        tree = self.tree
        """
        initial three points
        """
        cur_nearest_poi = find_fake_near(tree, point, 0)['data']
        target_poi = point       
        doubt_poi = path_stack.pop()
        axia = path_stack.length() % 2

        def backtrace(cur_nearest_poi, target_poi, doubt_poi):
            nonlocal axia
            d1 = cal_distance(target_poi, cur_nearest_poi)
            d2 = abs(doubt_poi['data'][axia] - target_poi[axia])
            cnp_father = doubt_poi
            if d1 > d2:
                tmp = cur_nearest_poi
                cur_nearest_poi = doubt_poi['data']
                if 'left' in cnp_father.keys() and 'right' in cnp_father.keys():
                    if cnp_father['left'] == tmp:
                        return backtrace(cur_nearest_poi, target_poi, cnp_father['right'] ) 
                    else:
                        return backtrace(cur_nearest_poi, target_poi, cnp_father['left'] )
                else:
                    if path_stack.length() != 0:
                        cnp_grand = path_stack.pop()
                        axia = path_stack.length() % 2
                        return backtrace(cur_nearest_poi, target_poi, cnp_grand)
                    else:
                        return cur_nearest_poi
            elif d1 < d2 and path_stack.length() != 0:
                doubt_poi = path_stack.pop()
                return backtrace(cur_nearest_poi, target_poi, doubt_poi)
            else:
                return cur_nearest_poi

        def cal_distance(poi1, poi2):
            return math.sqrt(pow((poi1[0] - poi2[0]), 2) + pow((poi1[1] - poi2[1]), 2))
        

        return backtrace(cur_nearest_poi, target_poi, doubt_poi)        


    def sort(self, data, axia):
        if len(data) < 2:
            return data
        else:
            pivot = data[0]
            less = [i for i in data[1:] if i[axia] <= pivot[axia]]
            greater = [i for i in data[1:] if i[axia] > pivot[axia]]

            return self.sort(less, axia) + [pivot] + self.sort(greater, axia)


if __name__ == "__main__":

    k = knn()
    Nearest = k.search([20, 5], 0)
    x = [i[0] for i in k.data if i != Nearest]
    y = [j[1] for j in k.data if j != Nearest]
    print(Nearest)
    plt.scatter(x,y)
    plt.scatter(20, 5)
    plt.scatter(Nearest[0],Nearest[1])
    plt.show()
