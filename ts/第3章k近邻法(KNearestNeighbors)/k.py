from utils import GeneraterData as gd
from collections import namedtuple
import numpy as np

class KdNode:
    def __init__(self, axis, point, left, right):
        self.axis = axis
        self.point = point
        self.left = left
        self.right = right


class KdTree(object):
    def __init__(self, data):
        """
        data is like [[1, 2], [3, 4], [5, 6]]
        """
        # 如果点是 二维的 K 值是 2
        # 如果点是 三维的 K 值是 3
        k = len(data[0])
        self.node_num = len(data)
        def create_node(axis, data_set):
            if not data_set:
                return None
            # 当前节点
            data_set.sort(key=lambda x: x[axis])
            point_pos = len(data_set) // 2
            point_media = data_set[point_pos]
            next_axis = (axis + 1) % k
            return KdNode(axis, point_media, 
                        create_node(next_axis, data_set[0:point_pos]),
                        create_node(next_axis, data_set[point_pos+1:]))
        
        self.root = create_node(0, data)
    
    def pre_order(self, ele):
        if not ele:
            return 
        print(ele.point)
        self.pre_order(ele.left)
        self.pre_order(ele.right)

def compute_dist(l1, l2):
    try:
        return np.linalg.norm(l1 - l2)
    except:
        return np.linalg.norm(np.array(l1) - np.array(l2))

Result = namedtuple("Result", "nearest_point  nearest_dist  nodes_visited")
class KNN:
    def __init__(self, kdtree, point, num=1):
        self.kdtree = kdtree
        self.point = point

    def find_nearest(self):
        k = len(self.point)
        def travel(current, target_point, max_dist):
            """
            max_dist 是半径
            """
            # nearest_point = None
            if not current:
                # 如果当前点是 None 就返回无穷大
                return Result([0]*k, float("inf"), 0)
            # print('current axis', current.point, current.axis)
            axis = current.axis
            current_point = current.point
            nodes_visited = 1
            # 选择更近的一个点
            near_point, far_point = [current.left, current.right] if target_point[
                axis] <= current_point[axis] else [current.right, current.left]
            # print('near point, fur_point', near_point.point, far_point.point)

            temp1 = travel(near_point, target_point, max_dist)
            nodes_visited += temp1.nodes_visited
            # 如果是叶子节点 temp1 返回的是 Result([0]*k, float("inf"), 0)
            # 选择是否更新
            dist = temp1.nearest_dist
            nearest_point = temp1.nearest_point
            if dist <= max_dist:
                max_dist = dist
            
            # 如果最大距离，小于到超平面的距离，则这个点另一侧的点到目标点的距离一定会比最大距离大
            if max_dist <= abs(current_point[axis] - target_point[axis]):
                return Result(nearest_point, max_dist, nodes_visited)
            # 计算当前节点到这个节点的距离
            temp_dist = compute_dist(current_point, target_point)
            # 更新最大距离
            if temp_dist < dist:
                dist = temp_dist
                max_dist = dist
                nearest_point = current_point
            # 检查另一个节点
            temp2 = travel(far_point, target_point, max_dist)
            nodes_visited += temp2.nodes_visited
            if temp2.nearest_dist < dist:        # 如果另一个子结点内存在更近距离
                nearest_point = temp2.nearest_point    # 更新最近点
                dist = temp2.nearest_dist        # 更新最近距离

            return Result(nearest_point, dist, nodes_visited)

        root = self.kdtree.root
        target_point = self.point
        return travel(root, target_point, float("inf"))

    def add_node(self, point):
        _Result = namedtuple('_Result', 'dist point')
        # 计算距离
        dist = compute_dist(self.point, point)
        # 组成点和距离的 tupple 
        r = _Result(dist, point)
        # 加入
        self.close_nodes.append(r)
        # 排序
        self.close_nodes.sort(key=lambda x : x.dist)
    
    def add_node2(self, point, num):
        """
        这个函数的目的是, 选够了点
        看新点能不能加入
        """
        _Result = namedtuple('_Result', 'dist point')
        # 计算距离
        dist = compute_dist(self.point, point)
        # 组成点和距离的 tupple
        r = _Result(dist, point)
        # 加入
        self.close_nodes.append(r)
        # 排序
        self.close_nodes.sort(key=lambda x: x.dist)

        self.close_nodes = self.close_nodes[:num]



    def find_nearest_with_num(self, num=1):
        if num > self.kdtree.node_num:
            print('要找的节点数目，大于树节点的数目')
            return
        self.close_nodes = []
        k = len(self.point)
        target_point = self.point
        def travel(current):
            if not current:
                # 如果当前点是 None 就返回无穷大
                return Result([0]*k, float("inf"), 0)
            axis = current.axis
            current_point = current.point
            # 选择更近的一个点
            near_point, far_point = [current.left, current.right] if target_point[
                axis] <= current_point[axis] else [current.right, current.left]

            travel(near_point)
            # 递归遍历回归以后
            if len(self.close_nodes) < num:
                self.add_node(current_point)
            else:
                # 检查当前节点及右边节点满不满足加入的条件
                # max_dist_point, max_dist = [self.close_nodes[num-1].point, self.close_nodes[num-1].dist]
                max_dist = self.close_nodes[num-1].dist
                if max_dist <= abs(current_point[axis] - target_point[axis]):
                    return
            
                self.add_node2(current_point, num)
                travel(far_point)

        travel(self.kdtree.root)
        return self.close_nodes          
            
def main():
    _gd = gd()
    data = _gd.ser_point(num=100)
    kd = KdTree(data)
    kd.pre_order(kd.root)
    # print('kd.root.axis', kd.root.axis)
    # point = [72, 144]
    # knn = KNN(kd, point)
    # r = knn.find_nearest()
    # print(r)
    for i in range(100):
        point = data[i]
        knn = KNN(kd, point)
        r = knn.find_nearest()
        print(r, point)
        if r.nearest_point != point:
            print(r, point)

def test():
    _gd = gd()
    data = _gd.ser_point(num=100)
    kd = KdTree(data)
    point = [141, 115]
    knn = KNN(kd, point)
    r = knn.find_nearest_with_num(num=3)
    test_a = []
    for i in range(100):
        dist = compute_dist(point, data[i])
        test_a.append(dist)
    test_a.sort()
    print(r)
    print(test_a[0:3])

if __name__ == "__main__":
    # main()
    test()


