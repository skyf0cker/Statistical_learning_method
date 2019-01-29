

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
        cur_nearest_poi = find_fake_near(tree, point, 0)
        target_poi = point       
        doubt_poi = path_stack.pop()
        axia = path_stack.length() % 2

        def backtrace(cur_nearest_poi, target_poi, doubt_poi):
            nonlocal axia
            d1 = cal_distance(target_poi, cur_nearest_poi)
            d2 = abs(doubt_poi[axia] - target_poi[axia])
            cnp_father = doubt_poi
            if d1 > d2:
                tmp = cur_nearest_poi
                cur_nearest_poi = doubt_poi
                if 'left' in cnp_father.keys() and 'right' in cnp_father.keys():
                    if cnp_father['left'] == tmp:
                        return backtrace(cur_nearest_poi, target_poi, cnp_father['right'] ) 
                    else:
                        return backtrace(cur_nearest_poi, target_poi, cnp_father['left'] )
                else:
                    if len(path_stack) != 0:
                        cnp_grand = path_stack.pop()
                        axia = path_stack.length() % 2
                        return backtrace(cur_nearest_poi, target_poi, cnp_grand)
                    else:
                        return cur_nearest_poi
            elif d1 < d2 and len(path_stack) != 0:
                doubt_poi = path_stack.pop()
                return backtrace(cur_nearest_poi, target_poi, doubt_poi)
            else:
                return cur_nearest_poi
        def cal_distance(poi1, poi2):
            return math.sqrt(pow((poi1[0] - poi2[0]), 2) + pow((poi1[1] - poi2[1]), 2))
        
        return fake_nearest        
