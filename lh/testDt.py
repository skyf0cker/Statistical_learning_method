from DecisionTree2 import Decision_Tree
import numpy as np


def test_entropy():
    d = Decision_Tree()
    entropy = d.cal_emp_entropy(d.data)
    return entropy

def test_div():
	d = Decision_Tree()
	dim_data = d.data[0].transpose()
	# print(dim_data[0])
	div = d.div_point((d.data[1], dim_data[0]))
	print(div)

if __name__ == "__main__":
    # test_div()
	d = Decision_Tree()
	# print(d.data)
	d.compare_features()
	tree = d.planet_tree(d.data[0])