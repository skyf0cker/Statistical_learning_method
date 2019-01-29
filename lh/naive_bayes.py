import numpy as np
import math


class NaiveBayes:
    """
    a step to archieve my dream
    """
    def __init__(self, data_dim, data_num, class_num = 2):
        super(NaiveBayes, self).__init__()
        self.data_dim = data_dim
        self.data_num = data_num
        self.class_num = class_num
        self.data = self.data_make(data_dim, data_num)
        self.mean = np.zeros((class_num, data_dim)).reshape((class_num, data_dim))
        self.var = np.zeros((class_num, data_dim)).reshape((class_num, data_dim))
        self.p_y1 = 0
        self.p_y2 = 0

    def data_make(self, data_dim, data_num):
        """
        make train data
        :param data_dim: dimension of the feature
        :param data_num: the number of train data
        :return: a list of data whose type is dict
        """
        train_data = []
        for _ in range(data_num):

            tmp_data = {}
            tmp_data["data"] = np.random.randn(1, data_dim)
            if _ % 2 == 0:
                flag = -1
            else:
                flag = 1
            tmp_data["flag"] = flag

            train_data.append(tmp_data)
        return train_data

    def fit(self):
        """
        calculate the mean and var of the data
        :return: NULL
        """
        data_list_1 = [data["data"] for data in self.data if data["flag"] == 1]
        data_list_2 = [data["data"] for data in self.data if data["flag"] == -1]
        self.p_y1 = len(data_list_1)/self.data_num
        self.p_y2 = len(data_list_2)/self.data_num
        self.mean[0] = np.mean(data_list_1, axis=0, keepdims=True).reshape(self.data_dim)
        self.mean[1] = np.mean(data_list_2, axis=0, keepdims=True).reshape(self.data_dim)
        self.var[0] = np.std(data_list_1, axis=0, keepdims=True, ddof=1).reshape(self.data_dim)
        self.var[1] = np.std(data_list_1, axis=0, keepdims=True, ddof=1).reshape(self.data_dim)

    def predict(self, data):
        """
        to classify the data
        :param data: the data u wanna classify
        :return: int
        """
        p1 = (1/(pow(2*math.pi, 0.5)*self.var[0][0]))*pow(math.e, -(pow((data[0] - self.mean[0][0]), 2)/2*pow(self.var[0][0], 2)))*(1/(pow(2*math.pi, 0.5)*self.var[0][1]))*pow(math.e, -(pow((data[1] - self.mean[0][1]), 2)/2*pow(self.var[0][1], 2)))
        p2 = (1/(pow(2*math.pi, 0.5)*self.var[1][0]))*pow(math.e, -(pow((data[0] - self.mean[1][0]), 2)/2*pow(self.var[1][0], 2)))*(1/(pow(2*math.pi, 0.5)*self.var[1][1]))*pow(math.e, -(pow((data[1] - self.mean[1][1]), 2)/2*pow(self.var[1][1], 2)))


        print(p1,',',p2)

        if p2 > p1:
            print("[*]: the data belongs to the second class")
        else:
            print("[*]: the data belongs to the first class")
