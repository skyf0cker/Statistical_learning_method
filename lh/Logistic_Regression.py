import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

class Logistic_Regression:
    """
    the class to implement the Logistic Regreesion
    """

    def __init__(self, features, nums, classes):
        self.features = features
        self.nums = nums
        self.classes = classes
        self.alpha = 0.3
        self.max_epoch = 50
        self.data = self.make_data()
        self.theta = self.train()

    def make_data(self):
        """
        this is the function to make the training data
        :param features: the dimension of data features
        :param nums: the number of data
        :return:data
        """
        _dataset = make_classification(n_samples=self.nums, n_features=self.features, n_classes=self.classes)
        return _dataset

    def sigmoid(self, x):
        """
        this is the function to implement the sigmoid function
        :param x:this is (theta * x)
        :return:result: probility
        """
        result = 1/(1+np.exp(-x))
        return result

    def train(self, newton_mothod=False):
        if newton_mothod:
            pass
        else:
            theta = np.ones((self.features, 1))
            for i in range(self.max_epoch):
                grident = (1/self.nums)*np.dot(self.data[0].T, (self.sigmoid(np.dot(self.data[0], theta)) - self.data[1].reshape((20, 1))))
                theta = theta + self.alpha * grident
            return theta

    def test(self, data):
        return np.round(self.sigmoid(np.dot(self.theta.T, data.T)))
