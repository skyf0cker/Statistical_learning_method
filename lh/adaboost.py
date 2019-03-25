import numpy as np
import json
from collections import Counter
import math

class adaboost:

    def __init__(self, max_iters=100):
        self.m = 0
        self.step = 0.1
        self.load_data()
        self.max_iters = max_iters
        self.data_num = len(self.data)
        self.D = np.ones((1,self.data_num)).reshape((self.data_num,))/self.data_num
        self.get_best_judge()
        self.judge = self.judge_list[self.m]
        self.am_list = []
        
    def get_best_judge(self):
        judge_list = [i for i in np.arange(0, self.data_num, 0.1)]
        judge_list.sort(key=lambda x:self.com_em(x))
        self.judge_list = judge_list

    def load_data(self):

        with open("./sample.json", "rb") as j:
            data = json.load(j)
            self.data = [i["data"] for i in list(data.values())]

    def basic_classifier(self, x, judge):

        if x < judge:
            return 1
        else:
            return -1

    def com_em(self, judge):
        
        error_num = len([i for i in self.data if i[1] != self.basic_classifier(i[0], judge)])
        return error_num/self.data_num

    def com_am(self, em):
        
        return 1/2*np.log((1 - em) / em)

    def update(self):
        
        self.judge = self.judge_list[self.m]
        em = self.com_em(self.judge)
        am = self.com_am(em)
        zm = self.com_zm(am)
        self.am_list.append(am)
        for i in range(self.data_num):
            data = self.data[i]
            self.D[i] = (self.D[i] / zm) * math.exp(-am * data[1] * self.basic_classifier(data[0], self.judge))

        self.m += 1

    def com_zm(self, am):

        z = 0
        for i in range(self.data_num):
            z += self.D[i] * math.exp(-am*self.data[i][1]*self.basic_classifier(self.data[i][0], self.judge))

        return z

    def train(self):

        for iter in range(self.max_iters):

            self.update()

    def predict(self, x):
        
        result = 0
        for i in range(self.m):
            result += self.basic_classifier(x, self.judge)*self.am_list[i]

        return int(result > 0)

def main():

    ada = adaboost()
    ada.train()
    result = ada.predict(3)
    print(result)

if __name__ == "__main__":
    main()