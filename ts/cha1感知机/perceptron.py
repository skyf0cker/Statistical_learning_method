import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generator_data():
    data = {
        'x': [],
        'y': []
    }
    np.random.seed(1)
    x1 = np.random.randint(0, 100, 50)
    y1 = np.random.randint(0, 100, 50)

    x2 = np.random.randint(100, 200, 50)
    y2 = np.random.randint(100, 200, 50)

    point_x = np.hstack((x1, x2))
    # print(len(point_x))
    point_y = np.hstack((y1, y2))

    for i in range(100):
        data['x'].append((point_x[i], point_y[i]))
        if(i < 50):
            data['y'].append(-1)
        else:
            data['y'].append(1)

    plt.scatter(x1, y1)
    plt.scatter(x2, y2)

    return data


class Perceptron(object):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']
        self.w = np.ones(2)
        self.b = 1
        self.lr = 0.1

    def update(self, x, y):
        self.w += self.lr * np.dot(y, x)
        self.b += self.lr * y

    def update1(self, x, y):
        return self.lr * np.dot(y, x), self.lr * y
    
    def judge(self, x, y):
        w = self.w
        t = np.dot(w, x) + self.b
        # print('t', t)
        # print('错误样本', w, t)
        if t * y <= 0:
            return True
        return False

    def train(self):
        tran_len = len(self.x)
        wrong_sample = True
        wrong_sample_num = 0
        while wrong_sample:
            if wrong_sample_num == 0:
                print('wrong_sample_num spbb', wrong_sample_num)
            wrong_sample_num = 0
            dw_sum = 0
            db_sum = 0
            dw = 0
            db = 0
            for i in range(tran_len):
                if self.judge(self.x[i], self.y[i]):
                    [dw, db] = self.update1(self.x[i], self.y[i])
                    dw_sum += dw
                    db_sum += db
                    wrong_sample_num += 1
                print('wrong_sample_num', wrong_sample_num)
                if not wrong_sample_num:
                    print('wrong_sample_num sp', wrong_sample_num)
                    wrong_sample = False
            if wrong_sample_num != 0:
                self.w += dw_sum / wrong_sample_num
                self.b += db_sum / wrong_sample_num
            
    


def main():
    data = generator_data()
    p = Perceptron(data)
    p.train()
    # w = p.w
    # b = p.b
    # plt.plot([0, 130], [325, 0])
    # plt.show()
    # testx1 = 0
    # testx2 = 


if __name__ == "__main__":
    main()
