import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generator_data():
    data = {
        'x': [],
        'y': []
    }
    np.random.seed(1)
    x1 = np.random.randint(-50, 50, 50)
    y1 = np.random.randint(-50, 50, 50)

    x2 = np.random.randint(50, 150, 50)
    y2 = np.random.randint(50, 150, 50)

    point_x1 = np.hstack((x1, x2))
    # print(len(point_x))
    point_x2 = np.hstack((y1, y2))

    data['x1'] = point_x1.reshape(1, 100)
    data['x2'] = point_x2.reshape(1, 100)
    for i in range(100):
        if(i < 50):
            data['y'].append(-1)
        else:
            data['y'].append(1)

    data['y'] = np.array(data['y']).reshape(1, 100)
    # plt.scatter(x1, y1)
    # plt.scatter(x2, y2)

    return data


class DualPerceptron(object):
    def __init__(self, data):
        self.x1 = data['x1']
        self.x2 = data['x2']
        self.y = data['y']
        self.w = np.ones(2)
        self.alpha = np.zeros((1, self.x2.shape[1]))
        print('alpha len', self.alpha.shape)
        self.b = 0
        self.lr = 1
        self.Gram = self.x1.T * self.x1 + self.x2.T * self.x2
        self.Gram1 = self.x1 * self.x1.T + self.x2 * self.x2.T
        print('self.Gram', self.Gram.shape)
        print('Gram1', self.Gram1.shape)

    def update(self, idx):
        # print('update idx', idx)
        # print(self.alpha[0, 0])
        self.alpha[0, idx] += self.lr
        self.b += self.y[0, idx]

    def judge(self, idx):
        y = self.y[0, idx]
        alpha_y = self.alpha * self.y
        # print('alpha_y', alpha_y.shape)
        # print('self.Gram[:, idx]', self.Gram[:, idx].reshape(1, 3).shape)
        _l = self.Gram.shape[0]
        _sum = np.sum(self.Gram[:, idx].reshape(1, _l) * alpha_y) + self.b
        # print('_sum', _sum, y)
        # print('_sum', _sum)
        if y * _sum <= 0:
            # print('错误样本', idx)
            return True
        # print('不是错误样本', idx)
        return False

    def train(self):
        tran_len = self.x1.shape[1]
        wrong_sample = True
        wrong_sample_num = 0
        while wrong_sample:
            wrong_sample_num = 0
            for i in range(tran_len):
                if self.judge(i):
                    self.update(i)
                    wrong_sample_num += 1
            if not wrong_sample_num:
                wrong_sample = False
            

def draw_line(w1, w2, b):
    y1 = b / -w1
    x2 = b / -w2
    plt.plot([0, y1], [x2, 0])            


def main():
    data = generator_data()
    dp = DualPerceptron(data)
    dp.train()
    print(dp.alpha.shape, dp.y.shape, dp.x1.shape)
    w1 = np.sum(dp.alpha * dp.y * dp.x1)
    w2 = np.sum(dp.alpha * dp.y * dp.x2)
    b = dp.b 

    plt.scatter(dp.x1, dp.x2)
    # plt.scatter(x2, y2)
    draw_line(w1, w2, b)
    plt.show()
    # print(w1, w2)



if __name__ == "__main__":
    main()
