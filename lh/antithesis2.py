import numpy as np
import matplotlib.pyplot as plt


class antithesis:

    def __init__(self, data):
        self.lr = 0.5
        self.data = data
        self.alpha = np.zeros((1, len(data))).reshape(len(data))
        self.bias = 0
        self.gram = np.zeros((len(data), len(data)))
        self.wrong_num = 0

    def judge(self, index):

        w = 0
        for i in self.data:
            w += self.alpha[self.data.index(i)]*self.data[self.data.index(i)]['flag'] * \
                self.gram[self.data.index(i)][index]
            
            # print('-----------------------')
            # print('alpha:',self.alpha)
            # print('w:',w)
            # print('bias',self.bias)
            # print('point:',i)

        judge = self.data[index]['flag'] * (w + self.bias)
        print('judge:', judge)
        
        if judge <= 0:
            return True        # True means wrong answer
        else:
            return False

    def cal_wrong_sample(self):
        for i in self.data:
            if self.judge(self.data.index(i)):
                self.wrong_num += 1
            else:
                continue

    def update(self, index):
        # print('i have updated')
        self.alpha[index] += self.lr
        self.bias += self.data[index]['flag']

    def get_gram(self):

        for i in self.data:
            for j in self.data:
                index_1 = self.data.index(i)
                index_2 = self.data.index(j)
                self.gram[index_1][index_2] = np.dot(i['data'], j['data'])


def data_produce():

    x1 = np.random.randint(0, 100, 10)
    y1 = np.random.randint(0, 100, 10)
    x2 = np.random.randint(100, 200, 10)
    y2 = np.random.randint(100, 200, 10)

    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))

    final_data = []

    for i in range(20):
        data = [x[i], y[i]]
        if(i < 10):
            data_dict = {'data': data, 'flag': -1}
        else:
            data_dict = {'data': data, 'flag': 1}
        final_data.append(data_dict)

    return final_data

if __name__ == "__main__":


    # test_data = [
    #     {'data': [8, 4], 'flag':1},
    #     {'data': [3, 3], 'flag':1},
    #     {'data': [0, 1], 'flag':-1},
    #     {'data': [-1, 0], 'flag':-1},
    #     {'data': [0, 2], 'flag':-1}
    # ]
    #
    real_data = data_produce()

    # print(real_data)

    a = antithesis(real_data)
    a.get_gram()                 # cal the gram matrix


    while True:
        wrong_num = 0
        for i in a.data:
            i_index = a.data.index(i)
            if a.judge(i_index):
                wrong_num += 1
                a.update(i_index)

        # a.cal_wrong_sample()
        print('[*]: wrong number: ', wrong_num)
        if wrong_num == 0:
            break

    w = np.zeros((1, 2)).reshape(2)
    for i in a.data:

        index = a.data.index(i)
        w += a.alpha[index] * a.data[index]['flag'] * \
            np.array(a.data[index]['data']).reshape(2)

    print(w)
    print(a.bias)

    testx1 = -100
    testy1 = testx1 * w[0] / (-1 * w[1]) + a.bias/(-1 * w[1])
    testx2 = 200
    testy2 = testx2 * w[0] / (-1 * w[1]) + a.bias/(-1 * w[1])

    x1 = np.random.randint(0,100,10)
    y1 = np.random.randint(0,100,10)
    x2 = np.random.randint(100,200,10)
    y2 = np.random.randint(100,200,10)
    #
    plt.scatter(x1,y1)
    plt.scatter(x2, y2)
    plt.plot([testx1,testx2],[testy1,testy2])
    plt.show()
