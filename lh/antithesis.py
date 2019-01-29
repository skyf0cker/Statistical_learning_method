import numpy as np
import matplotlib.pyplot as plt


def data_produce():

    # ----------------
    # 产 生 数 据
    # ----------------

    x1 = np.random.randint(0, 100, 5)
    y1 = np.random.randint(0, 100, 5)
    x2 = np.random.randint(100, 200, 5)
    y2 = np.random.randint(100, 200, 5)

    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))

    final_data = []

    for i in range(10):
        data = [x[i], y[i]]
        if(i < 5):
            data_dict = {'data': data, 'flag': -1}
        else:
            data_dict = {'data': data, 'flag': 1}
        final_data.append(data_dict)

    return final_data


def antithesis(x):
    data_matrix = np.zeros((len(x), len(x)))        #初始化Gram矩阵
    b_sum = 0
    for data_dict in x:
        for data_dict2 in x:
            sum = np.dot(data_dict['data'], data_dict2['data'])
            data_matrix[x.index(data_dict)][x.index(data_dict2)] = sum
                                                    #  计算Gram矩阵
    
    all_true = False                                # 是否存在错误样本

    alpha = np.zeros((1, len(x)))[0]                # 初始化alpha矩阵
    learn_num = np.zeros((1, len(x)))[0]            # 初始化训练次数n
    b = 0
    a = 1
    update_num = 0
    while not all_true:                             # 只要存在错误样本就继续迭代

        wrong_num = 0                               # 初始化错误样本数

        for data_dict in x:                         # 迭代每一个点，检查是否存在误分类
            judge = 0                               
            fir_index = x.index(data_dict)          
            Alpha = 0                               # Alpha是每一个点对应的Alpha
            for data_dict2 in x:
                sec_index = x.index(data_dict2)
                Alpha = learn_num[sec_index] * a    # Alpha = n * a
                judge += Alpha * data_dict2['flag'] * data_matrix[sec_index][fir_index]   #计算每一个点误分类的判据
            learn_num[fir_index] += 1               # 对应训练次数加1

            if data_dict['flag'] * (judge + b) <= 0:
                # print('wrong', judge)
                update_num += 1
                Alpha += a
                alpha[fir_index] = Alpha
                b += data_dict['flag']
                wrong_num += 1
        print(wrong_num)

        if wrong_num == 0:
            break

    return alpha, b


# x = data_produce()
x = [{'data': [3, 3], 'flag':1}, {
    'data': [4, 3], 'flag':1}, {'data': [1, 1], 'flag':-1}]
alpha, b = antithesis(x)
result = np.zeros((1, 2))[0]
for data_dict in x:
    x_data = np.array(data_dict['data'])
    y_data = data_dict['flag']

    result += alpha[x.index(data_dict)] * y_data * x_data

# testx1 = -1000
# testy1 = testx1 * result[0] / (-1 * result[1]) + b/(-1 * result[1])
# testx2 = 3000
# testy2 = testx2 * result[0] / (-1 * result[1]) + b/(-1 * result[1])

# x1 = np.random.randint(0,1000,50)
# y1 = np.random.randint(0,1000,50)
# x2 = np.random.randint(1000,2000,50)
# y2 = np.random.randint(1000,2000,50)

# plt.scatter(x1,y1)
# plt.scatter(x2, y2)
# plt.plot([testx1,testx2],[testy1,testy2])
# plt.show()
