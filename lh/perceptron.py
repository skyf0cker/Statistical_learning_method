import numpy as np
import matplotlib.pyplot as plt

def data_produce():
    
    # ----------------
    # 产 生 数 据
    # ----------------
    
    x1 = np.random.randint(0,1000,50)
    y1 = np.random.randint(0,1000,50)
    x2 = np.random.randint(1000,2000,50)
    y2 = np.random.randint(1000,2000,50)
    
    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))

    final_data = []

    for i in range(100):
        data = [x[i],y[i]]
        if(i < 50):
            data_dict = {'data':data,'flag':-1}
        else:
            data_dict = {'data':data, 'flag':1}
        final_data.append(data_dict)

    return final_data

def perceptron(x):
    
    w = np.ones(2)
    b = 0
    a = 3
    all_true = False

    while(not all_true):
        update_w = 0
        update_b = 0
        wrong_num = 0
        for data_dict in x:
        # ------------
        # 遍历所有的点
        # ------------
            x_data = np.array(data_dict['data'])
            judge = data_dict['flag']*(np.dot(x_data,w) + b)
            if judge < 0:
                wrong_num += 1
                print('wrong:', data_dict['data'])
                update_w += data_dict['flag']*np.array(data_dict['data'])
                update_b += data_dict['flag']

        print('wrong_num',wrong_num)
        if wrong_num == 0:
            all_true = True
            break
        w = w + a*update_w/wrong_num
        b = b + a*update_b/wrong_num
        print('w:',w)
        print('b',b)



    return w,b

if __name__ == "__main__":
    x = data_produce()
    w,b = perceptron(x)
    testx1 = -1000
    testy1 = testx1 * w[0] / (-1 * w[1]) + b/(-1 * w[1])
    testx2 = 3000
    testy2 = testx2 * w[0] / (-1 * w[1]) + b/(-1 * w[1])

    x1 = np.random.randint(0,1000,50)
    y1 = np.random.randint(0,1000,50)
    x2 = np.random.randint(1000,2000,50)
    y2 = np.random.randint(1000,2000,50)
    plt.scatter(x1,y1)
    plt.scatter(x2, y2)
    plt.plot([testx1,testx2],[testy1,testy2])
    plt.show()




