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
    
    w = np.zeros(2)
    b = 0
    a = 0.1


    for data_dict in x:
        
        x_data = np.array(data_dict['data'])

        judge = data_dict['flag']*(np.dot(x_data,w) + b)
        print('judge',judge)        
        while judge < 0:
            w = w + a*data_dict['flag']*np.array(data_dict['data'])
            print('weight:',w)
            b = b + a*data_dict['flag']
            judge = data_dict['flag']*(np.dot(x_data,w) + b)

    return w,b

if __name__ == "__main__":
    x = data_produce()
    w,b = perceptron(x)
    testx1 = -1000
    testy1 = testx1 * w[0] / (-1 * w[1]) + b/(-1 * w[1])
    testx2 = 1000
    testy2 = testx2 * w[0] / (-1 * w[1]) + b/(-1 * w[1])

    x1 = np.random.randint(0,1000,50)
    y1 = np.random.randint(0,1000,50)
    x2 = np.random.randint(1000,2000,50)
    y2 = np.random.randint(1000,2000,50)
    plt.scatter(x1,y1)
    plt.scatter(x2, y2)
    plt.plot([testx1,testx2],[testy1,testy2])
    plt.show()




