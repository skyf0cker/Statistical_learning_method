import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]

class SVM:

    def __init__(self, data, max_iters=1000, kernal="linear", C=0.0001):
        self.max_iters = max_iters
        self.Kernal = kernal
        self.load_data(data[0], data[1])
        self.C = C

    def load_data(self, X, y):
        self.X = X
        self.Y = y

    def init_args(self):

        self.data_num = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.alpha = np.zeros(self.data_num)
        self.b = 0
        self.f = [self.com_f(i) for i in range(self.data_num)]
        self.E = [self.com_E(i) for i in range(self.data_num)]

    def com_f(self, idx):
        
        return self.kernal(np.multiply(self.alpha, self.Y), np.dot(self.X[idx, :], self.X.T)) + self.b

    def com_E(self, idx):

        return self.f[idx] - self.Y[idx]

    def com_eta(self, i, j):

        return self.kernal(self.X[i], self.X[i]) + self.kernal(self.X[j], self.X[j]) - 2*self.kernal(self.X[i], self.X[j])

    def search_sample(self):
        
        l1 = [i for i in range(self.data_num) if 0 < self.alpha[i] < self.C]
        l2 = [i for i in range(self.data_num) if i not in l1]
        l1.extend(l2)

        for i in l1:
            if self.kkt(i):
                continue
            E = self.E[i]

            if E > 0:
                j = min(range(self.data_num), key=lambda x: self.E[x])
            else:
                j = max(range(self.data_num), key=lambda x: self.E[x])

            return i, j

    def kkt(self, idx):

        judge = self.Y[idx]*self.f[idx]
        if self.alpha[idx] == 0:
            return judge >= 1 
        elif self.alpha[idx] > 0 and self.alpha[idx] < self.C:
            return judge == 1
        else:
            return judge <= 1

    def compare(self, alpha_not_clip, L, H):

        if alpha_not_clip < L:
            return L
        elif alpha_not_clip > H:
            return H
        else:
            return alpha_not_clip

    def kernal(self, x_i, x_j):
        if self.Kernal == "linear":
            return np.dot(x_i, x_j)
        elif self.Kernal == "poly":
            return pow(np.dot(x_i, x_j)+1, 2)

    def train(self):

        for _ in range(self.max_iters):
            i, j = self.search_sample()
            print(i,",",j)
            eta = self.com_eta(i, j)
            # print(eta)
            if self.Y[i] == self.Y[j]:
                L = max(0, self.alpha[i]+self.alpha[j] - self.C)
                H = min(self.C, self.alpha[i]+self.alpha[j])
            else:
                L = max(0, self.alpha[j]-self.alpha[i])
                H = min(self.C, self.C + self.alpha[j]-self.alpha[i])

            E1 = self.E[i]
            E2 = self.E[j]


            if eta == 0:
                continue

            alpha2_new_unclip = self.alpha[j] + self.Y[j]*(E1 - E2) / eta
            alpha2 = self.compare(alpha2_new_unclip, L, H)
            alpha1 = self.alpha[i] + self.Y[i] * self.Y[j] * (self.alpha[j] - alpha2)
            
            b1_new = -E1 - self.Y[i] * self.kernal(self.X[i], self.X[i]) * (
                alpha1-self.alpha[i]) - self.Y[j] * self.kernal(self.X[j], self.X[i]) * (alpha2-self.alpha[j]) + self.b
            b2_new = -E2 - self.Y[i] * self.kernal(self.X[i], self.X[j]) * (
                alpha1-self.alpha[i]) - self.Y[j] * self.kernal(self.X[j], self.X[j]) * (alpha2-self.alpha[j]) + self.b

            if 0 < alpha1 < self.C:
                b_new = b1_new
            elif 0 < alpha2 < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i] = alpha1
            self.alpha[j] = alpha2
            self.b = b_new
            self.E[i] = self.com_E(i)
            self.E[j] = self.com_E(j)

        print('train done!')

    def predict(self, x):
        
        r = self.kernal(np.multiply(self.alpha, self.Y), np.dot(x, self.X.T)) + self.b
        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        
        r = 0
        for i in range(len(X_test)):
            x = X_test[i]
            result = self.predict(x)
            if result == y_test[i]:
                r += 1
        return r / len(X_test)

    def weight(self):
        """
        计算权值
        """
        yx = self.Y.reshape(-1, 1)*self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w


def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:, 0], X[50:, 1], label='1')
    svm = SVM((X_train, y_train))

    svm.init_args()

    svm.train()

    score = svm.score(X_test, y_test)

    print('score', score)

    a1, a2 = svm.weight()
    b = svm.b
    x_min = min(svm.X, key=lambda x: x[0])[0]
    x_max = max(svm.X, key=lambda x: x[0])[0]

    y1, y2 = (-b - a1 * x_min)/a2, (-b - a1 * x_max)/a2
    plt.plot([x_min, x_max], [y1, y2])
    plt.show()


if __name__ == "__main__":
    main()
                
        