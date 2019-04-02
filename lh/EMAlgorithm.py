import numpy as np
import math

class EmAlgorithm:
    """
    Implements of EM algorithm
    """
    def __init__(self, data, models_num):
        
        self.data = np.array(data)
        self.data_num = len(data)
        self.models_num = models_num
        self.mu = np.array([[1,2]]).T
        self.theta = np.array([[1,1]]).T
        self.alpha = np.array([0.1, 0.9],dtype=float).reshape(1,2)
        self.gama = np.zeros((self.models_num, self.data_num))
        self.max_iters = 500

    def compute_gama(self):
        
        self.gauss = self._compute_Guass(self.data, self.mu, self.theta)
        print(self.gauss,"\n",self.alpha.T)
        self.gama =  self.gauss*self.alpha.T / np.sum(self.gauss*self.alpha.T,axis=0)

    def _compute_Guass(self, x, mu, theta):

        x = np.tile(x, (self.models_num,1))
        mu = np.tile(mu, (1,self.data_num))
        theta = np.tile(theta, (1,self.data_num))
        return 1/np.sqrt(2*np.pi)*theta*np.exp((x-mu)**2/2*theta**2)

    def update(self):

        self.mu = (np.sum(self.gama*self.data, axis=1) / np.sum(self.gama, axis=1)).T.reshape(self.models_num,1)

        self.theta = (np.sum((((np.tile(self.data, (2,1)).T - self.mu.T)**2)*self.gama.T).T,axis=1) / np.sum(self.gama, axis=1)).T.reshape(self.models_num,1)

        self.alpha = ((np.sum(self.gama, axis=1) / self.data_num).T.reshape(self.models_num,1)).T

    def train(self):
        
        for i in range(self.max_iters):
            print(i)
            self.compute_gama()
            self.update()

        print("alpha:", self.alpha)
        print("mu:", self.mu)
        print("theta:", self.theta)


