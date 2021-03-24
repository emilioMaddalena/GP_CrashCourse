'''
Author: Emilio Maddalena
Date: March 2021

A library of interesting test functions:
https://www.sfu.ca/~ssurjano/optimization.html

Here we assume the inputs x have shape (N,D) because of GPflow
N is the number of data, and D the input dimension

The returned outputs y are always (N,1)
'''

import numpy as np

class Ackley_1D:
    
    def __init__(self):
        
        self.name = 'Ackley'
        self.xdim = 1
        self.xmin = -40.
        self.xmax = 40.
        self.analytic_min = 0.
        self.gran = 5000
        
    def __call__(self, x):
        
        a = 20.
        b = 0.2
        c = 2*np.pi
            
        term1 = -a * np.exp(-b * np.sqrt(x**2))
        term2 = -np.exp(np.cos(c * x))
    
        output = term1 + term2 + a + np.exp(1)
        return output.reshape(-1, 1)
    
class GandL_1D:
    
    def __init__(self):

        self.name = 'GandL'
        self.xdim = 1
        self.xmin = [0.5]
        self.xmax = [2.5]
        self.gran = 200
        self.analytic_min = 0.13098886501   
        
    def __call__(self, x):
        
        output = np.divide(np.sin(10*np.pi*x), 2*x) + np.power((x - 1), 4) + 1
        return output.reshape(-1, 1)
    
class branin_2D:
    
    def __init__(self):
        
        self.name = 'branin'
        self.PARAMS = {
                "a" : 1,
                "b" : 5.1 / (4 * np.pi**2),
                "c" : 5 / np.pi,
                "r" : 6,
                "s" : 10,
                "t" : 1 / (8 * np.pi)
                }
        self.xdim = 2
        self.xmin = [-5, 0]
        self.xmax = [10, 15]
        self.gran = 100
        self.analytic_min = 0.0
        
    def __call__(self, x):
        
        PARAMS = self.PARAMS        
        x1 = x[:,0]
        x2 = x[:,1]
        
        output = - 0.397887 + PARAMS["a"]* (x2 - PARAMS["b"]*x1**2 + PARAMS["c"]*x1 - PARAMS["r"])**2 + PARAMS["s"]*(1-PARAMS["t"])*np.cos(x1) + PARAMS["s"]
        
        return output.reshape(-1, 1)
    
class hartmann_3D:
    
    def __init__(self):
        
        self.name = 'hartmann'
        self.PARAMS = {
                "alpha" : [1, 1.2, 3, 3.2],
                "A" : [[3, 10, 30],
                       [0.1, 10, 35],
                       [3, 10, 30],
                       [0.1, 10, 35]],
                "P" : [[0.3689, 0.117, 0.2673],
                       [0.4699, 0.4387, 0.7470],
                       [0.1091, 0.8732, 0.5547],
                       [0.0381, 0.5743, 0.8828]]
                }
        self.xdim = 3
        self.xmin = [0, 0, 0]
        self.xmax = [1, 1, 1]
        self.gran = 100
        self.analytic_min = 0
        
    def __call__(self, x):
        
        alpha = np.array(self.PARAMS["alpha"])
        A     = np.array(self.PARAMS["A"])
        P     = np.array(self.PARAMS["P"])
        N     = x.shape[0]
        xdim  = self.xdim
        
        # Shifting the sample index to the third dimension
        x = x.T.reshape((1, xdim, N))
        A = np.repeat(A[:,:,np.newaxis], N, axis=2)
        P = np.repeat(P[:,:,np.newaxis], N, axis=2)
        alpha = np.repeat(alpha[:,np.newaxis], N, axis=1)
        
        output = 3.86278 - np.sum(np.multiply(alpha, np.exp(-np.sum(np.multiply(A, np.square(x-P)), axis=1))), axis=0)
        
        return output.reshape(-1, 1)
    
class schwefel_D:
    
    def __init__(self):
        
        d = input("Specify the Perm function dimension d (integer): ")
        d = int(d)
        
        self.name = 'schwefel'
        self.xdim = d
        self.xmin = -500 * np.ones(d)
        self.xmax =  500 * np.ones(d)
        self.gran = 100
        self.analytic_min = 0
        
    def __call__(self, x):
        
        d = self.xdim
        
        temp1 = np.absolute(x)
        temp2 = np.sqrt(temp1)
        temp3 = np.sin(temp2)
        temp4 = x * temp3
        temp5 = np.sum(temp4, axis=1)
        output = 418.9829*d - temp5

        return output.reshape(-1, 1)
    
class perm_D:
    
    def __init__(self):
        
        d = input("Specify the Perm function dimension d (integer): ")
        d = int(d)
        
        self.name = 'perm'
        self.xdim = d
        self.beta = 0.5
        self.xmin = -d * np.ones(d)
        self.xmax =  d * np.ones(d)
        self.gran = 100
        self.analytic_min = 0
        
    def __call__(self, x):
        
        d = self.xdim
        N = x.shape[0]
        beta = self.beta
        
        vec = np.arange(1, d + 1)        
        output = np.zeros((N, 1))
        
        for i in range(1, d + 1):
            
            temp1 = np.power(vec, i) + beta
            temp2 = np.power(np.divide(x, vec), i) - 1
            temp3 = np.power(np.sum(temp1*temp2, axis=1), 2)
            output = output + temp3.reshape((N, 1))
            
        return output.reshape(-1, 1)
    