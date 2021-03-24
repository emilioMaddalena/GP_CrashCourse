'''
Author: Emilio Maddalena
Date: March 2021

A simple GP regression example
'''
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import gpflow
from gpflow.utilities import print_summary, set_trainable

N     = 30  # number of samples
sigma = 0.2 # noise variance

# Sampling the ground-truth 
xmin  = -2*np.pi
xmax  =  2*np.pi
X     = np.random.uniform(xmin, xmax, (N, 1))
delta = np.random.normal(0, sigma, (N, 1))
Y     = np.sin(X) + delta 

# Dataset needs to be converted to tensor for GPflow to handle it
data  = (tf.convert_to_tensor(X, "float64"), tf.convert_to_tensor(Y, "float64"))

# Defining the GP
kernel = gpflow.kernels.SquaredExponential()
my_gp  = gpflow.models.GPR(data, kernel=kernel) 

# Let's take a look at its hyperparameters (before training)
print_summary(my_gp)

# Picking an optimizer and training the GP through MLE
opt = gpflow.optimizers.Scipy()
opt.minimize(my_gp.training_loss, my_gp.trainable_variables, tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')

# Let's take a look at its hyperparameters (after training)
print_summary(my_gp)

# Gridding the space and predicting!
xx = np.linspace(xmin * 1.4, xmax * 1.4, 1000).reshape(-1, 1) 
mean, var = my_gp.predict_f(xx)

# Plotting the results (two standard deviations = 95% confidence)
fig = plt.figure()
plt.plot(xx, mean, color='#0072BD', lw=2)
plt.fill_between(xx[:,0],
                 mean[:,0] - 2 * np.sqrt(var[:,0]),
                 mean[:,0] + 2 * np.sqrt(var[:,0]),
                 color='#0072BD',
                 alpha=0.2)
plt.plot(X, Y, "o", color='#484848', ms=3.5)
plt.xlabel('x'), plt.ylabel('f(x)')
plt.title('Gaussian process regression with ' + str(N) + ' samples')
plt.show()
