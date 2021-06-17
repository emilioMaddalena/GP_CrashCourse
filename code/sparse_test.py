'''
Author: Emilio Maddalena
Date: March 2021

A simple sparse GP regression example
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import numpy as np
import matplotlib.pyplot as plt

import gpflow
from gpflow.models import GPR, SGPR, GPRFITC
from gpflow.utilities import print_summary, set_trainable
import tensorflow as tf
from gt_funcs import GandL_1D

N      = 50  
sigma  = 0.2

f     = GandL_1D()
X     = np.random.uniform(f.xmin, f.xmax, (N, f.xdim))
delta = np.random.normal(0, sigma, (N, 1))
Y     = f(X) + delta
data  = (tf.convert_to_tensor(X, "float64"), tf.convert_to_tensor(Y, "float64"))

opt = gpflow.optimizers.Scipy()

##############################
# Exact GP part
##############################

kernel = gpflow.kernels.SquaredExponential()
my_gp  = gpflow.models.GPR(data, kernel=kernel) 

print('GPR init:'), print_summary(my_gp)
opt.minimize(my_gp.training_loss, my_gp.trainable_variables, tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')
print('GPR trained:'), print_summary(my_gp)

xx = np.linspace(f.xmin, f.xmax, 1000).reshape(-1, 1) 
mean, var = my_gp.predict_f(xx)

fig = plt.figure(1)
plt.xkcd()
plt.plot(xx, mean, color='#0072BD', lw=2, label="predictive mean")
plt.fill_between(xx[:,0],
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='#0072BD',
                 alpha=0.3)
plt.plot(X, Y, "o", color='#484848', ms=3.5, label='samples')
plt.xlabel('x'), plt.ylabel('f(x)')
plt.title('Exact GP with ' + str(N) + ' samples')
plt_xmin, plt_xmax, plt_ymin, plt_ymax = plt.axis()

##############################
# Sparse GP part
##############################

M = # Inducing points TO BE DEFINED
U = np.random.uniform(f.xmin, f.xmax, (M, f.xdim))
U = tf.convert_to_tensor(U, "float64")

kernel = gpflow.kernels.SquaredExponential()
my_sgp = gpflow.models.SGPR(data, kernel=kernel, inducing_variable=U)

print('VFE initialization:'), print_summary(my_sgp)
opt.minimize(my_sgp.training_loss, my_sgp.trainable_variables, options=dict(maxiter=1000),method='l-bfgs-b')
print('VFE trained:'), print_summary(my_sgp)

mean, var = my_sgp.predict_f(xx)
mean_U, _ = my_sgp.predict_f(U)

fig2 = plt.figure(2)
plt.plot(xx, mean, color='#FF6961', lw=2, label="predictive mean")
plt.fill_between(xx[:,0],
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='#FF6961',
                 alpha=0.3)
plt.plot(U, mean_U, "o", color='#484848', ms=3.5, label='samples')
plt.xlabel('x'), plt.ylabel('f(x)')
plt.xlim((plt_xmin, plt_xmax)), plt.ylim((plt_ymin, plt_ymax))
plt.title('SGP with ' + str(M) + ' inducing points')

plt.show()
plt.pause(0.1)
