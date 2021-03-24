'''
Author: Emilio Maddalena
Date: March 2021

A more complicated GP regression problem
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling some warnings

import matplotlib as mtplt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary, set_trainable
from gt_funcs import schwefel_D

N     = # TO BE DEFINED
sigma = 20  # noise variance

f = schwefel_D()

# Sampling the ground-truth 
X     = np.random.uniform(f.xmin, f.xmax, (N, f.xdim))
delta = np.random.normal(0, sigma, (N, 1))
Y     = f(X) + delta
data  = (tf.convert_to_tensor(X, "float64"), tf.convert_to_tensor(Y, "float64"))

# Defining the GP
kernel = # TO BE DEFINED (see https://gpflow.readthedocs.io/en/master/gpflow/kernels)
my_gp  = gpflow.models.GPR(data, kernel=kernel) 

# Learning the GP
print_summary(my_gp)
opt = gpflow.optimizers.Scipy()
opt.minimize(my_gp.training_loss, my_gp.trainable_variables, tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')
print_summary(my_gp)

# Gridding the space and predicting!
x = np.linspace(f.xmin, f.xmax, f.gran)
x1, x2 = np.meshgrid(x[:,0], x[:,1])
x1, x2 = x1.reshape(-1,  1), x2.reshape(-1, 1)
xx = np.hstack((x1, x2)) 
mean, var = my_gp.predict_f(xx)

# Plotting the results
BLUE = '#0072BD'
mtplt.interactive(True)
fig = plt.figure(dpi=120, figsize=(10,4))
ax = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]

ax[0].plot_surface(x1.reshape(f.gran, f.gran), 
                  x2.reshape(f.gran, f.gran), 
                  f(xx).reshape(f.gran, f.gran), 
                  color=BLUE,
                  alpha=0.8)
ax[1].plot_surface(x1.reshape(f.gran, f.gran), 
                   x2.reshape(f.gran, f.gran), 
                   tf.reshape(mean, (f.gran, f.gran)),
                   color=BLUE,
                   alpha=0.8)
ax[1].scatter(X[:,0], X[:,1], Y, s = 15, color='#484848')
  
for cnt, axis in enumerate(ax):
    axis.set_xlabel('X1'), axis.set_ylabel('X2')
    axis.set_zlabel('f(x)') if cnt == 0 else axis.set_zlabel('GP(x)')
    axis.set_title('Ground-truth') if cnt == 0 else axis.set_title('GP reconstruction')
    axis.set_zlim(ax[0].get_zlim()) 
    axis.set_xticks([-400, -200, 0, 200, 400])
    axis.set_yticks([-400, -200, 0, 200, 400])
    axis.set_zticks(np.array([0, 500, 1000, 1500]))

plt.figure(dpi=120, figsize=(6,4))
ax3 = plt.axes()
pcm = ax3.pcolor(x1.reshape(f.gran, f.gran), 
                 x2.reshape(f.gran, f.gran), 
                 tf.reshape(var, (f.gran, f.gran)),
                 norm=colors.LogNorm(vmin=np.min(var), vmax=np.max(var)),
                 cmap='inferno')
fig.colorbar(pcm, ax=ax3, extend='max')
ax3.set_xlabel('X1'), ax3.set_ylabel('X2')
ax3.set_title('GP variance')

plt.show(block=True)
