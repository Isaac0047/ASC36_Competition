import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
# import tensorflow as tf

import pandas as pd
import scipy.io

#%% Extract the data inputs

# Heat Rate 1
# Ramp 1 Duration
# Temperature Dwell 1
# Heat Rate 2
# Ramp 2 Duration
# Temprature Dwell 2
# Vacuum Pressure
# Vacuum start time
# Vacuum duration
# Autoclave Pressure
# Autoclave Start Time
# Autoclave Duration

# AD Porosity -- 15
# PR Porosity -- 16
# Eff Porosity -- 17
# Max Fiber Volume Fraction -- 18
# Cure Cycle Total Time     -- 19
# AD Volume    -- 20
# PR Volume    -- 21

#%% Pre-Define modules

def feature_extract(X):
    
    input_linear_3 = np.expand_dims(X[:,7],  axis=1)
    input_linear_6 = np.expand_dims(X[:,10], axis=1)
    
    
    input_linear   = np.concatenate([input_linear_3,input_linear_6], axis=1)
    
    input_data  = input_linear
    
    return input_data

#%% Extract the dataset

index = 18

mat = scipy.io.loadmat('data.mat')
mat_input  = mat['num'][:,1:13]
# mat_input  = np.concatenate((mat_input,np.expand_dims(mat['num'][:,19], axis=1)), axis=1)
mat_mean = np.mean(mat_input, axis=0)
mat_std  = np.std(mat_input, axis=0)

mat_norm = (mat_input - mat_mean) / mat_std

[m,n]    = mat_norm.shape
vec_zero = np.zeros((m,1))
vec_one  = np.ones((m,1))
mat_new  = np.concatenate([vec_one,mat_norm], axis=1)

input_data = feature_extract(mat_new)
# input_data  = mat_norm

output_data = mat['num'][:,index]

#%% Import modules

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#%% Train the model

X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=0.2)

rid = Ridge(fit_intercept=True, alpha = 1.0, max_iter=4000)
rid.fit(X_train, Y_train)

rid.coef_
rid.intercept_
pred_rid = rid.predict(X_test)

las = Lasso(fit_intercept=True, alpha = 0.5, max_iter=4000)
las.fit(X_train, Y_train)

las.coef_
las.intercept_
pred_las = las.predict(X_test)

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train, Y_train)

reg.intercept_
reg.coef_
pred_reg = reg.predict(X_test)

#%% Evaluating the error

err_rid  = np.linalg.norm(Y_test - pred_rid)
err_las  = np.linalg.norm(Y_test - pred_las)
err_reg  = np.linalg.norm(Y_test - pred_reg)

from sklearn.metrics import mean_squared_error

mse_rid  = mean_squared_error(Y_test, pred_rid)
mse_las  = mean_squared_error(Y_test, pred_las)
mse_reg  = mean_squared_error(Y_test, pred_reg)

#%% Extract random rows (This part of code is for one-time use)

# Y_test_new = np.expand_dims(Y_test, axis=1)
# data = np.concatenate((X_test, Y_test_new), axis=1)
# df = pd.DataFrame(data)
# test_sample = df.sample(n=15)

# filename = 'outfile_' + str(index) + '.txt'

# np.savetxt(filename, test_sample)

#%% Load trained Fully Connected Network

# model_name = 'my_model_' + str(index) + '.h5'
# new_model = tf.keras.models.load_model(model_name)

#%% Model obtained from Matlab

#%% Test the accuracy on test set

filename  = 'outfile_' + str(index) + '_new.txt'
test_load = np.loadtxt(filename)
Temp_X    = test_load[:,0:-1]
# temp_load = np.expand_dims(test_load[:,10] + test_load[:,11], axis=1)
# Temp_X    = np.concatenate((Temp_X, temp_load), axis=1)

Temp      = (Temp_X - mat_mean) / mat_std

[m,n] = test_load.shape
vec_zero = np.zeros((m,1))
vec_one  = np.ones((m,1))
test_new = np.concatenate([vec_one,Temp], axis=1)

Test_X = feature_extract(test_new)
# Test_X = Temp
Test_Y = test_load[:,-1]

# Pred_mat   = np.dot(Test_X, coeff)
Pred_rid   = rid.predict(Test_X)
Pred_las   = las.predict(Test_X)
Pred_reg   = reg.predict(Test_X)
# Pred_ann = new_model.predict(Test_X) 

plt.figure()

plt.plot(Pred_reg)
plt.plot(Test_Y, 'r--')
plt.legend(['Linear Predicted data', 'Real data'])
plt.title('Consolidation Prediction')
plt.xlabel('Data Sample')
plt.ylabel('Consolidation')

#%% Test only

# test = np.array([  2,  71, 101,   0,  11, 114,  0,  76, 248,  4,  33, 224])
# test = np.array([  5,  29,  67,   0,   0,   0,  0,   7, 233,  2, 108, 132])
# test = np.array([  2,  41,  80,   2,  29, 119,   0,  31, 239,   2,   2, 268])
# test = np.array([  2,  39,  82,   1,  47,  57,   0,  18, 207,   4,  28, 197])
# test = np.array([ 20,   4, 107,   1,  59,  27,   0,  14, 185,   4,  54, 145])
# test = np.array([  3,  28,  73,   6,   9,   8,   0,  29,  90,   4,  27,  92])
# test = npt.array([  2,  44,  85,   2,  28,  40,   0,  27, 172,   4,  57, 142])
# test = np.array([  2,  80, 120,   0,   0,   0,   0.01,  79, 120,   2,  66, 133])
test = np.array([  2,  80, 120,   0,   0,   0,   0.01,  79, 120,   4,  55, 144])

def f(x):

    x = (x - mat_mean) / mat_std
    
    x = np.expand_dims(x,  axis=0)
    x = np.insert(x, 0, 1, axis=1)
    
    x = feature_extract(x)
    
    return np.absolute(reg.predict(x))

result = f(test)

























