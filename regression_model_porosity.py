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
    
    combine        = X[:,6]-X[:,3]
    
    input_linear_1 = np.expand_dims(X[:,4],  axis=1)
    input_linear_2 = np.expand_dims(combine,  axis=1)
    input_linear_3 = np.expand_dims(X[:,7],  axis=1)
    input_linear_4 = np.expand_dims(X[:,9],  axis=1)
    input_linear_5 = np.expand_dims(X[:,-1], axis=1)
        
    input_linear   = np.concatenate([input_linear_1,input_linear_2,input_linear_3,input_linear_4,
                                     input_linear_5], axis=1)
    
    input_cross_1  = np.expand_dims(X[:,4]*X[:,7],  axis=1)
    input_cross_2  = np.expand_dims(combine*X[:,7],  axis=1)
    input_cross_3  = np.expand_dims(X[:,7]*X[:,7],  axis=1)
    input_cross_4  = np.expand_dims(X[:,-1]*X[:,7],  axis=1)
  
    input_cross_5  = np.expand_dims(X[:,4]*X[:,9],  axis=1)
    input_cross_6  = np.expand_dims(combine*X[:,9],  axis=1)
    input_cross_7  = np.expand_dims(X[:,7]*X[:,9],  axis=1)
    input_cross_8  = np.expand_dims(X[:,-1]*X[:,9],  axis=1)
    
    input_cross_9  = np.expand_dims(X[:,4]*X[:,7]*X[:,7],  axis=1)
    input_cross_10  = np.expand_dims(combine*X[:,7]*X[:,7],  axis=1)
    input_cross_11  = np.expand_dims(X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_12  = np.expand_dims(X[:,-1]*X[:,7]*X[:,7], axis=1)
    
    input_cross_13  = np.expand_dims(X[:,4]*X[:,9]*X[:,9],  axis=1)
    input_cross_14  = np.expand_dims(combine*X[:,9]*X[:,9],  axis=1)
    input_cross_15  = np.expand_dims(X[:,7]*X[:,9]*X[:,9],  axis=1)
    input_cross_16  = np.expand_dims(X[:,-1]*X[:,9]*X[:,9], axis=1)
    
    input_cross_17  = np.expand_dims(X[:,4]*X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_18  = np.expand_dims(combine*X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_19  = np.expand_dims(X[:,7]*X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_20  = np.expand_dims(X[:,-1]*X[:,7]*X[:,7]*X[:,7], axis=1)
    
    input_cross_21  = np.expand_dims(X[:,4]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_22  = np.expand_dims(combine*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_23  = np.expand_dims(X[:,7]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_24  = np.expand_dims(X[:,-1]*X[:,9]*X[:,9]*X[:,9], axis=1)
    
    input_cross_25  = np.expand_dims(X[:,9]*X[:,9],  axis=1)
    input_cross_26  = np.expand_dims(X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_27  = np.expand_dims(X[:,9]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_28  = np.expand_dims(X[:,7]*X[:,7]*X[:,9],  axis=1)
    input_cross_29  = np.expand_dims(X[:,7]*X[:,7]*X[:,9]*X[:,9],  axis=1)
    input_cross_30  = np.expand_dims(X[:,7]*X[:,7]*X[:,7]*X[:,9],  axis=1)
    
    input_cross_31  = np.expand_dims(X[:,7]*X[:,7]*X[:,7]*X[:,7]*X[:,9],  axis=1)
    input_cross_32  = np.expand_dims(X[:,7]*X[:,7]*X[:,7]*X[:,9]*X[:,9],  axis=1)
    input_cross_33  = np.expand_dims(X[:,7]*X[:,7]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_34  = np.expand_dims(X[:,7]*X[:,9]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    
    input_cross_35  = np.expand_dims(X[:,7]*X[:,7]*X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_36  = np.expand_dims(X[:,9]*X[:,9]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    
    input_cross_37  = np.expand_dims(X[:,4]*X[:,9]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_38  = np.expand_dims(combine*X[:,9]*X[:,9]*X[:,9]*X[:,9],  axis=1)
    input_cross_39  = np.expand_dims(X[:,-1]*X[:,9]*X[:,9]*X[:,9]*X[:,9], axis=1)
    
    input_cross_40  = np.expand_dims(X[:,4]*X[:,7]*X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_41  = np.expand_dims(combine*X[:,7]*X[:,7]*X[:,7]*X[:,7],  axis=1)
    input_cross_42  = np.expand_dims(X[:,-1]*X[:,7]*X[:,7]*X[:,7]*X[:,7], axis=1)
    
    input_cross = np.concatenate([input_cross_1,input_cross_2,input_cross_3,input_cross_4,input_cross_5,input_cross_6,
                                  input_cross_7,input_cross_8,input_cross_9,input_cross_10,input_cross_11,input_cross_12,input_cross_13,
                                  input_cross_14,input_cross_15,input_cross_16,input_cross_17,input_cross_18,input_cross_19,
                                  input_cross_20,input_cross_21,input_cross_22,input_cross_23,input_cross_24,input_cross_25,input_cross_26,
                                  input_cross_27,input_cross_28,input_cross_29,input_cross_30,input_cross_31,input_cross_32,input_cross_33,
                                  input_cross_34,input_cross_35,input_cross_36,input_cross_37,input_cross_38,input_cross_39,input_cross_40,
                                  input_cross_41,input_cross_42], axis=1)
    
    input_data  = np.concatenate([input_linear,input_cross], axis=1) 
    
    return input_data

#%% Extract the dataset

index = 17

mat = scipy.io.loadmat('data.mat')
mat_input  = mat['num'][:,1:13]
mat_input  = np.concatenate((mat_input,np.expand_dims(mat['num'][:,19], axis=1)), axis=1)
mat_mean = np.mean(mat_input, axis=0)
mat_std  = np.std(mat_input, axis=0)

mat_norm = (mat_input - mat_mean) / mat_std

[m,n]    = mat_norm.shape
vec_zero = np.zeros((m,1))
vec_one  = np.ones((m,1))
mat_new  = np.concatenate([vec_one,mat_norm], axis=1)

input_data = feature_extract(mat_new)

output_data = mat['num'][:,index]

#%% Import modules

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#%% Train the model

X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=0.2)

rid = Ridge(fit_intercept=False, alpha = 1.0, max_iter=4000)
rid.fit(X_train, Y_train)

rid.coef_
rid.intercept_
pred_rid = rid.predict(X_test)

las = Lasso(fit_intercept=False, alpha = 0.5, max_iter=4000)
las.fit(X_train, Y_train)

las.coef_
las.intercept_
pred_las = las.predict(X_test)

reg = linear_model.LinearRegression(fit_intercept=False)
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

#%% Load trained Fully Connected Network


#%% Model obtained from Matlab

#%% Test the accuracy on test set

filename  = 'outfile_' + str(index) + '_new.txt'
test_load = np.loadtxt(filename)
Temp_X    = test_load[:,0:-1]
temp_load = np.expand_dims(test_load[:,10] + test_load[:,11], axis=1)
Temp_X    = np.concatenate((Temp_X, temp_load), axis=1)

Temp      = (Temp_X - mat_mean) / mat_std

[m,n] = test_load.shape
vec_zero = np.zeros((m,1))
vec_one  = np.ones((m,1))
test_new = np.concatenate([vec_one,Temp], axis=1)

Test_X = feature_extract(test_new)
Test_Y = test_load[:,-1]

Pred_reg   = reg.predict(Test_X)

plt.figure()
# plt.plot(Pred_mat)
# plt.plot(Pred_rid)
# plt.plot(Pred_las)
plt.plot(Pred_reg)
# plt.plot(Pred_ann)
plt.plot(Test_Y, 'r--')
# plt.legend(['Ridge Predict data', 'Lasso Predicted data', 'Linear Predicted data', 'Real data'])
plt.legend(['Linear Predicted data', 'Real data'])
plt.title('Effective Porosity Prediction')
plt.xlabel('Data Sample')
plt.ylabel('Effective Porosity')

Error_reg = np.linalg.norm(Pred_reg-Test_Y)

print('Error for LINEAR is %d', Error_reg)

#%% Optimization

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

#%% Establish Genetic Algorithm

from geneticalgorithm import geneticalgorithm as ga

def f1(x):
    
    x_new = np.array([2, 80, 120, 0, 0, 0, x[0], x[1], x[4]-x[1], x[2], x[3], x[4]-x[3], x[4]])
    
    x_new = (x_new - mat_mean) / mat_std
    
    x_new = np.expand_dims(x_new,  axis=0)
    x_new = np.insert(x_new, 0, 1, axis=1)
    
    x_new = feature_extract(x_new)
    
    return np.absolute(reg.predict(x_new))

def f2(x):
    
    x_new = np.array([x[0], 90/x[0], x[7]-x[2]-90/x[0]-67/x[1], x[1], 67/x[1], x[2], x[3], x[4], x[7]-x[4], x[5], x[6], x[7]-x[6], x[7]])
    
    x_new = (x_new - mat_mean) / mat_std
    
    x_new = np.expand_dims(x_new,  axis=0)
    x_new = np.insert(x_new, 0, 1, axis=1)
    
    x_new = feature_extract(x_new)
    
    return np.absolute(reg.predict(x_new))

#%% Start the algorithm

# x1_bounds  = [0.01, 0.01]
# x2_bounds  = [1, 80]
# x3_bounds  = [4, 4]
# x4_bounds  = [1, 80]
# x5_bounds  = [200, 200]

x1_bounds  = [1, 4]
x2_bounds  = [0, 4]
x3_bounds  = [0, 120]
x4_bounds  = [0.01, 0.01]
x5_bounds  = [1, 80]
x6_bounds  = [4, 4]
x7_bounds  = [1, 80]
x8_bounds  = [200, 200]

# bounds = np.array([x1_bounds, x2_bounds, x3_bounds, x4_bounds, x5_bounds])
bounds = np.array([x1_bounds, x2_bounds, x3_bounds, x4_bounds, x5_bounds, x6_bounds, x7_bounds, x8_bounds])

varbound = bounds

algorithm_param = {'max_num_iteration': 3000,\
                    'population_size':100,\
                    'mutation_probability':0.2,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

model = ga(function=f2, dimension=8, variable_type='real', 
           variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()

#%% Test the solution

solution = model.output_dict

sol = solution['variable']





