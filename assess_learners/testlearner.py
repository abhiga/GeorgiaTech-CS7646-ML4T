"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""

import numpy as np
import math
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import pandas as pd
import matplotlib.pyplot as plt
import time

import sys

if __name__ == "__main__":
    path = "Data/Istanbul.csv"
    if len(sys.argv) == 2:
        path = sys.argv[1]

    inf = open(path)
    alldata = np.genfromtxt(inf, delimiter=',')
    if path == "Data/Istanbul.csv":
        alldata = alldata[1:, 1:]

    datasize = alldata.shape[0]
    cutoff = int(datasize * 0.6)
    permutation = np.random.permutation(alldata.shape[0])
    col_permutation = np.random.permutation(alldata.shape[1] - 1)
    train_data = alldata[permutation[:cutoff], :]
    trainX = train_data[:, col_permutation]
    trainY = train_data[:, -1]
    test_data = alldata[permutation[cutoff:], :]
    testX = test_data[:, col_permutation]
    testY = test_data[:, -1]

    dt_in_sample_error_rmse = []
    dt_out_sample_error_rmse = []
    bl_in_sample_error_rmse = []
    bl_out_sample_error_rmse = []
    for i in range(1, 50):

        # Question 1: Decision Tree
        dt_tree = dt.DTLearner(i)
        dt_tree.addEvidence(trainX, trainY)
        dt_train_pred_y = dt_tree.query(trainX)
        dt_train_mean_error = math.sqrt(((trainY - dt_train_pred_y) ** 2).sum() / trainY.shape[0])
        dt_in_sample_error_rmse.append(dt_train_mean_error)
        dt_test_pred_y = dt_tree.query(testX)
        dt_test_mean_error = math.sqrt(((testY - dt_test_pred_y) ** 2).sum() / testY.shape[0])
        dt_out_sample_error_rmse.append(dt_test_mean_error)

        # Question 2: Bagging
        bl_tree = bl.BagLearner(dt.DTLearner, {'leaf_size': i}, 10)
        bl_tree.addEvidence(trainX, trainY)
        bl_train_pred_y = bl_tree.query(trainX)
        bl_train_rmse = math.sqrt(((trainY - bl_train_pred_y) ** 2).sum() / trainY.shape[0])
        bl_in_sample_error_rmse.append(bl_train_rmse)
        bl_test_pred_y = bl_tree.query(testX)
        bl_test_rmse = math.sqrt(((testY - bl_test_pred_y) ** 2).sum() / testY.shape[0])
        bl_out_sample_error_rmse.append(bl_test_rmse)

    # Question 1 figure
    error_data = pd.DataFrame({'In sample RMSE': dt_in_sample_error_rmse,
                               'Out sample RMSE': dt_out_sample_error_rmse
                               }, index=range(1, 50))
    error_data.plot()
    plt.title("RMSE for Decision Tree vs leaf size")
    plt.ylabel("RMSE")
    plt.xlabel("Leaf size")
    plt.savefig("question_1")

    # Question 2 figure
    error_data = pd.DataFrame(
        {'In sample RMSE': bl_in_sample_error_rmse,
         'Out sample RMSE': bl_out_sample_error_rmse
         }, index=range(1, 50))

    error_data.plot()
    plt.title("RMSE for Bagging learner vs leaf size")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.savefig("question_2")

    # Question 3: mean absolute error
    # dt_in_sample_error_mean = []
    dt_out_sample_error_mean = []
    # rt_in_sample_error_mean = []
    rt_out_sample_error_mean = []
    for i in range(1, 50):

        # Question 3: Decision Tree
        dt_tree = dt.DTLearner(i)
        dt_tree.addEvidence(trainX, trainY)
        # dt_train_pred_y = dt_tree.query(trainX)
        # dt_train_mean_error = abs(testY - dt_train_pred_y).sum() / testY.shape[0]
        # dt_in_sample_error_mean.append(dt_train_mean_error)
        dt_test_pred_y = dt_tree.query(testX)
        dt_test_mean_error = abs(testY - dt_test_pred_y).sum()/testY.shape[0]
        dt_out_sample_error_mean.append(dt_test_mean_error)

        # Question 3: Random Tree
        rt_tree = rt.RTLearner(i)
        rt_tree.addEvidence(trainX, trainY)
        # rt_train_pred_y = rt_tree.query(trainX)
        # rt_train_mean_error = abs(testY - rt_train_pred_y).sum() / testY.shape[0]
        # rt_in_sample_error_mean.append(rt_train_rmse)
        rt_test_pred_y = rt_tree.query(testX)
        rt_test_rmse = abs(testY - rt_test_pred_y).sum() / testY.shape[0]
        rt_out_sample_error_mean.append(rt_test_rmse)


    error_data = pd.DataFrame(
        {'Decision Tree mean absolute error': dt_out_sample_error_mean,
         'Random Tree mean absolute error': rt_out_sample_error_mean
         }, index=range(1, 50))

    error_data.plot()
    plt.title("Mean absolute error for Decision Trees vs Random Trees")
    plt.xlabel("Leaf size")
    plt.ylabel("Mean Absolute Error")
    plt.savefig("question_3_mean_error")

    # Question 3: train time comparison
    dt_train_time = []
    rt_train_time = []
    for i in range(1, 50):
        # Question 3: Decision Tree train time
        dt_tree = dt.DTLearner(i)
        dt_start = time.time()
        dt_tree.addEvidence(trainX, trainY)
        dt_end = time.time()
        dt_train_lapse = 1000 * (dt_end - dt_start)
        dt_train_time.append(dt_train_lapse)

        # Question 3: Random Tree train time
        rt_tree = rt.RTLearner(i)
        rt_start = time.time()
        rt_tree.addEvidence(trainX, trainY)
        rt_end = time.time()
        rt_train_lapse = 1000 * (rt_end - rt_start)
        rt_train_time.append(rt_train_lapse)

    train_time_data = pd.DataFrame({'Decision Trees training time': dt_train_time,
                               'Random Trees training time': rt_train_time
                                }, range(1, 50))
    train_time_data.plot()
    plt.title("Training time for Decision Trees vs Random Trees")
    plt.xlabel("Leaf size")
    plt.ylabel("Training time (ms)")
    plt.savefig("question_3_training_time")
