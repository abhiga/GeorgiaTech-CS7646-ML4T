"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Abhijeet Gaurav (replace with your name)
GT User ID: agaurav6 (replace with your User ID)
GT ID: 903076714 (replace with your GT ID)
"""

import numpy as np


# this function should return a dataset (X and Y) that will work  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
# better for linear regression than decision trees  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def best4LinReg(seed=5):
    np.random.seed(seed)
    X = np.random.random(size=(150, 5)) * 10
    Y = X[:, 0] * 5 + X[:, 1] * 4 + X[:, 2] * 3 + X[:, 3] * 2 + X[:, 4] * 1
    return X, Y


def best4DT(seed=5):
    np.random.seed(seed)
    X = np.random.random(size=(150, 5)) * 10
    Y = X[:, 0] ** 5 + X[:, 1] ** 4 + X[:, 2] ** 3 + X[:, 3] ** 2 + X[:, 4] ** 1
    return X, Y


def author():
    return 'agaurav6'  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Abhijeet.")
    # print(best4LinReg(5))
    # print(best4DT(5))
