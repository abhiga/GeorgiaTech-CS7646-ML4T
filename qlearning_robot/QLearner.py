"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions=4, \
        alpha=0.2, \
        gamma=0.9, \
        rar=0.98, \
        radr=0.999, \
        dyna=0, \
        verbose=False):

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.q = np.zeros([num_states, num_actions])
        self.random_delta = 0.0001
        self.tc = np.zeros([num_states, num_states, num_actions]) + self.random_delta
        self.R = np.zeros([num_states, num_actions])

    def author(self):
        return 'agaurav6'

    def querysetstate(self, s):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Update the state without updating the Q-table  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param s: The new state  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns: The selected action  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        self.s = s
        action = rand.randint(0, self.num_actions - 1)
        if rand.random() >= self.rar:
            action = np.argmax(self.q[s])
        self.a = action
        self.rar = self.rar*self.radr
        return action

    def query(self,s_prime,r):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Update the Q table and return an action  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param s_prime: The new state  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param r: The reward  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns: The selected action  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        if self.dyna > 0:
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r
            self.tc[self.s, s_prime, self.a] += 1

        for i in range(self.dyna):
            s_d = rand.randint(0, self.num_states - 1)
            a_d = rand.randint(0, self.num_actions - 1)
            if (self.tc[s_d, :, a_d].sum() == self.random_delta * self.num_states):
                continue
            self.q[s_d, a_d] = (1 - self.alpha) * self.q[s_d, a_d] \
                               + self.alpha * (self.R[s_d, a_d] + self.gamma * self.q[np.argmax(self.tc[s_d, :, a_d])].max())

        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] \
                                 + self.alpha * (r + self.gamma * self.q[s_prime].max())

        return self.querysetstate(s_prime)

def author():
        return 'agaurav6'

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
    # learner = QLearner()
    # print(learner.author())
    # print(author())