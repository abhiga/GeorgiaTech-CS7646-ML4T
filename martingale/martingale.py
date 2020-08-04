"""Assess a betting strategy.
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
GT ID: 900897987 (replace with your GT ID)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""

import numpy as np
import matplotlib.pyplot as plt


def author():
    return 'agaurav6'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 903076714  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def each_winnings_experiment_1(win_prob, bets_count):
    winnings = np.zeros(bets_count)
    index = 0
    episode_winnings = 0

    # copy pasted the pseudo code from the given betting strategy with slight modification to capture winnings per bet.
    while episode_winnings < 80 and index < bets_count:
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            index = index + 1
            if won:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
            winnings[index] = episode_winnings

    winnings[index + 1:] = 80
    return winnings


def each_winnings_experiment_2(win_prob, bets_count=1000):
    bank_roll = 256
    winnings = np.zeros(bets_count)
    index = 0
    episode_winnings = 0

    # pseudo code modified based on the requirement of experiment 2.
    while (episode_winnings + bank_roll) > 0 and episode_winnings < 80 and index < bets_count-1:
        won = False
        bet_amount = 1
        while not won:
            # print(index)
            if index == bets_count-1 or episode_winnings+bank_roll <= 0:
                break
            won = get_spin_result(win_prob)
            index = index + 1
            if won:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = (episode_winnings + bank_roll) \
                    if (bet_amount * 2) > (episode_winnings + bank_roll) \
                    else (bet_amount * 2)
            winnings[index] = episode_winnings

    winnings[index + 1:] = winnings[index]
    return winnings


def generate_figure_1(win_prob, bets_count=1000, simulation_count=10):
    for i in range(simulation_count):
        plt.plot(each_winnings_experiment_1(win_prob, bets_count))
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.ylabel('Winning $')
    plt.xlabel('Bet')
    plt.title('Figure 1: Experiment1 simulator 10 times')
    plt.savefig('Figure 1')
    # plt.show()
    plt.close()


def generate_figure_2_and_3(win_prob, bets_count=1000, simulation_count=1000):
    winnings_list = []
    for i in range(simulation_count):
        winnings_list.append(each_winnings_experiment_1(win_prob, bets_count))

    winnings_list = np.asarray(winnings_list)
    mean_winnings_list = np.mean(winnings_list, axis=0)
    std_dev_winnings_list = np.std(winnings_list, axis=0)
    mean_plus_std_dev_winnings_list = mean_winnings_list + std_dev_winnings_list
    mean_minus_std_dev_winnings_list = mean_winnings_list - std_dev_winnings_list
    plt.plot(mean_winnings_list)
    plt.plot(mean_plus_std_dev_winnings_list)
    plt.plot(mean_minus_std_dev_winnings_list)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.ylabel('Winning $')
    plt.xlabel('Bet')
    plt.title('Figure 2: Experiment 1 simulator 1000 times, mean+std, mean, mean-std')
    plt.legend(['Mean', 'Mean + std dev', 'Mean - std dev'])
    plt.savefig('Figure 2')
    # plt.show()
    plt.close()

    median_winnings_list = np.median(winnings_list, axis=0)
    median_plus_std_dev_winnings_list = median_winnings_list + std_dev_winnings_list
    median_minus_std_dev_winnings_list = median_winnings_list - std_dev_winnings_list
    plt.plot(median_winnings_list)
    plt.plot(median_plus_std_dev_winnings_list)
    plt.plot(median_minus_std_dev_winnings_list)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.ylabel('Winning $')
    plt.xlabel('Bet')
    plt.title('Figure 3: Experiment 1 simulator 1000 times, median+std, median, median-std')
    plt.legend(['Median', 'Median + std dev', 'Median - std dev'])
    plt.savefig('Figure 3')
    # plt.show()
    plt.close()


def generate_figure_4_and_5(win_prob, bets_count=1000, simulation_count=1000):
    winnings_list = []
    for i in range(simulation_count):
        winnings_list.append(each_winnings_experiment_2(win_prob, bets_count))

    winnings_list = np.asarray(winnings_list)
    mean_winnings_list = np.mean(winnings_list, axis=0)
    std_dev_winnings_list = np.std(winnings_list, axis=0)
    mean_plus_std_dev_winnings_list = mean_winnings_list + std_dev_winnings_list
    mean_minus_std_dev_winnings_list = mean_winnings_list - std_dev_winnings_list
    plt.plot(mean_winnings_list)
    plt.plot(mean_plus_std_dev_winnings_list)
    plt.plot(mean_minus_std_dev_winnings_list)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.ylabel('Winning $')
    plt.xlabel('Bet')
    plt.title('Figure 4: Experiment 2 simulator 1000 times, mean+std, mean, mean-std')
    plt.legend(['Mean', 'Mean + std dev', 'Mean - std dev'])
    plt.savefig('Figure 4')
    # plt.show()
    plt.close()

    median_winnings_list = np.median(winnings_list, axis=0)
    median_plus_std_dev_winnings_list = median_winnings_list + std_dev_winnings_list
    median_minus_std_dev_winnings_list = median_winnings_list - std_dev_winnings_list
    plt.plot(median_winnings_list)
    plt.plot(median_plus_std_dev_winnings_list)
    plt.plot(median_minus_std_dev_winnings_list)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.ylabel('Winning $')
    plt.xlabel('Bet')
    plt.title('Figure 5: Experiment 2 simulator 1000 times, median+std, median, median-std')
    plt.legend(['Median', 'Median + std dev', 'Median - std dev'])
    plt.savefig('Figure 5')
    # plt.show()
    plt.close()


def test_code():
    win_prob = 18 / 38  # There are 18 black, 18 red, 1 '00' and 1 '0'.
    np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    # experiment 1
    # print(win_prob)
    generate_figure_1(win_prob)
    generate_figure_2_and_3(win_prob)

    # experiment 2
    generate_figure_4_and_5(win_prob)


if __name__ == "__main__":
    test_code()
