"""MC1-P2: Optimize a portfolio.

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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as sciopt


def negative_sharpe_ratio(intial_allocs, prices_all_normalized, syms):
    temp = prices_all_normalized[syms] * intial_allocs
    temp = temp.sum(axis=1)
    daily_rets = (temp / temp.shift(1)) - 1
    # remove first entry and calculate the negative of sharpe's ratio.
    return -1 * np.sqrt(252) * daily_rets.iloc[1:].mean() / daily_rets.iloc[1:].std()


def calculate_cr(allocs, prices_all_normalized, syms):
    temp = prices_all_normalized[syms] * allocs
    return temp.sum(axis=1)


def calculate_adr(allocs, prices_all_normalized, syms):
    temp = prices_all_normalized[syms] * allocs
    temp = temp.sum(axis=1)
    daily_rets = (temp / temp.shift(1)) - 1
    return daily_rets.iloc[1:].mean()


def calculate_sddr(allocs, prices_all_normalized, syms):
    temp = prices_all_normalized[syms] * allocs
    temp = temp.sum(axis=1)
    daily_rets = (temp / temp.shift(1)) - 1
    return daily_rets.iloc[1:].std()


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = prices_all.ffill()
    prices_all = prices_all.bfill()
    prices_all_normalized = prices_all / prices_all.iloc[0]

    syms_size = len(syms)
    initial_allocs = np.full(syms_size, 1 / syms_size)

    bounds = [(0.0, 1.0)] * syms_size
    constraints = ({'type': 'eq', 'fun': lambda initial_allocs: 1.0 - np.sum(initial_allocs)})
    minimized_func = sciopt.minimize(negative_sharpe_ratio, initial_allocs,
                                     args=(prices_all_normalized, syms),
                                     method='SLSQP',
                                     bounds=bounds,
                                     constraints=constraints,
                                     options={'disp': False},
                                     )

    negative_sr = minimized_func.fun
    allocs = minimized_func.x

    cr = calculate_cr(allocs, prices_all_normalized, syms)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df = pd.concat([cr, prices_all_normalized['SPY']], keys=['Portfolio', 'SPY'], axis=1)
        df.plot()
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.title('Daily portfolio value and SPY')
        plt.savefig('figure')
        # plt.show()

    return allocs, \
           cr, \
           calculate_adr(allocs, prices_all_normalized, syms), \
           calculate_sddr(allocs, prices_all_normalized, syms), \
           -negative_sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
