"""MC2-P1: Market simulator.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def author():
    return 'agaurav6'  # replace tb34 with your Georgia Tech username


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # code should work correctly with either input  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # TODO: Your code here

    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan']).sort_index()

    # Filter orders by removing dates which are outside SPY trading window
    df_orders = filter_on_spy_trading_dates(df_orders)

    index_date_range = get_index_date_range(df_orders)

    symbol_list = get_unique_symbol_list(df_orders)

    df_prices = get_prices(symbol_list, index_date_range)

    df_trades = get_trades(symbol_list, index_date_range, df_orders, df_prices, impact, commission)

    df_holdings = get_holdings(symbol_list, index_date_range, df_trades, start_val)

    df_values = get_values(df_holdings, df_prices)

    df_port_values = df_values.sum(axis=1)
    return df_port_values


def get_index_date_range(df_orders):
    return pd.date_range(df_orders.index[0], df_orders.index[-1])

def get_unique_symbol_list(df_orders):
    return list(df_orders['Symbol'].unique())

def get_prices(symbol_list, index_date_range):
    # Process to make the df_prices table as mentioned in the https://www.youtube.com/watch?v=1ysZptg2Ypk
    df_prices = get_data(symbol_list, index_date_range, addSPY=True, colname='Adj Close')

    # Add column Cash
    df_prices['Cash'] = 1.0

    # remove SPY, forward fill and backward fill
    df_prices = df_prices[symbol_list + ['Cash']].ffill().bfill()
    return df_prices

def get_trades(symbol_list, index_date_range, df_orders, df_prices, impact, commission):
    # Process to make the df_trades table as mentioned in the https://www.youtube.com/watch?v=1ysZptg2Ypk
    df_trades = pd.DataFrame(columns=symbol_list + ['Cash'], index=index_date_range)
    df_trades = df_trades.fillna(0.0)

    for index in range(len(df_orders)):
        date_index = df_orders.index[index]
        order = df_orders['Order'].iloc[index]

        if order == 'SELL':
            df_trades.ix[date_index]['Cash'] = df_trades.ix[date_index]['Cash'] - commission
            symbol = df_orders['Symbol'].iloc[index]
            price = df_prices.ix[date_index][symbol]
            shares = df_orders['Shares'].iloc[index]
            df_trades.ix[date_index]['Cash'] = df_trades.ix[date_index]['Cash'] + shares * price * (1 - impact)
            df_trades.ix[date_index][symbol] = df_trades.ix[date_index][symbol] - shares

        elif order == 'BUY':
            df_trades.ix[date_index]['Cash'] = df_trades.ix[date_index]['Cash'] - commission
            symbol = df_orders['Symbol'].iloc[index]
            price = df_prices.ix[date_index][symbol]
            shares = df_orders['Shares'].iloc[index]
            df_trades.ix[date_index]['Cash'] = df_trades.ix[date_index]['Cash'] - shares * price * (1 + impact)
            df_trades.ix[date_index][symbol] = df_trades.ix[date_index][symbol] + shares

    return df_trades

def get_holdings(symbol_list, index_date_range, df_trades, start_val):
    # Process to make the df_holdings table as mentioned in the https://www.youtube.com/watch?v=1ysZptg2Ypk
    df_holdings = pd.DataFrame(columns=symbol_list + ['Cash'], index=index_date_range)
    df_holdings.ix[:, :] = 0
    df_holdings.ix[0, :] = df_trades.ix[0, :]
    df_holdings.ix[0, 'Cash'] = df_holdings.ix[0, 'Cash'] + start_val

    for i in range(1, len(df_holdings)):
        df_holdings.ix[i] = df_holdings.ix[i - 1] + df_trades.ix[i]

    return df_holdings

def get_values(df_holdings, df_prices):
    df_values = df_holdings * df_prices
    return df_values.dropna()

def filter_on_spy_trading_dates(df_orders):
    # Filter orders by removing dates which are outside SPY trading window
    df_prices_spy = get_data(['SPY'], get_index_date_range(df_orders), addSPY=False, colname='Adj Close')
    df_prices_spy = df_prices_spy.dropna()
    order_dates_valid_list = []
    for dateIndex in range(df_orders.shape[0]):
        if df_orders.iloc[dateIndex].name not in df_prices_spy.index:
            pass
        else:
            order_dates_valid_list.append(dateIndex)
    return df_orders.iloc[order_dates_valid_list, :]


def test_code():
    # this is a helper function you can use to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # note that during autograding his function will not be called.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[
            0]]  # just get the first column  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    else:
        "warning, code did not return a DataFrame"

        # Get portfolio stats  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    # print (author())
    test_code()
