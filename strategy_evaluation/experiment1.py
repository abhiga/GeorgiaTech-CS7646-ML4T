import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import StrategyLearner as sl
import ManualStrategy as ms
from marketsimcode import compute_portvals

def author():
    return 'agaurav6'

def executeExperiment1():
    slearner = sl.StrategyLearner(verbose=False, impact=0.005)
    slearner.addEvidence(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
    df_trades_sl = slearner.testPolicy(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
    df_trades_sl.index.names = ['Date']
    df_trades_sl.columns = ['Shares']
    df_trades_sl["Symbol"] = "JPM"
    df_trades_sl["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades_sl['Shares']]
    df_trades_sl.drop(df_trades_sl[df_trades_sl['Shares'] == 0].index, inplace=True)
    df_trades_sl['Shares'] = df_trades_sl['Shares'].abs()
    df_trades_sl = df_trades_sl.reindex(columns=['Symbol', 'Order', 'Shares'])
    df_trades_sl.to_csv('orders.csv')
    slearner_portvals = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.0)
    slearner_cumulative_returns = slearner_portvals[-1] / slearner_portvals[0] - 1
    slearner_daily_returns = slearner_portvals/slearner_portvals.shift(1) - 1
    slearner_daily_returns = slearner_daily_returns[1:]
    slearner_portvals_normalized = slearner_portvals/slearner_portvals[0]
    slearner_std_dev = slearner_daily_returns.std()
    slearner_mean = slearner_daily_returns.mean()

    ms_learner = ms.ManualStrategy()
    df_trades_ms = ms_learner.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_ms.to_csv('orders.csv')
    ms_portvals = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.0)
    ms_cumulative_returns = ms_portvals[-1]/ms_portvals[0] - 1
    ms_daily_returns = ms_portvals/ms_portvals.shift(1) - 1
    ms_daily_returns = ms_daily_returns[1:]
    ms_portvals_normalized = ms_portvals/ms_portvals[0]
    ms_std_dev = ms_daily_returns.std()
    ms_mean = ms_daily_returns.mean()
    print("Cumulative return for strategy learner: ", slearner_cumulative_returns)
    print("Cumulative return for manual strategy: ", ms_cumulative_returns)
    print("Mean for strategy learner: ", slearner_mean)
    print("Mean for manual strategy: ", ms_mean)
    print("Std dev for strategy learner: ", slearner_std_dev)
    print("Std dev for manual strategy: ", ms_std_dev)

    plt.plot(slearner_portvals_normalized, color="blue")
    plt.plot(ms_portvals_normalized, color="green")
    plt.xlabel("date")
    plt.xticks(rotation=30)
    plt.ylabel("normalized portfolio value")
    plt.title("experiment 1 - Strategy Learner vs Manual Strategy")
    plt.legend(["Strategy Learner portfolio", "Manual Strategy Portfolio"])
    plt.savefig("experiment1.png")
    plt.close()

if __name__ == '__main__':
    executeExperiment1()