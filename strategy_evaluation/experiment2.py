import pandas as pd
import matplotlib.pyplot as plt
import StrategyLearner as sl
import datetime as dt
from marketsimcode import compute_portvals

def author():
    return 'agaurav6'

def executeExperiment2():
    strategy_learner = sl.StrategyLearner(verbose=False, impact=0.0001)
    strategy_learner.addEvidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_low_impact = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_low_impact.index.names = ['Date']
    df_trades_low_impact.columns = ['Shares']
    df_trades_low_impact["Symbol"] = "JPM"
    df_trades_low_impact["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades_low_impact['Shares']]
    df_trades_low_impact.drop(df_trades_low_impact[df_trades_low_impact['Shares'] == 0].index, inplace=True)
    df_trades_low_impact['Shares'] = df_trades_low_impact['Shares'].abs()
    df_trades_low_impact = df_trades_low_impact.reindex(columns=['Symbol', 'Order', 'Shares'])
    df_trades_low_impact.to_csv('orders.csv')
    df_trades_low_impact = pd.read_csv('./orders.csv', index_col='Date', parse_dates=True, na_values=['nan'])
    sl_portvals_low_impact = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.0001)

    strategy_learner = sl.StrategyLearner(verbose=False, impact=0.001)
    strategy_learner.addEvidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                 sv=100000)
    df_trades_medium_impact = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_medium_impact.index.names = ['Date']
    df_trades_medium_impact.columns = ['Shares']
    df_trades_medium_impact["Symbol"] = "JPM"
    df_trades_medium_impact["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades_medium_impact['Shares']]
    df_trades_medium_impact.drop(df_trades_medium_impact[df_trades_medium_impact['Shares'] == 0].index, inplace=True)
    df_trades_medium_impact['Shares'] = df_trades_medium_impact['Shares'].abs()
    df_trades_medium_impact = df_trades_medium_impact.reindex(columns=['Symbol', 'Order', 'Shares'])
    df_trades_medium_impact.to_csv('orders.csv')
    df_trades_medium_impact = pd.read_csv('./orders.csv', index_col='Date', parse_dates=True, na_values=['nan'])
    sl_portvals_medium_impact = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.001)

    strategy_learner = sl.StrategyLearner(verbose=False, impact=0.01)
    strategy_learner.addEvidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_large_impact = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_large_impact.index.names = ['Date']
    df_trades_large_impact.columns = ['Shares']
    df_trades_large_impact["Symbol"] = "JPM"
    df_trades_large_impact["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades_large_impact['Shares']]
    df_trades_large_impact.drop(df_trades_large_impact[df_trades_large_impact['Shares'] == 0].index, inplace=True)
    df_trades_large_impact['Shares'] = df_trades_large_impact['Shares'].abs()
    df_trades_large_impact = df_trades_large_impact.reindex(columns=['Symbol', 'Order', 'Shares'])
    df_trades_large_impact.to_csv('orders.csv')
    df_trades_large_impact = pd.read_csv('./orders.csv', index_col='Date', parse_dates=True, na_values=['nan'])
    portvals_large_impact = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.01)

    sl_cumulative_return_low = sl_portvals_low_impact[-1] / sl_portvals_low_impact[0] - 1
    sl_cumulative_return_medium = sl_portvals_medium_impact[-1] / sl_portvals_medium_impact[0] - 1
    sl_cumulative_return_large = portvals_large_impact[-1] / portvals_large_impact[0] - 1

    sl_daily_returns_low = sl_portvals_low_impact / sl_portvals_low_impact.shift(1) - 1
    sl_daily_returns_low = sl_daily_returns_low[1:]

    sl_daily_returns_medium = sl_portvals_medium_impact / sl_portvals_medium_impact.shift(1) - 1
    sl_daily_returns_medium = sl_daily_returns_medium[1:]

    sl_daily_returns_large = portvals_large_impact / portvals_large_impact.shift(1) - 1
    sl_daily_returns_large = sl_daily_returns_large[1:]

    sl_portvals_norm_low = sl_portvals_low_impact / sl_portvals_low_impact[0]
    sl_portvals_norm_medium = sl_portvals_medium_impact / sl_portvals_medium_impact[0]
    sl_portvals_norm_large = portvals_large_impact / portvals_large_impact[0]

    sl_std_low = sl_daily_returns_low.std()
    sl_std_medium = sl_daily_returns_medium.std()
    sl_std_large = sl_daily_returns_large.std()

    sl_mean_low = sl_daily_returns_low.mean()
    sl_mean_medium = sl_daily_returns_medium.mean()
    sl_mean_large = sl_daily_returns_large.mean()

    plt.plot(sl_portvals_norm_low, color="blue")
    plt.plot(sl_portvals_norm_medium, color="green")
    plt.plot(sl_portvals_norm_large, color="red")

    plt.xlabel("Date")
    plt.xticks(rotation=30)
    plt.ylabel("Normalized portfolio value")
    plt.title("Experiment 2: Changing impact value")
    plt.legend(["Impact = 0.0001", "Impact = 0.001", "Impact = 0.01"])
    print("Cumulative return impact = 0.0001: ", sl_cumulative_return_low)
    print("Cumulative return impact = 0.001: ", sl_cumulative_return_medium)
    print("Cumulative return impact = 0.01: ", sl_cumulative_return_large)
    print("Std impact = 0.0001: ", sl_std_low)
    print("Std impact = 0.001: ", sl_std_medium)
    print("Std impact = 0.01: ", sl_std_large)
    print("Mean impact = 0.0001: ", sl_mean_low)
    print("Mean impact = 0.001: ", sl_mean_medium)
    print("Mean impact = 0.01: ", sl_mean_large)

    print("Trading events impact = 0.0001: ", (df_trades_low_impact != 0).sum()[0])
    print("Trading events impact = 0.001: ", (df_trades_medium_impact != 0).sum()[0])
    print("Trading events impact = 0.01: ", (df_trades_large_impact != 0).sum()[0])
    plt.savefig("experiment2.png")
    plt.close()


if __name__ == '__main__':
    executeExperiment2()
