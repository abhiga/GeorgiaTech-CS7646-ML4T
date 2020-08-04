import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
from marketsimcode import compute_portvals

class TheoreticallyOptimalStrategy(object):
    def __init__(self):
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):

        df_prices = get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
        df_prices_jpm = df_prices[[symbol]]

        df_trades = df_prices_jpm.copy()
        df_trades[symbol] = 0

        current_holding = 0
        for i in range(df_prices_jpm.shape[0] - 1):
            current_day_price = df_prices_jpm.iloc[i, 0]
            next_day_price = df_prices_jpm.iloc[i + 1, 0]
            difference_next_current = next_day_price - current_day_price

            sign = difference_next_current/abs(difference_next_current)

            if current_holding == 0:
                sv = sv - current_day_price * 1000 * sign
                df_trades.iloc[i, 0] = 1000 * sign
                current_holding = 1000 * sign
            elif current_holding == -1000 * sign:
                sv = sv - current_day_price * 2000 * sign
                df_trades.iloc[i, 0] = 2000 * sign
                current_holding = 1000 * sign

        df_trades.index.names = ['Date']
        df_trades.columns = ['Shares']
        df_trades["Symbol"] = "JPM"
        df_trades["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades['Shares']]
        df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)
        df_trades['Shares'] = df_trades['Shares'].abs()
        df_trades = df_trades.reindex(columns=['Symbol', 'Order', 'Shares'])
        return df_trades

    def author(self):
        return 'agaurav6'


def author():
    return 'agaurav6'

if __name__ == "__main__":
    ms = TheoreticallyOptimalStrategy()
    df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades.to_csv('orders1.csv')
    df_trades = pd.read_csv('./orders1.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()

    optimal_portvals = compute_portvals('./orders1.csv', start_val=100000, commission=0.0, impact=0.0)

    df_trades_bench = df_trades.copy()
    df_trades_bench.iloc[:] = 0
    df_trades_bench['Order'] = 'BUY'
    df_trades_bench['Symbol'] = 'JPM'
    df_trades_bench['Shares'].iloc[0] = 1000
    df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)
    df_trades_bench.to_csv('orders2.csv')
    bench_portvals = compute_portvals('./orders2.csv', start_val=100000, commission=0.0, impact=0.0)

    # daily returns
    bench_daily_returns = bench_portvals / bench_portvals.shift(1) - 1
    bench_daily_returns = bench_daily_returns[1:]
    optimal_daily_returns = optimal_portvals / optimal_portvals.shift(1) - 1
    optimal_daily_returns = optimal_daily_returns[1:]

    # cumulative returns
    bench_cumulative_returns = bench_portvals[-1] / bench_portvals[0] - 1
    optimal_cumulative_returns = optimal_portvals[-1] / optimal_portvals[0] - 1

    # normalized returns
    optimal_portvals_normalized = optimal_portvals / optimal_portvals[0]
    bench_portvals_normalized = bench_portvals / bench_portvals[0]

    # mean daily return
    bench_daily_mean = bench_daily_returns.mean()
    optimal_daily_mean = optimal_daily_returns.mean()

    # std daily return
    bench_std_dev = bench_daily_returns.std()
    optimal_std_dev = optimal_daily_returns.std()

    plt.figure(1)
    plt.plot(bench_portvals_normalized, color="green")
    plt.plot(optimal_portvals_normalized, color="red")
    plt.xlabel("date")
    plt.xticks(rotation=30)
    plt.ylabel("normalized portfolio value")
    plt.title("normalized theoretically optimal strategy vs benchmark portfolio")
    plt.legend(["benchmark portfolio", "theoretically optimal strategy portfolio"])

    print("mean benchmark ", bench_daily_mean)
    print("mean optimal ", optimal_daily_mean)
    print("cumulative return benchmark ", bench_cumulative_returns)
    print("cumulative return optimal ", optimal_cumulative_returns)
    print("std deviation benchmark ", bench_std_dev)
    print("std deviation optimal ", optimal_std_dev)

    plt.savefig("TheoreticallyOptimalStrategy.png")