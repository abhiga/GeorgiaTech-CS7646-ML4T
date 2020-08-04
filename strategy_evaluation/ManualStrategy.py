import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data
from marketsimcode import compute_portvals
from indicators import get_sma_r, get_bbp, get_momentum

def author():
    return 'agaurav6'

class ManualStrategy(object):
    def __init__(self):
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 31), sv=100000):
        df_prices = get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
        df_prices_symbol = df_prices[[symbol]]

        momentum = get_momentum(df_prices_symbol)
        sma_r = get_sma_r(df_prices_symbol)
        bbp = get_bbp(df_prices_symbol)

        df_trades = df_prices_symbol.copy()
        df_trades[symbol] = 0

        impact = 0.005
        commission = 9.95
        current_holding = 0

        for i in range(df_prices_symbol.shape[0] - 1):
            price = df_prices_symbol.iloc[i, 0]
            sign = 0
            if (bbp.iloc[i, 0] <= 0.6 and sma_r.iloc[i, 0] < 1) or \
                    (sma_r.iloc[i, 0] < 1 and momentum.iloc[i, 0] > 0.1) or \
                    (bbp.iloc[i, 0] <= 0.6 and momentum.iloc[i, 0] > 0.1):
                sign = 1
                price = price * (1 + impact * sign)
            if (bbp.iloc[i, 0] >= 0.8 and sma_r.iloc[i, 0] > 1) or \
                    (sma_r.iloc[i, 0] > 1 and momentum.iloc[i, 0] < 0) or \
                    (bbp.iloc[i, 0] >= 0.8 and momentum.iloc[i, 0] < 0):
                sign = -1
                price = price * (1 + impact * sign)
            if sign != 0:
                if current_holding == 0:
                    sv -= commission
                    sv = sv - price * 1000 * sign
                    current_holding = 1000 * sign
                    df_trades.iloc[i, 0] = 1000 * sign
                elif current_holding == -1000 * sign:
                    sv -= commission
                    sv = sv - price * 2000 * sign
                    current_holding = 1000 * sign
                    df_trades.iloc[i, 0] = 2000 * sign
                else:
                    pass

        df_trades.index.names = ['Date']
        df_trades.columns = ['Shares']
        df_trades["Symbol"] = "JPM"
        df_trades["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades['Shares']]
        df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)
        df_trades['Shares'] = df_trades['Shares'].abs()
        df_trades = df_trades.reindex(columns=['Symbol', 'Order', 'Shares'])

        return df_trades

    def author(self):
        return 'aguarav6'

def executeManualStrategy():
    for i in [1, 2]:
        ms = ManualStrategy()
        if i == 1:
            print("In-sample performance")
            run_type = 'In-sample'
            df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
        elif i == 2:
            print("Out-sample performance")
            run_type = 'Out-sample'
            df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

        df_trades.to_csv('orders.csv')

        df_trades = pd.read_csv('./orders.csv', index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
        optimal_portvals = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.0)

        df_trades_benchmark = df_trades.copy()
        df_trades_benchmark.iloc[:] = 0
        df_trades_benchmark['Order'] = 'BUY'
        df_trades_benchmark['Symbol'] = 'JPM'
        df_trades_benchmark['Shares'].iloc[0] = 1000
        df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)
        df_trades_benchmark.to_csv('orders.csv')
        bm_portvals = compute_portvals('./orders.csv', start_val=100000, commission=0.0, impact=0.0)

        bm_cum_return = bm_portvals[-1] / bm_portvals[0] - 1
        op_cum_return = optimal_portvals[-1] / optimal_portvals[0] - 1

        bm_daily_returns = bm_portvals / bm_portvals.shift(1) - 1
        bm_daily_returns = bm_daily_returns[1:]

        opt_daily_returns = optimal_portvals / optimal_portvals.shift(1) - 1
        opt_daily_returns = opt_daily_returns[1:]

        optimal_portvals_norm = optimal_portvals / optimal_portvals[0]
        bm_portvals_norm = bm_portvals / bm_portvals[0]

        bm_std = bm_daily_returns.std()
        opt_std = opt_daily_returns.std()

        bm_mean = bm_daily_returns.mean()
        opt_mean = opt_daily_returns.mean()

        plt.plot(optimal_portvals_norm, color="red")
        plt.plot(bm_portvals_norm, color="green")
        for index in range(0, df_trades.shape[0]):
            if df_trades['Order'].iloc[index] == 'SELL':  # SHORT
                plt.axvline(df_trades.iloc[index].name, color='black')
            elif df_trades['Order'].iloc[index] == 'BUY':
                plt.axvline(df_trades.iloc[index].name, color='blue')
        plt.xlabel("Date")
        plt.xticks(rotation=30)
        plt.ylabel("Normalized portfolio value")
        plt.title(run_type + " normalized rule-based manual vs benchmark portfolio")
        plt.legend(["Manual Strategy Portfolio", "Benchmark portfolio"])
        print("")
        print("Cumulative return benchmark: ", bm_cum_return)
        print("Cumulative return manual: ", op_cum_return)
        print("Std benchmark: ", bm_std)
        print("Std manual: ", opt_std)
        print("Mean benchmark: ", bm_mean)
        print("Mean manual: ", opt_mean)
        print("")
        plt.savefig(run_type + ".png")
        plt.close()

if __name__ == "__main__":
    executeManualStrategy()