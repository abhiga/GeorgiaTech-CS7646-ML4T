import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from util import get_data

def author():
    return 'agaurav6'

def get_volatility(df, window = 10):
    rolling_std = df.rolling(window=window, min_periods=window).std()
    return rolling_std

def get_momentum(df, window = 10):
    mmt = df / df.shift(window - 1) - 1
    return mmt

def get_cci(df, lookback = 10):
    rm = df.rolling(window=lookback, center=False).mean()
    return (df-rm)/(2.5 * df.std())

def get_sma(df, window = 10):
    sma = df.rolling(window=window, min_periods=window).mean()
    return sma

def get_ema(df, window=10):
    EMA_df = df.copy()
    EMA_df.iloc[:] = np.NaN
    EMA_df.iloc[window - 1] = df.iloc[0:window].mean()
    for i in range(window, df.index.size):
        EMA_df.iloc[i] = 2. / (1 + window) * (df.iloc[i] - EMA_df.iloc[i - 1]) + EMA_df.iloc[i - 1]

    return EMA_df, (df) / EMA_df

def get_sma_r(df, window = 10):
    sma = get_sma(df)
    sma_ratio = df/sma
    return sma_ratio

def get_bb_upper(df, window=10):
    sma = get_sma(df)
    rolling_std = get_volatility(df)
    boll_upper = sma + 2*rolling_std
    return boll_upper

def get_bb_lower(df, window=10):
    sma = get_sma(df)
    rolling_std = get_volatility(df)
    boll_lower = sma - 2*rolling_std
    return boll_lower

def get_bbp(df, window = 10):
    boll_upper = get_bb_upper(df)
    boll_lower = get_bb_lower(df)
    bbp = (df - boll_lower)/(boll_upper-boll_lower)
    return bbp

if __name__ == "__main__":

    dates = pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))

    df_prices = get_data(['JPM'], dates)
    df_prices.ffill().bfill()

    df_prices = df_prices / df_prices.iloc[0]
    df_prices_jpm = df_prices[['JPM']]
    mmt = get_momentum(df_prices_jpm)
    sma = get_sma(df_prices_jpm)
    smr = get_sma_r(df_prices_jpm)
    cci = get_cci(df_prices_jpm)

    bbp = get_bbp(df_prices_jpm)
    bb_upper = get_bb_upper(df_prices_jpm)
    bb_lower = get_bb_lower(df_prices_jpm)

    volatility = get_volatility(df_prices_jpm)

    ema_df, ema_index_df = get_ema(df_prices_jpm)
    ema_upper = ema_df * 1.05
    ema_lower = ema_df * 0.95

    plt.plot(df_prices_jpm)
    plt.plot(mmt)
    plt.xlabel('date')
    plt.xticks(rotation=30)
    plt.title('normalized price and momentum')
    plt.legend(['normalised price','momentum'])
    plt.savefig("momentum")

    plt.plot(df_prices_jpm)
    plt.plot(cci)
    plt.xlabel('date')
    plt.xticks(rotation=30)
    plt.title('normalized price and commodity channel index')
    plt.legend(['normalised price', 'commodity channel index'])
    plt.savefig("cci")

    plt.plot(df_prices_jpm)
    plt.plot(sma)
    plt.plot(smr)
    plt.xlabel('date')
    plt.title('normalized price and SMA')
    plt.legend(['normalised price','10-day SMA', 'price/SMA ratio'])
    plt.savefig("sma")

    plt.plot(df_prices_jpm)
    plt.plot(bbp)
    plt.plot(sma)
    plt.plot(bb_lower)
    plt.plot(bb_upper)
    plt.xlabel('date')
    plt.xticks(rotation=30)
    plt.title('normalized price and bollinger band indicator')
    plt.legend(['normalised prices', 'bollinger band percentage'])
    plt.savefig("bollinger bands")

    plt.plot(df_prices_jpm)
    plt.plot(volatility)
    plt.xlabel('date')
    plt.xticks(rotation=30)
    plt.title('normalized price volatility indicator')
    plt.legend(['normalised prices', 'volatility'])
    plt.savefig("volatility")


    plt.plot(ema_upper, label='price/EMA-over estimate')
    plt.plot(ema_lower, label='price/EMA-under estimate')
    plt.plot(ema_df, label='ema')
    plt.plot(df_prices_jpm, label='stock price')
    plt.xticks(rotation=30)
    plt.xlabel('date')
    plt.title('normalised price and price/EMA ')
    plt.legend(['normalised prices', '10-day EMA', 'price/EMA ratio'])
    plt.savefig('ema.png')
    plt.close()
