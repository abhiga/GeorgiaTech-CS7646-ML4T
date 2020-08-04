import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand	
import indicators as idc	  
import math	 	
import QLearner as ql	  
import marketsimcode as ms		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):

    def get_gtid(self):
        return 903076714

    def __init__(self, verbose = False, impact=0.0):
        self.momentum = 'momentum'
        self.sma_ratio = 'sma_ratio'
        self.bollinger_percent = 'bollinger_percent'
        self.impact = impact
        self.window = 10
        self.max_runs = 50
        rand.seed(self.get_gtid())
        self.threshold_momentum = []
        self.threshold_sma_ratio = []
        self.threshold_bollinger_percent = []

        self.learner = ql.QLearner(num_states=1000,num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)

    def serialize(self, indicators):
        serialized_state = 0
        for i in range(0, len(self.threshold_bollinger_percent)):
            if indicators[2] > self.threshold_bollinger_percent[i]:
                continue
            else:
                serialized_state = serialized_state + i*2
                break
        for i in range(0, len(self.threshold_sma_ratio)):
            if indicators[1] > self.threshold_sma_ratio[i]:
                continue
            else:
                serialized_state = serialized_state + i*10
                break
        for i in range(0, len(self.threshold_momentum)):
            if indicators[0] > self.threshold_momentum[i]:
                continue
            else:
                serialized_state = serialized_state + i*50
                break
        return serialized_state

    def add_indicator_details(self, indicator_details):
        momentum_values = indicator_details[self.momentum].sort_values()
        momentum_values = momentum_values.dropna()
        for i in range(self.window):
            self.threshold_momentum.append(momentum_values[(i + 1) * math.floor(momentum_values.shape[0]/self.window)])

        sma_values = indicator_details[self.sma_ratio].sort_values()
        sma_values = sma_values.dropna()
        for i in range(0,self.window):
            self.threshold_sma_ratio.append(sma_values[(i + 1) * math.floor(sma_values.shape[0]/self.window)])

        bollinger_percent_values = indicator_details[self.bollinger_percent].sort_values()
        bollinger_percent_values = bollinger_percent_values.dropna()
        for i in range(0,self.window):
            self.threshold_bollinger_percent.append(bollinger_percent_values[(i + 1) * math.floor(bollinger_percent_values.shape[0]/self.window)])

    def addEvidence(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):

        df_prices = ut.get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
        df_prices = df_prices[[symbol]]

        indicator_details = df_prices.copy()
        indicator_details.drop([symbol], axis=1, inplace=True)
        indicator_details[self.momentum] = idc.get_momentum(df_prices)
        indicator_details[self.sma_ratio] = idc.get_sma_r(df_prices)
        indicator_details[self.bollinger_percent] = idc.get_bbp(df_prices)

        self.add_indicator_details(indicator_details)

        indicators = indicator_details.iloc[self.window - 1]
        s = self.serialize(indicators)
        a = self.learner.querysetstate(s)
        df_prices_len = df_prices.shape[0]

        ith_run = 0
        run_results = []

        while ith_run < self.max_runs:
            if len(run_results) > 10 and round(run_results[-1],4) == round(run_results[-2],4):
                break
            ith_run = ith_run + 1
            holding = 0
            df_trades = df_prices.copy()
            df_trades[symbol] = 0

            for i in range(self.window, df_prices_len - 1):
                reward = 0
                s = self.serialize(indicator_details.iloc[i])
                current_price = df_prices.iloc[i, 0]
                next_price = df_prices.iloc[i+1, 0]
                if rand.uniform(0.0, 1.0) <= 0:
                    a = rand.randint(0, 2)
                if a == 1:
                    pass
                elif a == 2 and holding == 0:
                    holding = 1000
                    df_trades.iloc[i, 0] = holding
                    reward = - self.impact * (current_price+next_price) * holding
                elif a == 2 and holding == -1000:
                    holding = 1000
                    df_trades.iloc[i, 0] = 2 * holding
                    reward = - self.impact*(current_price+next_price) * 2 * holding
                elif a == 0 and holding == 0:
                    holding = -1000
                    df_trades.iloc[i, 0] = holding
                    reward = - self.impact*(current_price+next_price) * 1000

                elif a == 0 and holding == 1000:
                    holding = -1000
                    df_trades.iloc[i, 0] = 2 * holding
                    reward = - self.impact*(current_price+next_price) * 2000

                reward += (next_price - current_price) * holding
                a = self.learner.query(s, reward)

            df_trades.index.names = ['Date']
            df_trades.columns = ['Shares']
            df_trades["Symbol"] = "JPM"
            df_trades["Order"] = ["BUY" if x > 0 else "SELL" if x < 0 else "NA" for x in df_trades['Shares']]
            df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)
            df_trades['Shares'] = df_trades['Shares'].abs()
            df_trades = df_trades.reindex(columns=['Symbol', 'Order', 'Shares'])
            df_trades.to_csv('orders.csv')

            port_values = ms.compute_portvals('./orders.csv', start_val=sv, commission=0.0, impact=self.impact)
            cumulative_return = port_values[-1] / port_values[0] - 1
            run_results.append(cumulative_return)
  		   	  			  	 		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):

        df_prices = ut.get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
        df_prices = df_prices[[symbol]]

        # Make indicators df
        indicator_details = df_prices.copy()
        indicator_details.drop([symbol],axis=1,inplace=True)
        indicator_details[self.momentum] = idc.get_momentum(df_prices)
        indicator_details[self.sma_ratio] = idc.get_sma_r(df_prices)
        indicator_details[self.bollinger_percent] = idc.get_bbp(df_prices)

        df_trades = df_prices.copy()
        df_trades[symbol] = 0
        current_holding = 0

        indicators = indicator_details.iloc[self.window - 1]
        s = self.serialize(indicators)
        a = self.learner.querysetstate(s)
        prices_len = df_prices.shape[0]

        for i in range(self.window, prices_len - 1):
            s = self.serialize(indicator_details.iloc[i])
            if a == 1:
                pass
            elif a == 0 and current_holding == 0:
                current_holding = -1000
                df_trades.iloc[i, 0] = current_holding
            elif a == 0 and current_holding == 1000:
                current_holding = -1000
                df_trades.iloc[i, 0] = current_holding * 2
            elif a == 2 and current_holding == 0:
                current_holding = 1000
                df_trades.iloc[i, 0] = current_holding
            elif a == 2 and current_holding == -1000:
                current_holding = 1000
                df_trades.iloc[i, 0] = current_holding * 2

            a = self.learner.querysetstate(s)

        return df_trades

    def author(self):
        return 'aguarav6'

def author():
    return 'aguarav6'
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")