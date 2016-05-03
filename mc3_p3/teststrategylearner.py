
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import StrategyLearner as sl
import pandas as pd
import datetime as dt
import pandas as pd
import math
import numpy as np
from timeit import default_timer as timer
from marketsim import compute_portvals


def convertTradeFileToOldFormat(df_trades, symbol):
    df_trades_old_format = pd.DataFrame(df_trades.loc[df_trades['Shares'] != 0, 'Shares'])
    df_trades_old_format['Symbol'] = symbol
    df_trades_old_format['Order'] = 'BUY'
    df_trades_old_format.loc[df_trades_old_format['Shares'] < 0, 'Order'] = 'SELL'
    df_trades_old_format['Shares'] = abs(df_trades_old_format['Shares'])
    return df_trades_old_format
    
def run():
    #symbol = 'ML4T-220'
    #symbol = 'SPY'
    symbol = 'IBM'
    symbols = ['IBM', 'ML4T-220']
    
    for symbol in symbols:
        start1 = timer()
        
        learner = sl.StrategyLearner(verbose = False) # constructor
        start = timer()
        learner.addEvidence(symbol=symbol, sd=dt.datetime(2007,12,31), ed=dt.datetime(2009,12,31), sv = 10000) # training step
        end = timer()
        print('addEvidence took:', str(end - start), ' seconds')
        
        start = timer()
        df_trades_train = learner.testPolicy(symbol=symbol, sd=dt.datetime(2007,12,31), ed=dt.datetime(2009,12,31), sv = 10000) # testing step
        end = timer()
        print('testPolicy took:', str(end - start), ' seconds')
        
        start = timer()
        df_trades_test = learner.testPolicy(symbol=symbol, sd=dt.datetime(2009,12,31), ed=dt.datetime(2011,12,31), sv = 10000) # testing step
        end = timer()
        print('testPolicy took:', str(end - start), ' seconds')
            
        end1 = timer()
        dur = end1 - start1
        print ('Total time: ', str(dur))
        
        #### RUBIC
        #### Training and testing for each situation should run in less than 30 seconds. We reserve the right to use different time periods if necessary to reduce auto grading time.
        if dur > 30.:
            print('FAILED TOTAL TIME CHECK! EXCEEDED 30 SECONDS. T=', dur)
        else:
            print('Passed time check: ', dur)
        
        
        ##
        ## Test trade file
        ##
        if isinstance(df_trades_test, pd.DataFrame) == False:
            print "Returned result is not a DataFrame"
        #if prices.shape != df_trades_new1.shape:
        #    print "Returned result is not the right shape"
        tradecheck = abs(df_trades_test.cumsum()).values
        tradecheck[tradecheck<=100] = 0
        tradecheck[tradecheck>0] = 1
        if tradecheck.sum(axis=0) > 0:
            print "Returned result violoates holding restrictions (more than 100 shares)"
    
        df_trades_old_format = convertTradeFileToOldFormat(df_trades_test, symbol)
        port_val_test = compute_portvals(orders_file="", df_orders=df_trades_old_format, 
                        start_val=10000, check_leverage=False, 
                        start_date=None, end_date=None)
        
        
        ##
        ## Create trade file for buy and hold during test period
        ##
        df_trades_test_buy_and_hold = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
        df_trades_test_buy_and_hold = df_trades_test_buy_and_hold.append({'Date': port_val_test.index[0], 'Symbol': symbol, 'Order' : 'BUY', 'Shares' : 100}, ignore_index=True)
        df_trades_test_buy_and_hold = df_trades_test_buy_and_hold.append({'Date': port_val_test.index[-1], 'Symbol': symbol, 'Order' : 'SELL', 'Shares' : 100}, ignore_index=True)
        
        df_trades_test_buy_and_hold = df_trades_test_buy_and_hold.set_index(['Date'])
        port_val_test_buy_and_hold = compute_portvals(orders_file="", df_orders=df_trades_test_buy_and_hold, 
                        start_val=10000, check_leverage=False, 
                        start_date=None, end_date=None, 
                        max_shares=100)
        
        #### RUBIC
        #### For ML4T-220, the trained policy should significantly outperform the benchmark out of sample (7 points)
        if symbol == 'ML4T-220':
            pv_benchmark = port_val_test_buy_and_hold.values[-1][0]
            pv_test = port_val_test.values[-1][0]
            if pv_test < 2. * pv_benchmark:
                print('FAILED IN SAMPLE PERFORMANCE CHECK. B=', pv_benchmark, 'T=', pv_test, 'SYM=', symbol)
            else:
                print('Passed test performance test')
                
            plt.clf()
            plt.cla()
            plt.plot(pd.concat([port_val_test_buy_and_hold, port_val_test], axis=1))
            plt.savefig(symbol + 'test_fig.png')
        
        
        ##
        ## Train trade file
        ##
        if isinstance(df_trades_train, pd.DataFrame) == False:
            print "Returned result is not a DataFrame"
        #if prices.shape != df_trades_new1.shape:
        #    print "Returned result is not the right shape"
        tradecheck = abs(df_trades_train.cumsum()).values
        tradecheck[tradecheck<=100] = 0
        tradecheck[tradecheck>0] = 1
        if tradecheck.sum(axis=0) > 0:
            print "Returned result violoates holding restrictions (more than 100 shares)"
        
        df_trades_old_format = convertTradeFileToOldFormat(df_trades_train, symbol)
        port_val_train = compute_portvals(orders_file="", df_orders=df_trades_old_format, 
                        start_val=10000, check_leverage=False, 
                        start_date=None, end_date=None)
        
        
        ##
        ## Create trade file for buy and hold during train period
        ##
        df_trades_train_buy_and_hold = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
        df_trades_train_buy_and_hold = df_trades_train_buy_and_hold.append({'Date': port_val_train.index[0], 'Symbol': symbol, 'Order' : 'BUY', 'Shares' : 100}, ignore_index=True)
        df_trades_train_buy_and_hold = df_trades_train_buy_and_hold.append({'Date': port_val_train.index[-1], 'Symbol': symbol, 'Order' : 'SELL', 'Shares' : 100}, ignore_index=True)
        
        df_trades_train_buy_and_hold = df_trades_train_buy_and_hold.set_index(['Date'])
        port_val_train_buy_and_hold = compute_portvals(orders_file="", df_orders=df_trades_train_buy_and_hold, 
                        start_val=10000, check_leverage=False, 
                        start_date=None, end_date=None, 
                        max_shares=100)
        
        #### RUBIC
        #### For ML4T-220, the trained policy should significantly outperform the benchmark in sample (7 points)
        #### For IBM, the trained policy should significantly outperform the benchmark in sample (7 points)
        pv_benchmark = port_val_train_buy_and_hold.values[-1][0]
        pv_train = port_val_train.values[-1][0]
        if pv_train < 2. * pv_benchmark:
            print('FAILED IN SAMPLE PERFORMANCE CHECK. B=', pv_benchmark, 'T=', pv_train, 'SYM=', symbol)
        else:
            print('Passed train performance test')
        
        plt.clf()
        plt.cla()
        plt.plot(pd.concat([port_val_train_buy_and_hold, port_val_train], axis=1))
        plt.savefig(symbol + 'train_fig.png')
        

# run the code to test a learner
if __name__=="__main__":
    run()
    
    





