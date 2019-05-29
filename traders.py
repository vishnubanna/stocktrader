import pandas as pd
import requests

import time
import datetime as dt
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
#from mpl_finance import candlestick_ohlc

from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
from pandas_datareader import data as pdr
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import io

import fix_yahoo_finance as yf

def getData2(ticker, tsfunc = 1, tlen = '1', FoC = 'full'):
    #FoC = full or compact
    apikey = '9OWLY1IEBT5CH6I8'
    intraday = '&interval='+tlen+'min&'
    if tsfunc == 1:
        tssfunc = 'TIME_SERIES_DAILY'
        url = 'https://www.alphavantage.co/query?function=' + tssfunc + '&symbol='+ ticker +'&outputsize='+FoC+'&apikey=' + apikey +'&datatype=csv'
    elif tsfunc == 2:
        tssfunc = 'TIME_SERIES_INTRADAY'
        url = 'https://www.alphavantage.co/query?function=' + tssfunc + '&symbol='+ ticker + intraday + '&outputsize='+FoC+'&apikey=' + apikey +'&datatype=csv'

    csv = requests.get(url).content
    df = pd.read_csv(io.StringIO(csv.decode('utf-8')))
    df.rename(columns = {'timestamp':'Date', 'open':'Open', 'low':'Low', 'high':'High', 'close':'Close', 'volume':'Volume'}, inplace = True)
    df.set_index('Date', inplace = True)
    df = df.sort_index(ascending = True, axis = 0)
    return df


def getData(ticker, day = 20000):
  #we may have to change thit to get the data from more stable source. this will break if the ppl at yahoo change their site evena little
  try:
    print(ticker)
    #start = dt.datetime(1950, 1, 1)
    #start = dt.datetime(2000, 1, 1)
    #start = dt.datetime(2018, 1, 1)
    #start = dt.datetime(2018, 1, 1)
    start = dt.datetime.today() - dt.timedelta(days = day)
    end = dt.datetime.today() #- dt.timedelta(days = 1)
    #end = dt.datetime(2013, 10, 12)
    df = pdr.get_data_yahoo(ticker, start, end)

    while emptychek(df):
        df = pdr.get_data_yahoo(ticker, start, end)

    #print(df.head())
    #print(df.tail())
    return(df)
  except:
    pass

def emptychek(df):
    empty = True
    if empty:
        try:
            print(df.head())
            empty = False
        except:
            empty = True

    print(empty)
    return

def dataSpecify(df):
    #print(df.head(), df.tail())

    df['m50MAvg'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
    df['m20MAvg'] = df['Close'].rolling(window = 20, min_periods = 0).mean()
    df['m10MAvg'] = df['Close'].rolling(window = 10, min_periods = 0).mean()
    df['VolumeMAvg'] = df['Volume'].rolling(window = 50, min_periods = 0).mean()

    df['e50MAvg'] = df['Close'].ewm(span = 50, adjust = False, min_periods = 0).mean()
    df['e26MAvg'] = df['Close'].ewm(span = 26, adjust = False, min_periods = 0).mean()
    df['e12MAvg'] = df['Close'].ewm(span = 12, adjust = False, min_periods = 0).mean()
    #df['e10MAvg'] = df['Close'].ewm(span = 10, adjust = False, min_periods = 0).mean()
    #df['e3MAvg'] = df['Close'].ewm(span = 3, adjust = False, min_periods = 0).mean()

    df['MACD'] = df['e12MAvg'] - df['e26MAvg']

    df['MFlowMult'] = ((df.Close - df.Low) - (df.High - df.Close))/(df.High - df.Low)
    df['MFlowVol'] = (df['Volume'])*df['MFlowMult']
    #df['temp'] = df['MFlowVol'].shift(1)
    #df.fillna(0, inplace = True)
    df['ADL'] = df['MFlowVol'].values.cumsum()
    df['ADL'] = df['ADL'].shift(1)
    df.fillna(0, inplace = True)
    df['e50AAvg'] = df['ADL'].ewm(span = 50, adjust = False, min_periods = 0).mean()
    df['e10MAvg'] = df['ADL'].ewm(span = 10, adjust = False, min_periods = 0).mean()
    df['e3MAvg'] = df['ADL'].ewm(span = 3, adjust = False, min_periods = 0).mean()

    df['Chakin'] = df['e3MAvg'] - df['e10MAvg']
    df['cmoveA'] = df['Chakin'].ewm(span = 50, adjust = False, min_periods = 0).mean()
    df['cmoveA4'] = df['Chakin'].ewm(span = 100, adjust = False, min_periods = 0).mean()
    df['cmoveA3'] = df['Chakin'].ewm(span = 10, adjust = False, min_periods = 0).mean()
    df['cmoveA2'] = df['Chakin'].ewm(span = 26, adjust = False, min_periods = 0).mean() #increasing means uptend
    #ch moving avg positiv and (chakin positive, trending up still, negative gonna start trending down soon)
                                                      #ch moving avg p and chakin n = sell
    #ch moving avg negtive and (chakin positive, trending up , negative trending down still)
    #ch moving avg n and chakin p = buy
    # chakin slope is similar to chakin itslef

    df['MACDT'] = df['cmoveA3'] - df['cmoveA4'] # just a check means nothing for now
    #df['chakinslope'] = df['cmoveA2'].diff()
    #df.drop(['temp'], inplace = True)

    df['cmoveA_certinty'] = df['cmoveA'].apply(cmoveA_cert)
    df['MACD_certinty'] = df['MACD'].apply(MACD_cert)
    df['MACDT_certinty'] = df['MACDT'].apply(MACDT_cert)
    df['vol_certinty'] = df['Volume'] > df['VolumeMAvg']
    df['vol_certinty'] = df['vol_certinty'].apply(vol_cert)

    df['certainty'] = df['cmoveA_certinty']+df['MACD_certinty']+df['MACDT_certinty']+df['vol_certinty']

    print(df.head(), df.tail())
    return df

def getGraphSpecified(df):
    df.reset_index(inplace = True)
    #print(df.head())
    fig1 = plt.figure()
    ax1 = plt.subplot2grid((12,4),(0,0), rowspan = 4, colspan = 4)
    ax1.plot(df.Date, df.e26MAvg)
    ax1.plot(df.Date, df.e12MAvg)
    ax1.plot(df.Date, df.Close)

    ax1.grid(which='both', linestyle='-', linewidth='0.2', color='red')
    ax1.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylabel('Stock Price')

    ax2 = plt.subplot2grid((12,4), (4,0), sharex = ax1, rowspan = 2, colspan = 4)
    ax2.bar(df.Date, df.Volume)
    ax2.plot(df.Date, df.VolumeMAvg)
    ax2.plot(df.Date, df.ADL)
    ax2.plot(df.Date, df.e10MAvg)
    ax2.grid(which='both', linestyle='-', linewidth='0.2', color='red')
    ax2.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylabel('Volume + ADL')

    ax3 = plt.subplot2grid((12,4), (6,0), sharex = ax1, rowspan = 2, colspan = 4)
    #ax3.plot(df.Date, df.Chakin)
    #ax3.plot(df.Date, df.cmoveA)
    #ax3.plot(df.Date, df.cmoveA2)
    ax3.plot(df.Date, df.cmoveA3)
    ax3.plot(df.Date, df.cmoveA4)
    ax3.grid(which='both', linestyle='-', linewidth='0.2', color='red')
    ax3.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylabel('chakin')

    ax4 = plt.subplot2grid((12,4), (8,0), sharex = ax1, rowspan = 1, colspan = 4)
    ax4.bar(df.Date, df.MACD)
    ax4.grid(which='both', linestyle='-', linewidth='0.2', color='red')
    ax4.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
    plt.ylabel('MACD')

    ax5 = plt.subplot2grid((12,4), (9,0), sharex = ax1, rowspan = 1, colspan = 4)
    ax5.bar(df.Date, df.MACDT)
    #ax5.plot(df.Date, df.chakinslope)
    #ax5.grid(True)
    ax5.grid(which='both', linestyle='-', linewidth='0.2', color='red')
    ax5.grid(which='minor', linestyle='-', linewidth='0.2', color='black')
    plt.ylabel('MACDT')

    ax6 = plt.subplot2grid((12,4), (10,0), sharex = ax1, rowspan = 2, colspan = 4)
    ax6.plot(df.Date, df.certainty)
    ax6.grid(which='both', linestyle='-', linewidth='0.2', color='red')
    ax6.grid(which='minor', linestyle='-', linewidth='0.2', color='black')

    plt.show()
    return

#def slope():


def buy(money, price, stocks, buys):
    money = money - price
    stocks = stocks + 1
    buys = price
    return money, stocks, buys

def sell(money, price, stocks, sells, buy_price):
    tax = (price - buy_price) * 0.25
    money = money + price - tax
    stocks = stocks - 1
    return money, stocks, sells

def hold(holds):
    pass

def cmoveA_cert(num):
    if num > 0:
        num = 3
    else:
        num = 0
    return num

def MACD_cert(num):
    if num > 0:
        num = 4
    else:
        num = 0
    return num

def MACDT_cert(num):
    if num > 0:
        num = 1
    else:
        num = 0
    return num


def vol_cert(vol):
    if True:
        num = 1
    else:
        num = 0
    return num

def rise_indicator():
    #if certainty is low and it is falling dont buy
    #   possibley sell
    #if certainty is high and it is falling buy uncertain
    #   or hold, dont buy, and or short
    # MACDT is good indicator, we need to find something not used for certianty
    pass

def fall_indicator():
    pass

def cert(df):
    print('\n')
    print((df.certainty).values)
    print('\n')

def MACD_strat(df):
    macda = np.ndarray.flatten(np.array(df.MACD))
    chakina = np.ndarray.flatten(np.array(df.cmoveA3))
    macdtesta = np.ndarray.flatten(np.array(df.MACDT))
    closea = np.ndarray.flatten(np.array(df.Close))
    datea = np.ndarray.flatten(np.array(df.Date))

    funds = int(closea[0]) + 50

    money = funds
    stocks = 0
    bought = 0
    sells = 0
    holds = 0
    uncertain = 0
    buys = 0
    count = 0
    start = funds
    buy_price = funds

    length = len(closea)

    print(funds)

    for i in range(length):
        macd = macda[i]
        chakin = chakina[i]
        macdt = macdtesta[i]
        price = closea[i]
        date = datea[i]
        count = count + 1

        if macd >= 0 and stocks == 0:
            if chakin > 0 and macdt >= 0 and money >= price:
                print('buyat: money {}, condition {}, price {} , date {}, count {}\n\n'.format(money, macd, price, date, count))
                money, stocks, buy_price = buy(money, price, stocks, buy_price)
                buys = buys + 1
            else:
                holds = holds + 1
        elif macd < 0 and stocks == 1:
            if chakin < 0 and macdt < 0:
                print('sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, macd, price, date))
                money, stocks, sells = sell(money, price, stocks, sells, buy_price)
                sells = sells + 1
            else:
                holds = holds + 1
        else:
            holds = holds + 1

        if count == (length-1) and stocks == 1:
            money, stocks, sells = sell(money, price, stocks, sells, buy_price)
            print('--> forced sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, macd, price, date))

    print(length, money, stocks, sells, uncertain , holds, buys)
    print(count)
    profit = (money - start)/start * 100
    print('profit', profit)
    count = 0

    return

def MACD_basic(df):
    # this would require extra money to be in the account, and is risky, but done right it has alot of potential.
    macda = np.ndarray.flatten(np.array(df.MACD))
    chakina = np.ndarray.flatten(np.array(df.cmoveA))
    macdtesta = np.ndarray.flatten(np.array(df.MACDT))
    closea = np.ndarray.flatten(np.array(df.Close))
    datea = np.ndarray.flatten(np.array(df.Date))

    funds = int(closea[0]) + 50

    money = funds
    stocks = 0
    bought = 0
    sells = 0
    holds = 0
    uncertain = 0
    buys = 0
    count = 0
    start = funds
    buy_price = funds

    length = len(closea)
    for i in range(length):
        macd = macda[i]
        chakin = chakina[i]
        macdt = macdtesta[i]
        price = closea[i]
        date = datea[i]
        count = count + 1

        if macd >= 0 and money >= price and stocks == 0:
            #if chakin > 0 and macdt >= 0:
            print('buyat: money {}, condition {}, price {} , count {}\n\n'.format(money, macd, price, date))
            money, stocks, buy_price = buy(money, price, stocks, buy_price)
            buys = buys + 1
            #else:
                #holds = holds + 1
        elif macd < 0 and stocks == 1:
            #if chakin < 0 and macdt < 0:
            print('sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, macd, price, date))
            money, stocks, sells = sell(money, price, stocks, sells, buy_price)
            sells = sells + 1
            #else:
                #holds = holds + 1
        else:
            holds = holds + 1

        if count == (length-1) and stocks == 1:
            money, stocks, sells = sell(money, price, stocks, sells, buy_price)
            print('--> forced sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, macd, price, date))

    print(length, money, stocks, sells, uncertain , holds, buys)
    print(count)

    profit = (money - start)/start * 100
    print('profit', profit)
    count = 0
    return

def certainty_basic(df):
    #####++>>>>>>>>>>>>>> make a buy certainty and a sell certianty
    # th
    macda = np.ndarray.flatten(np.array(df.MACD_certinty))
    #chakina = np.ndarray.flatten(np.array(df.cmoveA))
    #macdtesta = np.ndarray.flatten(np.array(df.MACDT))
    certa = np.ndarray.flatten(np.array(df.certainty))
    closea = np.ndarray.flatten(np.array(df.Close))
    datea = np.ndarray.flatten(np.array(df.Date))

    funds = int(closea[0]) + 50

    money = funds
    stocks = 0
    bought = 0
    sells = 0
    holds = 0
    uncertain = 0
    uncertsells = 0
    buys = 0
    count = 0
    start = funds
    buy_price = funds

    print(funds)

    length = len(closea)
    for i in range(length):
        macd = macda[i]
        cert = certa[i]
        price = closea[i]
        date = datea[i]
        count = count + 1

        if macd == 4 and stocks == 0 and money >= price:
        #if cert < 4 and macd == 4 and stocks == 0 and money >= price:
            print('buyat: money {}, condition {}, price {} , date {}, count {}\n\n'.format(money, cert, price, date, count))
            money, stocks, buy_price = buy(money, price, stocks, buy_price)
            buys = buys + 1
        elif cert >= 8 and stocks == 1:
#             if (price >= buy_price):
#                 print('certain sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))
#                 money, stocks, sells = sell(money, price, stocks, sells, buy_price)
#                 sells = sells + 1
            holds = holds + 1
        elif cert >= 5 and stocks == 1:
            if (price > buy_price):
                print('certain sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))
                money, stocks, sells = sell(money, price, stocks, sells, buy_price)
                sells = sells + 1
        elif cert < 2 and stocks == 1:
            if (price < buy_price):
                print('uncertain sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))
                money, stocks, sells = sell(money, price, stocks, sells, buy_price)
                uncertsells = uncertsells + 1
        else:
            holds = holds + 1

        if count == (length) and stocks == 1:
            money, stocks, sells = sell(money, price, stocks, sells, buy_price)
            print('--> forced sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))

    print(length, money, stocks, sells, uncertain, uncertsells , holds, buys)
    print(count)
    profit = (money - start)/start * 100
    print('profit', profit)
    count = 0

    return
def certainty_complex(df):
    #####++>>>>>>>>>>>>>> make a buy certainty and a sell certianty
    # th
    macda = np.ndarray.flatten(np.array(df.MACD_certinty))
    #chakina = np.ndarray.flatten(np.array(df.cmoveA))
    #macdtesta = np.ndarray.flatten(np.array(df.MACDT))
    certa = np.ndarray.flatten(np.array(df.certainty))
    closea = np.ndarray.flatten(np.array(df.Close))
    datea = np.ndarray.flatten(np.array(df.Date))

    funds = int(closea[0]) + 50

    money = funds
    stocks = 0
    bought = 0
    sells = 0
    holds = 0
    uncertain = 0
    uncertsells = 0
    buys = 0
    count = 0
    start = funds
    buy_price = funds

    print(funds)

    length = len(closea)
    for i in range(length):
        macd = macda[i]
        cert = certa[i]
        price = closea[i]
        date = datea[i]
        count = count + 1

        #if macd == 0 and stocks == 0 and money >= price:
        #if cert < 4 and macd == 0 and stocks == 0 and money >= price:
        if cert < 4  and stocks == 0 and money >= price:
            print('buyat: money {}, condition {}, price {} , date {}, count {}\n\n'.format(money, cert, price, date, count))
            money, stocks, buy_price = buy(money, price, stocks, buy_price)
            buys = buys + 1
        elif cert >= 8 and stocks == 1:
#             if (price >= buy_price):
#                 print('certain sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))
#                 money, stocks, sells = sell(money, price, stocks, sells, buy_price)
#                 sells = sells + 1
            holds = holds + 1
        elif cert >=6 and stocks == 1:
            if (price > buy_price):
                print('certain sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))
                money, stocks, sells = sell(money, price, stocks, sells, buy_price)
                sells = sells + 1
        elif cert >= 3 and stocks == 1:
            if (price < buy_price):
                print('uncertain sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))
                money, stocks, sells = sell(money, price, stocks, sells, buy_price)
                uncertsells = uncertsells + 1
        else:
            holds = holds + 1

        if count == (length) and stocks == 1:
            money, stocks, sells = sell(money, price, stocks, sells, buy_price)
            print('--> forced sellat: money {}, condition {}, price {} , count {}\n\n'.format(money, cert, price, date))

    print(length, money, stocks, sells, uncertain, uncertsells , holds, buys)
    print(count)
    profit = (money - start)/start * 100
    print('profit', profit)
    count = 0

    return

df = getData('AAPL', day = 500)
#df = getData('TSLA', day = 500)
#df = getData('GOOG', day = 500)
#df = getData('^GSPC', day = 500)
#df = getData('NFLX', day = 500)
#df = getData('AMD', day = 500)
#df = getData('ROKU', day = 500)
#df = getData('^VIX', day = 500)
df = dataSpecify(df)
getGraphSpecified(df)
df.reset_index(inplace = True)
cert(df)

certainty_complex(df)
print('\n\n\n\n\n')

certainty_basic(df)

MACD_strat(df)
print('\n\n\n')
MACD_basic(df)
