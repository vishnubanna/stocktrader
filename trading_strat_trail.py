import pandas as pd
import requests

import time
import datetime as dt
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc

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


def getData(ticker):
  #we may have to change thit to get the data from more stable source. this will break if the ppl at yahoo change their site evena little
  try:
    print(ticker)
    #start = dt.datetime(1950, 1, 1)
    start = dt.datetime(2000, 1, 1)
    #start = dt.datetime(2010, 1, 1)
    end = dt.datetime.today() #- dt.timedelta(days = 1)
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
    print(df.head(), df.tail())

    df['m50MAvg'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
    df['m20MAvg'] = df['Close'].rolling(window = 20, min_periods = 0).mean()
    df['m10MAvg'] = df['Close'].rolling(window = 10, min_periods = 0).mean()
    df['VolumeMAvg'] = df['Volume'].rolling(window = 50, min_periods = 0).mean()
    df['volitility1'] = (df['Close']-df['m50MAvg'])/df['Close'] * 100

    df['e50MAvg'] = df['Close'].ewm(span = 50, adjust = False, min_periods = 0).mean()
    df['e26MAvg'] = df['Close'].ewm(span = 26, adjust = False, min_periods = 0).mean()
    df['e12MAvg'] = df['Close'].ewm(span = 12, adjust = False, min_periods = 0).mean()

    df['MACD'] = df['e12MAvg'] - df['e26MAvg']

    df['50MAshift'] = df['e50MAvg'].shift(1)
    df['e50MASlope'] = (df['e50MAvg'] - df['50MAshift'])

    df['% change day'] = (df['Open'] - df['Close']) / df['Open'] * 100
    df['% daily volit'] = (df['High'] - df['Close']) / df['High'] * 100

    df['cPrev'] = df['Close'].shift(1)
    df['%volit'] = (df['cPrev'] - df['Close']) / df['Close'] * 100
    df.fillna(0, inplace = True)

    print(df['%volit'].head(15))
    #df = df[['Close', '50 MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', '%volit']]
    #df = df[['Close', '50 MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', '%volit']]
    #dg = df[['Close', 'Open', 'm50MAvg', 'm10MAvg', 'm20MAvg',  'volitility1', '% daily volit','% change day', 'Volume','VolumeMAvg', '%volit']]
    #df = df[['Close', 'Open', 'm50MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', '%volit']]
    df = df[['Close', 'Open', 'm50MAvg', 'e50MAvg', 'e26MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', 'cPrev', 'MACD', 'e50MASlope']]
    print(df.head(), df.tail())
    return df

def getGraphSpecified(df):
    df.reset_index(inplace = True)
    print(df.head())
    fig1 = plt.figure()
    ax1 = plt.subplot2grid((8,4),(0,0), rowspan = 4, colspan = 4)
    #ax1.plot(df.Date, df.Open)
    #ax1.plot(df.Date, df.m10MAvg)
    #ax1.plot(df.Date, df.m20MAvg)
    #ax1.plot(df.Date, df.m50MAvg)
    ax1.plot(df.Date, df.e50MAvg)
    ax1.plot(df.Date, df.e26MAvg)
    ax1.plot(df.Date, df.Close)
    #ax1.plot(df.Date, df.volitility1)
    #ax1.plot(df.Date, df['% change day'])

    ax1.grid(True)
    plt.ylabel('Stock Price')

    ax2 = plt.subplot2grid((8,4), (4,0), sharex = ax1, rowspan = 2, colspan = 4)
    ax2.bar(df.Date, df.Volume)
    ax2.plot(df.Date, df.VolumeMAvg)
    #ax2.plot(df.Date, df.e50MAvg)
    ax2.grid(True)

    ax3 = plt.subplot2grid((8,4), (6,0), sharex = ax1, rowspan = 2, colspan = 4)
    ax3.plot(df.Date, df.MACD)
    ax3.plot(df.Date, df.e50MASlope)
    ax3.grid(True)

    plt.ylabel('Volume')
    plt.show()
    return


df = getData('AAPL')
df = dataSpecify(df)
getGraphSpecified(df)
