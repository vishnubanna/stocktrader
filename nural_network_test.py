import pandas as pd

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

import fix_yahoo_finance as yf

import tensorflow as tf
import tensorflow.keras as ks

yf.pdr_override()

# def get_data(ticker):
#   try:
#     print(ticker)
#     #start = dt.datetime(1950, 1, 1)
#     start = dt.datetime(2000, 1, 1)
#     #start = dt.datetime(2010, 1, 1)
#     end = dt.datetime.today() #- dt.timedelta(days = 1)
#     df = pdr.get_data_yahoo(ticker, start, end)
#
#     while df.empty:
#         print('here')
#         df = pdr.get_data_yahoo(ticker, start, end)
#
#     print(df.head())
#     print(df.tail())
#     return(df)
#   except:
#     pass
def getData2(ticker, tsfunc = 1, tlen = '1', FoC = 'full'):
    #FoC = full or compact
    apikey = '9OWLY1IEBT5CH6I8'
    intraday = '&interval='+tlen+'min&'
    if tsfunc == 1:
        tsfunc = 'TIME_SERIES_DAILY'
        url = 'https://www.alphavantage.co/query?function=' + tsfunc + '&symbol= '+ ticker +' &outputsize='+FoC+'&apikey=' + apikey +'&datatype=csv'
    elif tsfunc == 2:
        tsfunc = 'TIME_SERIES_INTRADAY'
        url = 'https://www.alphavantage.co/query?function=' + tsfunc + '&symbol= '+ ticker + intraday + ' &outputsize='+FoC+'&apikey=' + apikey +'&datatype=csv'


def get_data(ticker):
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

    print(df.head())
    print(df.tail())
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

def graphData(stock):
    df = stock
    df.reset_index(inplace = True)
    fig1 = plt.figure()
    ax1 = plt.subplot2grid((5,4),(0,0), rowspan = 4, colspan = 4)
    ax1.plot(df.Date, df.Open)
    ax1.plot(df.Date, df.High)
    ax1.plot(df.Date, df.Low)
    ax1.plot(df.Date, df.Close)
    ax1.grid(True)
    plt.ylabel('Stock Price')
    #ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    #for label in ax1.xaxis.get_ticklabels():
    #label.set_rotation(45)

    ax2 = plt.subplot2grid((5,4), (4,0), sharex = ax1, rowspan = 1, colspan = 4)
    ax2.bar(df.Date,df.Volume)
    ax2.grid(True)
    plt.ylabel('Volume')

    plt.xlabel('Date')
    plt.suptitle('Price v Time')
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.show()
    return

def getGraphSpecified(df):
    df.reset_index(inplace = True)
    print(df.head())
    fig1 = plt.figure()
    ax1 = plt.subplot2grid((5,4),(0,0), rowspan = 4, colspan = 4)
    #ax1.plot(df.Date, df.Open)
    ax1.plot(df.Date, df.m10MAvg)
    ax1.plot(df.Date, df.m20MAvg)
    ax1.plot(df.Date, df.m50MAvg)
    ax1.plot(df.Date, df.Close)
    ax1.plot(df.Date, df.volitlity)
    ax1.plot(df.Date, df['% change day'])

    ax1.grid(True)
    plt.ylabel('Stock Price')

    ax2 = plt.subplot2grid((5,4), (4,0), sharex = ax1, rowspan = 1, colspan = 4)
    ax2.bar(df.Date, df.Volume)
    ax2.plot(df.Date, df.VolumeMAvg)
    ax2.grid(True)
    plt.ylabel('Volume')
    plt.show()
    return

def getCandlestick(stock):
    df = stock
    dv = df[['Volume']]

    df.reset_index(inplace = True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    df['Date'] = df.Date.map(mdates.date2num)
    print(df.head())

    fig2 = plt.figure()
    ax1 = plt.subplot2grid((5,4),(0,0), rowspan = 4, colspan = 4)
    ax2 = plt.subplot2grid((5,4), (4,0), sharex = ax1, rowspan = 1, colspan = 4)
    ax1.xaxis_date()

    candlestick_ohlc(ax1, df.values, width = 0.75, colorup = 'g')
    #df['26Ma'] = df.Close.rolling(window = 26, min_periods = 0).mean()
    #ax2.fill_between(df.Date.map(mdates.date2num), dv.values, 0)
    #ax2 = plt.subplot2grid((5,4), (4,0), sharex = ax1, rowspan = 1, colspan = 4)
    ax2.bar(df.Date,dv.Volume)
    ax2.grid(True)
    plt.ylabel('Volume')

    plt.show()
    return

def timeSeries(df):
    df['f01'] = df['Close'].shift(-1)
    df['f02'] = df['Close'].shift(-2)
    df['f03'] = df['Close'].shift(-3)
    df['f04'] = df['Close'].shift(-4)
    df['f05'] = df['Close'].shift(-5)
    df['f06'] = df['Close'].shift(-6)
    df['f07'] = df['Close'].shift(-7)

    df = df[['Close', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07']]

    #print(df.head())
    #print(df.tail())
    df.dropna(inplace = True)

    return df

# def dataSpecify(df):
#
#     #print(df.head(), df.tail())
#     df['m50MAvg'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
#     df['m20MAvg'] = df['Close'].rolling(window = 20, min_periods = 0).mean()
#     df['m10MAvg'] = df['Close'].rolling(window = 10, min_periods = 0).mean()
#     df['VolumeMAvg'] = df['Volume'].rolling(window = 50, min_periods = 0).mean()
#     df['volitility'] = (df['Close']-df['m50MAvg'])/df['Close'] * 100
#
#     df['vshift'] = df['volitility'].shift(1)
#     df['volSlope'] = df['volitility'] - df['vshift']
#
#     df['50MAshift'] = df['m50MAvg'].shift(1)
#     df['50MASlope'] = (df['m50MAvg'] - df['50MAshift'])
#
#     df['% change day'] = (df['Open'] - df['Close']) / df['Open'] * 100
#     df['% daily volit'] = (df['High'] - df['Close']) / df['High'] * 100
#
#     df.fillna(0, inplace = True)
#
#     #df = df[['Close', '50 MAvg', 'Volume', 'volitility', '% daily volit','% change day' ,'VolumeMAvg']]
#     #df = df[['Close', '50 MAvg', 'Volume', 'volitility', '% daily volit','% change day' ,'VolumeMAvg']]
#     dg = df[['Close', 'Open', 'm50MAvg', 'm10MAvg', 'm20MAvg',  'volitility', '% daily volit','% change day', 'Volume','VolumeMAvg']]
#     df = df[['Close', 'Open', 'm50MAvg',  'volitility', '% daily volit','% change day', 'Volume','VolumeMAvg']]
#     print(df.head(), df.tail())
#
#     return df, dg

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

    df['vshift'] = df['volitility1'].shift(1)
    df['volSlope'] = df['volitility1'] - df['vshift']

    df['50MAshift'] = df['m50MAvg'].shift(1)
    df['50MASlope'] = (df['m50MAvg'] - df['50MAshift'])

    df['% change day'] = (df['Open'] - df['Close']) / df['Open'] * 100
    df['% daily volit'] = (df['High'] - df['Close']) / df['High'] * 100

    df['cShift'] = df['Close'].shift(1)
    df['%volit'] = (df['cShift'] - df['Close']) / df['Close'] * 100
    df.fillna(0, inplace = True)

    print(df['%volit'].head(15))
    #df = df[['Close', '50 MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', '%volit']]
    #df = df[['Close', '50 MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', '%volit']]
    dg = df[['Close', 'Open', 'm50MAvg', 'm10MAvg', 'm20MAvg',  'volitility1', '% daily volit','% change day', 'Volume','VolumeMAvg', '%volit']]
    #df = df[['Close', 'Open', 'm50MAvg', 'Volume', 'volitility1', '% daily volit','% change day' ,'VolumeMAvg', '%volit']]
    df = df[['Close', 'Open', 'e50MAvg', 'Volume', 'volitility1','% change day' ,'VolumeMAvg', 'MACD']]
    print(df.head(), df.tail())

    return df, dg

def certainty_testing(df):
    close = np.ndarray.flatten(df.VolumeMAvg.values)
    test = (df.Volume.values).flatten('F')[:len(close)]
    print(test, len(test))
    print(len(close))
    certainty = []
    for i in range(len(close)):
        volume = test[i]
        vavg = close[i]

        if volume > vavg:
            certainty.append(1)
        else:
            certainty.append(0)


    print(volume)
    print(test[len(close)])
    print(vavg)
    print(close[len(close)])
    print(certainty)
    return certainty

def volumeCertainty(df):
    vavg = np.ndarray.flatten(df.VolumeMAvg.values)
    print(vavg, len(vavg))
    length = len(vavg)
    vavgt = vavg[length-1]
    print(vavgt)
    vol = (df.Volume.values).flatten('F')[:length]

    volt = vol[length-1]
    print(volt)

    #volume = test[i]
    #vavg = close[i]

    if volt >= vavgt:
        return 1
    else:
        return 0

    #we need to take volitlity as an index of certaitnty
    #we need to take compare the price prediction with open to predict the trejectory


def train_test_sets(df, scaling = False, days = 1):
    #what we need to do
    #plan change so that all previous days are the train and test
    #then remove the close price as a feature, turn it into the label
    #then we use the data points for today, the entire row, and to predict the closing price
    # we do this pultple times per day:
    # see if it is more accurate
    # right now we are taking data fortoday and trying to predict tomorrow
    # could work, but i think it is less useful as, we could still use it though
    forecast_col = 'Close'
    df.fillna(-99999, inplace = True)
    forecast_out = days

    df['Label'] = df[forecast_col].shift(-forecast_out)
    #print(df.tail())

    #X = np.array(df.drop(['Label', 'Close'], axis = 1))
    #X = np.array(df.drop(['Label', '%volit'], axis = 1))
    X = np.array(df.drop(['Label'], axis = 1))
    if scaling:
        X = preprocessing.scale(X)
    X_lately = X[-forecast_out:] #forcastout = 30, (-)30 from the bottom
    X = X[:-forecast_out]

    df.dropna(inplace = True)
    y = np.array(df['Label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #print(X_train)
    return X_train, X_test, y_train, y_test, X_lately

def nuralNetOne(X_train, X_test, y_train, y_test, X_lately):
    layers = [
        ks.layers.InputLayer(input_shape = (len(X_train[1]), )),
        ks.layers.Dense(units = 64, activation = ks.activations.relu),
        ks.layers.Dense(units = 64, activation = ks.activations.relu),
        ks.layers.Dense(units = 1)
    ]
    model = ks.Sequential(layers)

    model.compile(optimizer = ks.optimizers.Adam(),
                loss = ks.losses.MeanSquaredError(),
                metrics = [ks.metrics.MeanAbsoluteError(), ks.metrics.MeanSquaredError()]
                )

    model.fit(x = X_train, y = y_train, batch_size = 20, epochs = 100)
    model.evaluate(x = X_test, y = y_test, batch_size = 1)

    prediction = model.predict(X_lately, verbose = 0)
    print(prediction)
    return(prediction)

def nuralNetTwo(X_train, X_test, y_train, y_test, X_lately):
    layers = [
        ks.layers.InputLayer(input_shape = (len(X_train[1]), )),
        ks.layers.Dense(units = 128, activation = ks.activations.relu),
        ks.layers.Dense(units = 64, activation = ks.activations.relu),
        ks.layers.Dense(units = 1)
    ]
    model = ks.Sequential(layers)

    model.compile(optimizer = ks.optimizers.Adam(),
                loss = ks.losses.MeanSquaredError(),
                metrics = [ks.metrics.MeanAbsoluteError(), ks.metrics.MeanSquaredError()]
                )

    model.fit(x = X_train, y = y_train, batch_size = 20, epochs = 100)
    model.evaluate(x = X_test, y = y_test, batch_size = 1)

    prediction = model.predict(X_lately, verbose = 0)
    print(prediction)
    return(prediction)

def predictionReduce(pred, pred1):
    return (pred + pred1)/2

def certainty():
    print('need to work on it')

def search():
    print('we pick compnaies that we update regularly, 500, we pull their prices for today hourly and search to see which company we well to or hop to. we \n need predict their stock price, and run a company certainty check to see if it is worth hopping to them, then we need to make sure we arent loosing money\n by hopping to them')

def companyCertainty():
    print('take the price data for a company, maybe the last 10 days or so and see if they are worth hopping to from the stock we are on right now')
    print('we need to train a ml on the company, then save those weights so that we can quickly search compnayies and repeatedly predict their prices.')
    print('model should be updated weekly')

def companylistUpdate():
    print('if we make this an app we can just have the user pick the companies they would like to invest them, then auto populate the rest of list based on those companies')

def sentimentAnalysis():
    print('mostly done on weekends to track the trejectory of the stock for the next week')

df = get_data('AAPL')
#getCandlestick(df)
data, gdata = dataSpecify(df)

# subp = mp.Process(target = getGraphSpecified, args = (gdata, ))
# subp.start()

c = volumeCertainty(data)
print(c)


#ts = timeSeries(df)
#certainty_t = certainty_testing(df)
#ct = pd.DataFrame(certainty_t)
#X_train, X_test, y_train, y_test, X_lately = train_test_sets(ts, scaling = False, days = 8)

X_train, X_test, y_train, y_test, X_lately = train_test_sets(data, scaling = True, days = 1)
pred = nuralNetOne(X_train, X_test, y_train, y_test, X_lately)
pred1 = nuralNetTwo(X_train, X_test, y_train, y_test, X_lately)
#model updated daily or every 2 days
pred = predictionReduce(pred, pred1)
print(pred)
