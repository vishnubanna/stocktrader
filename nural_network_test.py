import pandas as pd

import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

import tensorflow as tf
import tensorflow.keras as ks

yf.pdr_override()

def get_data(ticker):
  try:
    print(ticker)
    #start = dt.datetime(1950, 1, 1)
    start = dt.datetime(2000, 1, 1)
    #start = dt.datetime(2010, 1, 1)
    end = dt.datetime.today() #- dt.timedelta(days = 1)
    df = pdr.get_data_yahoo(ticker, start, end)

    while df.empty:
        print('here')
        df = pdr.get_data_yahoo(ticker, start, end)

    print(df.head())
    print(df.tail())
    return(df)
  except:
    pass

def graphData(stock):
  try:
    df = stock
    df.reset_index(inplace = True)
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)
    ax1.plot(df.Date, df.Open)
    ax1.plot(df.Date, df.High)
    ax1.plot(df.Date, df.Low)
    ax1.plot(df.Date, df.Close)
    ax1.grid(True)

    ax2 = plt.subplot(2,1,2)
    ax2.bar(df.Date,df.Volume)

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    for label in ax1.xaxis.get_tickerlabels():
      label.set_rotation(45)

    return
  except:
    pass

def timeSeries(df):
    df['f01'] = df['Adj Close'].shift(-1)
    df['f02'] = df['Adj Close'].shift(-2)
    df['f03'] = df['Adj Close'].shift(-3)
    df['f04'] = df['Adj Close'].shift(-4)
    df['f05'] = df['Adj Close'].shift(-5)
    df['f06'] = df['Adj Close'].shift(-6)
    df['f07'] = df['Adj Close'].shift(-7)

    df = df[['Close', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07']]

    #print(df.head())
    #print(df.tail())
    df.dropna(inplace = True)

    return df

def dataSpecify(df):

    #print(df.head(), df.tail())
    df['50 MAvg'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
    df['20 MAvg'] = df['Close'].rolling(window = 20, min_periods = 0).mean()
    df['10 MAvg'] = df['Close'].rolling(window = 10, min_periods = 0).mean()
    df['VolumeMAvg'] = df['Volume'].rolling(window = 50, min_periods = 0).mean()
    df['volitility'] = (df['Close']-df['50 MAvg'])/df['Close'] * 100

    df['vshift'] = df['volitility'].shift(1)
    df['volSlope'] = df['volitility'] - df['vshift']

    df['50MAshift'] = df['50 MAvg'].shift(1)
    df['50MASlope'] = (df['50 MAvg'] - df['50MAshift'])

    df['% change day'] = (df['Open'] - df['Close']) / df['Open'] * 100
    df['% daily volit'] = (df['High'] - df['Close']) / df['High'] * 100

    df.fillna(0, inplace = True)

    #df = df[['Close', '50 MAvg', 'Volume', 'volitility', '% daily volit','% change day' ,'VolumeMAvg']]
    #df = df[['Close', '50 MAvg', 'Volume', 'volitility', '% daily volit','% change day' ,'VolumeMAvg']]
    df = df[['Close', 'Open', '50 MAvg',  'volitility', '% daily volit','% change day', 'Volume','VolumeMAvg']]
    print(df.head(), df.tail())

    return df

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

    if volt > vavgt:
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



df = get_data('AAPL')
data = dataSpecify(df)
c = volumeCertainty(data)

print(c)

#ts = timeSeries(df)
#certainty_t = certainty_testing(df)
#ct = pd.DataFrame(certainty_t)
#X_train, X_test, y_train, y_test, X_lately = train_test_sets(ts, scaling = False, days = 8)

#X_train, X_test, y_train, y_test, X_lately = train_test_sets(data, scaling = True, days = 1)
#pred = nuralNetOne(X_train, X_test, y_train, y_test, X_lately)
#pred1 = nuralNetTwo(X_train, X_test, y_train, y_test, X_lately)
#pred = predictionReduce(pred, pred1)
