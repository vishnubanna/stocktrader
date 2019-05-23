# from google.colab import drive
# drive.mount("/content/drive")
# import os

import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()


def get_data(ticker):
  try:
    print(ticker)
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.today()
    df = pdr.get_data_yahoo(ticker, start, end)
    print(df.head())
    return(df)
  except:
    pass

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

df = get_data('AAPL')
print(df.head())
graphData(df)
getCandlestick(df)
