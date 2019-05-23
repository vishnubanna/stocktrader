import bs4 as bs
import pickle
import requests
import datetime as dt
import time
import os
import pandas as pd
#import pandas_datareader.data as web
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = str(ticker).replace('.','-')
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    print(tickers)
    return tickers

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2019, 5, 16)

    for ticker in tickers[0:10]:
        print(ticker)
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            #df = web.DataReader(ticker, 'yahoo', start, end)
            df = pdr.get_data_yahoo(ticker, start, end)
            df.to_csv('stocks_dfs/{}.csv'.format(ticker))
        else:
            print('already have {}'.format(ticker))
    return

#get_data_from_yahoo()
