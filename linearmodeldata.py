import pandas as pd
import numpy as np
import math

from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression

import datetime
import matplotlib.pyplot as plt
from matplotlib import style

import pickle

style.use('ggplot')


df = pd.read_csv("GOOG (1).csv", parse_dates = True, index_col = 0)
print(df.tail())


#takes todays price and 49 of the previous days price and takes the avaerage, and that is todays moving average, does this for every single day. can signal an uptrend or down trend in price
df['50 Moving avg'] = df['Close'].rolling(window = 50, min_periods = 0).mean()

#window = how many data points
#min_periods = 0 --> takes care of NAN vals. so if you dont have enough data points, so at data point 1 it does nothing at point 2 it does the avg of 1 and 2
# it does this until point 50 at which point it uses the max period.
# you could also remove nan values from the data yet, but then you loose data




df['delta day'] = (df['Open'] - df['Close'])/df['Open'] * 100
df['HL volility'] = (df['High'] - df['Close'])/df['Close'] * 100

df = df[['Close', 'HL volility', 'delta day', 'Volume', '50 Moving avg']]
print(df.tail())

forecast_col = 'Close'
#forecast_col = '50 Moving avg'
df.fillna(-99999, inplace = True)
forecast_out = 30

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.tail())

X = np.array(df.drop(['label'], 1))
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs = 10)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)


df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix) #iterates through the forecast_set
    #makes the future features not a number
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #sets all the first colomns to not a numbers and make the last colomns what ever i is # rewatch to make sure
    #df.loc :: references the index of the data frame
    #the indec in this case is the date, so if the date doesn't exist it is saying make a date
df['Close'].plot()
df['Forecast'].plot()
df['50 Moving avg'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
