import pandas as pd
import numpy as np
import math

from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("GOOG (1).csv")
print(df.tail())

df['delta day'] = (df['Open'] - df['Close'])/df['Open'] * 100
df['HL volility'] = (df['High'] - df['Close'])/df['Close'] * 100

df = df[['Date', 'Close', 'HL volility', 'delta day', 'Volume']]
print(df.tail())

forecast_col = 'Close'
df.fillna(-99999, inplace = True)
forecast_out = 3

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.tail())

X = np.array(df.drop(['label', 'Date'], 1))
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
