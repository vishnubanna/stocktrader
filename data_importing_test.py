import pandas as pd
import quandl

'''
lesson 1
regression modeling: commonly used with stock prices

fitting data to a curve, most commonly a line

with supervised ML: everything boiles down to features and label
    features: the attributes you want the model to be used on
    label: the name or the target of the machine learning algorithem, what you want it to predict
        eventually: what you eventually want to predict

in Machine learning:
    you want meaningful features or else you will find useless coralations that lead to bad predictions

'''
#lesson 1 code
quandl.ApiConfig.api_key = "cnwwaCXC3M_dXVi6pEzN"
df = quandl.get('WIKI/GOOGL')
#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['high/low volility PCT change'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT change price'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close','high/low volility PCT change', 'PCT change price', 'Adj. Volume']]

print(df.head())

'''
about stock market
output:
              Open    High     Low    Close      Volume  Ex-Dividend  Split Ratio  Adj. Open  Adj. High   Adj. Low  Adj. Close  Adj. Volume
Date
2004-08-19  100.01  104.06   95.96  100.335  44659000.0          0.0          1.0  50.159839  52.191109  48.128568   50.322842   44659000.0
2004-08-20  101.01  109.08  100.50  108.310  22834300.0          0.0          1.0  50.661387  54.708881  50.405597   54.322689   22834300.0
2004-08-23  110.76  113.48  109.05  109.400  18256100.0          0.0          1.0  55.551482  56.915693  54.693835   54.869377   18256100.0
2004-08-24  111.24  111.60  103.57  104.870  15247300.0          0.0          1.0  55.792225  55.972783  51.945350   52.597363   15247300.0
2004-08-25  104.76  108.00  103.88  106.000   9188600.0          0.0          1.0  52.542193  54.167209  52.100830   53.164113    9188600.0


of these we want to look at the adjusted values because they are better representatiosn of the date we want
adjusted is the value of the stock after stock Splits. say one share = 1000 dollars no one will buy that so you split the share into 2 stocks, so each stock is 500 dollars, more reasonable. adjusted acounts for this

using both open and adjusting open, is use less because both are so highly coralated. useless features can fuck up your ML model
Volume in this case: the volume of stock bought, how many trades were made that day, could be realted to volume(suposedly check on this tomorrow)

what is Open? the starting price of the day
what is High? the margin of high and low tell you the voltility of the stock for the day, all stock relative to the open stock (i think this is realtive max)
what is Low? (i think this is realtive min)
what is Close? the final cost of the day (check this) relation ship to open, in one day, how much did it go up

lesson : sentdex machine learning: regression
'''

'''
we can use a ticker based system that feeds data into our algorithem at some rate, we add it to a que data structure, so a first in first out system, and we analyze the data set in batches
of like 2 to 3 data points.

if the ticker passes data in at 1 time every 20 seconds we will be getting updated info every 20 seconds. good stuff.
'''
