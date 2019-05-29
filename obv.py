from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import csv
import pandas as pd
ti = TechIndicators(key='IA5WT3NKED90E2DF', output_format='pandas')
data, metadata = ti.get_obv(symbol='NASDAQ:AAPL', interval='1mmin') #chage interval


data.plot()
plt.show()