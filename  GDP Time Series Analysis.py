##### GDP Time Series Analysis  ########
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.api import add_constant, OLS
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import detrend
from numpy import cumsum



folder_path = 'working directory here'
GDP_file_path = 'data file path here'

def test_stationarity(timeseries):
    # Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')  
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dfoutput  



def apply_transformations_and_test(dataframe, column_name):
    transformations = {
        'original': lambda x: x,
        'log': np.log,
        'sqrt': np.sqrt,
        'square': np.square,
        'diff': lambda x: x.diff().dropna(),
        'log_diff': lambda x: np.log(x).diff().dropna(),
        'sqrt_diff': lambda x: np.sqrt(x).diff().dropna()
    }
    stationary_results = {}
    for trans_name, transform_function in transformations.items():
        print(f"\nApplying transformation: {trans_name}")
        transformed_data = transform_function(dataframe[column_name])
        dfoutput = test_stationarity(transformed_data)
        # if test statistic is less than the critical value at 5%, consider it stationary
        if dfoutput['Test Statistic'] < dfoutput['Critical Value (5%)']:
            stationary_results[trans_name] = True
        else:
            stationary_results[trans_name] = False
    return stationary_results


gdp_data = pd.read_csv(GDP_file_path, names=['Quarter', 'GDP (£m)'])


def quarter_to_datetime(quarter_str):
    year, qtr = quarter_str.split(' Q')
    month = (int(qtr) - 1) * 3 + 1  #
    return pd.to_datetime(f'{year}-{month:02d}-01')  


gdp_data['Quarter'] = gdp_data['Quarter'].apply(quarter_to_datetime)

gdp_data.set_index('Quarter', inplace=True)


gdp_data['GDP (£m)'] = pd.to_numeric(gdp_data['GDP (£m)'], errors='coerce')

# plot time series
plt.figure(figsize=(12, 6))
plt.plot(gdp_data.index, gdp_data['GDP (£m)'], 'o-', label='GDP (£m)')
plt.title('Time Series Plot of UK GDP in £m')
plt.xlabel('Quarter')  
plt.ylabel('GDP (£m)') 
plt.xticks(rotation=45) 
plt.legend()
plt.tight_layout()  
plt.show()


gdp_data['Time'] = gdp_data.index.year + gdp_data.index.quarter / 4.0

#fit a linear model 
X = add_constant(gdp_data['Time'])  # Add a constant term for the intercept
model = OLS(gdp_data['GDP (£m)'], X).fit()

#predict the trend 
gdp_data['Trend'] = model.predict(X)

# subtract trend from the original data to get detrended data
gdp_data['Detrended_GDP'] = gdp_data['GDP (£m)'] - gdp_data['Trend']


# plot detrended data
plt.figure(figsize=(12, 6))
plt.plot(gdp_data['Time'], gdp_data['Detrended_GDP'], label='Detrended GDP (£m)', color='green')
plt.title('Detrended Time Series of UK GDP')
plt.xlabel('Time')
plt.ylabel('Detrended GDP (£m)')
plt.legend()
plt.show()

gdp_data['Diff_Detrended_GDP'] = gdp_data['Detrended_GDP'].diff().dropna()






fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# ACF plot
plot_acf(gdp_data['Diff_Detrended_GDP'].dropna(), lags=40, alpha=0.05, ax=axes[0])  # Use dropna() to remove any NaN values
axes[0].set_title('Autocorrelation Function (ACF) of UK GDP Growth')
axes[0].set_xlabel('Lag')  # Label for the x-axis
axes[0].set_ylabel('Autocorrelation', labelpad=-10)  # Adjust the labelpad as needed

# PACF plot
plot_pacf(gdp_data['Diff_Detrended_GDP'].dropna(), lags=40, alpha=0.05, method='ywm', ax=axes[1])  # Use dropna() to remove any NaN values
axes[1].set_title('Partial Autocorrelation Function (PACF) of UK GDP Growth')
axes[1].set_xlabel('Lag')  # Label for the x-axis
axes[1].set_ylabel('Partial Autocorrelation', labelpad=-10)  # Adjust the labelpad as needed

plt.tight_layout()
plt.show()




test_stationarity(gdp_data['Detrended_GDP'])


GDPtest = apply_transformations_and_test(gdp_data, 'Detrended_GDP')
print(GDPtest)

#fit ARIMA(0,1,0) model to the detrended GDP data
arima_model = ARIMA(gdp_data['Detrended_GDP'].dropna(), order=(0, 1, 0))

# fit the model
arima_result = arima_model.fit()


gdp_data['Diff_Detrended_GDP'] = gdp_data['Detrended_GDP'].diff()  #differencing


gdp_data_diff = gdp_data['Diff_Detrended_GDP'].dropna()


test_stationarity(gdp_data_diff)


# plot detrended differenced data
plt.figure(figsize=(12, 6))
plt.plot(gdp_data['Time'][1:], gdp_data_diff, label='Differenced Detrended GDP (£m)', color='green')
plt.title('Differenced Detrended Time Series of UK GDP')
plt.xlabel('Time')
plt.ylabel('Differenced Detrended GDP (£m)')
plt.legend()
plt.show()

# plot ACF and PACF 
plt.figure(figsize=(12, 6))
plot_acf(gdp_data_diff, lags=40, alpha=0.05)  
plt.title('Autocorrelation Function (ACF) of UK GDP')
plt.xlabel('Lag (year)') 
plt.ylabel('Autocorrelation')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(gdp_data_diff, lags=40, alpha=0.05, method='ywm')  
plt.title('Partial Autocorrelation Function (PACF) of UK GDP')
plt.xlabel('Lag (year)')  
plt.ylabel('Partial Autocorrelation')
plt.show()
