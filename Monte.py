
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

folder_path = # FILE DIRECTORY HERE
GDP_file_path = #GDP FILE PATH HERE
unemployment_file_path = #YUR FILE PATH HERE
le_file_path = #LE FILE PATH HERE


gdp_data = pd.read_csv(GDP_file_path, names=['Quarter', 'GDP (£m)'])
  
years = 5
num_simulations = 10
def quarter_to_datetime(quarter_str):
    year, qtr = quarter_str.split(' Q')
    month = (int(qtr) - 1) * 3 + 1  
    return pd.to_datetime(f'{year}-{month:02d}-01')  
gdp_data['Quarter'] = gdp_data['Quarter'].apply(quarter_to_datetime)
gdp_data.set_index('Quarter', inplace=True)
gdp_data['GDP (£m)'] = pd.to_numeric(gdp_data['GDP (£m)'], errors='coerce')


def convert_time_to_date(time_str):
    year, quarter = time_str.split(' Q')
    month = (int(quarter) - 1) * 3 + 1 
    return pd.to_datetime(f'{year}-{month:02d}-01')  
unemployment_data = pd.read_csv(unemployment_file_path, skiprows=61, nrows=273-62, names=['Time', 'Unemployment Rate (%)'])
unemployment_data['Date'] = pd.to_datetime(unemployment_data['Time'].apply(convert_time_to_date))
unemployment_data['Unemployment Rate (%)'] = pd.to_numeric(unemployment_data['Unemployment Rate (%)'], errors='coerce')
unemployment_data.set_index('Date', inplace=True)

def process_life_exp_data(file_path):
    data = pd.read_csv(file_path, skiprows=4, header=None, usecols=[0, 1])
    data.columns = ['Year', 'LifeExpectancy']
    data['Year'] = pd.to_datetime(data['Year'])
    data['Year'] = data['Year'].dt.year
    data.set_index('Year', inplace=True)
    return data

data1 = process_life_exp_data(le_file_path)
# fit models
gdp_model = ARIMA(gdp_data['GDP (£m)'], order=(0,1,0))
gdp_model_fit = gdp_model.fit()

yur_model = ARIMA(unemployment_data['Unemployment Rate (%)'], order=(2,1,0))
yur_model_fit = yur_model.fit()

# extract residuals
gdp_residuals = gdp_model_fit.resid
yur_residuals = yur_model_fit.resid
years = 5
quarters = years * 4  # Assuming 4 quarters per year
num_simulations = 1000


simulated_gdp_paths = np.zeros((quarters, num_simulations))
simulated_yur_paths = np.zeros((quarters, num_simulations))
# last known values
last_gdp_value = gdp_data['GDP (£m)'].iloc[-1]
last_yur_value = unemployment_data['Unemployment Rate (%)'].iloc[-1]

# monte carlo simulations for GDP
for i in range(num_simulations):
    random_residuals_gdp = np.random.choice(gdp_residuals, size=quarters)
    future_values_gdp = [last_gdp_value]
    for residual in random_residuals_gdp:
        future_value_gdp = future_values_gdp[-1] + residual
        future_values_gdp.append(future_value_gdp)
    simulated_gdp_paths[:, i] = future_values_gdp[1:]

# monte carlo simulations for YUR
for i in range(num_simulations):
    random_residuals_yur = np.random.choice(yur_residuals, size=quarters)
    future_values_yur = [last_yur_value]
    for residual in random_residuals_yur:
        future_value_yur = future_values_yur[-1] + residual
        future_values_yur.append(future_value_yur)
    simulated_yur_paths[:, i] = future_values_yur[1:]


life_exp_data = process_life_exp_data(le_file_path)


life_exp_model = ARIMA(life_exp_data['LifeExpectancy'], order=(2,0,0))
life_exp_model_fit = life_exp_model.fit()

# extract life expectancy residuals
life_exp_residuals = life_exp_model_fit.resid

# define simulation parameters
years = 5
num_simulations = 1000
simulated_life_exp_paths = np.zeros((years, num_simulations))

# last known value of life expectancy
last_life_exp_value = life_exp_data['LifeExpectancy'].iloc[-1]

# monte Carlo simulations for life expectancy
for i in range(num_simulations):
    random_residuals_life_exp = np.random.choice(life_exp_residuals, size=years)
    future_values_life_exp = [last_life_exp_value]
    for residual in random_residuals_life_exp:
        future_value_life_exp = future_values_life_exp[-1] + residual
        future_values_life_exp.append(future_value_life_exp)
    simulated_life_exp_paths[:, i] = future_values_life_exp[1:]