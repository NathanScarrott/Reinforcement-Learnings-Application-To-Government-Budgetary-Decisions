

############################# MONTE CARLO GENERATION (SAME AS MONTE.PY BUT NO PLOTS) #########################

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


GDP_file_path = 'GDP FILE PATH'
unemployment_file_path = 'YUR FILE PATH'
le_file_path = 'LE FILE PATH'


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


######################   PPO   ############################

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import MinMaxScaler

class BudgetAllocationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, simulated_gdp_paths, simulated_le_paths, simulated_yur_paths, sigma_G, sigma_L, sigma_Y, initial_B):
        super(BudgetAllocationEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([1, 1, 1, 1]),  # normalized values
                                            dtype=np.float32)
        

        self.sigma_G = sigma_G  #
        self.sigma_L = sigma_L  
        self.sigma_Y = sigma_Y  
        self.initial_B = initial_B   

        self.simulated_gdp_paths = simulated_gdp_paths
        self.simulated_le_paths = simulated_le_paths
        self.simulated_yur_paths = simulated_yur_paths

        self.gdp_min, self.gdp_max = self.simulated_gdp_paths.min(), self.simulated_gdp_paths.max()
        self.le_min, self.le_max = self.simulated_le_paths.min(), self.simulated_le_paths.max()
        self.yur_min, self.yur_max = self.simulated_yur_paths.min(), self.simulated_yur_paths.max()

        self.scaler = MinMaxScaler()
        
        # scale to the min and max bounds
        self.scaler.fit([[self.gdp_min, self.le_min, self.yur_min],
                         [self.gdp_max, self.le_max, self.yur_max]])

    def reset(self):
        self.current_step = 0
        self.current_gdp_path = np.random.choice(range(self.simulated_gdp_paths.shape[1]))
        self.current_le_path = np.random.choice(range(self.simulated_le_paths.shape[1]))
        self.current_yur_path = np.random.choice(range(self.simulated_yur_paths.shape[1]))
        self.G = self.simulated_gdp_paths[0, self.current_gdp_path]
        self.L = self.simulated_le_paths[0, self.current_le_path]
        self.Y = self.simulated_yur_paths[0, self.current_yur_path]
        self.B = 300000000000*5  # initial budget
        
        scaled_values = self.scaler.transform([[self.G, self.L, self.Y]])
        self.G, self.L, self.Y = scaled_values[0]

        return np.array([self.G, self.L, self.Y, self.B / self.B], dtype=np.float32)  # normalize budget

    def step(self, action):
        # convert the action to budget allocation amounts
        f_i, f_h, f_e = self.convert_action_to_funding(action)
        
        # calculate the stochastic elements
        epsilon_G = np.random.normal(0, math.sqrt(gdp_data['GDP (£m)'].std()))
        epsilon_L = epsilon_L = np.random.normal(0, math.sqrt(data1['LifeExpectancy'].std()))
        epsilon_Y = np.random.normal(0, math.sqrt(unemployment_data['Unemployment Rate (%)'].std()))#np.random.normal(0, self.sigma_Y)

        # calculate the new indicators
        self.G += 4.17e-3 * f_i / self.B + epsilon_G
        self.L += 35 * f_h / self.B + epsilon_L
        self.Y += -10.81 * f_e / self.B + epsilon_Y

        # update budget
        self.B -= (f_i + f_h + f_e)

        # normalize the state values to be between 0 and 1
        scaled_state = self.scaler.transform([[self.G, self.L, self.Y]])
        self.G, self.L, self.Y = scaled_state[0]

        # calculate reward
        reward = self.calculate_reward(self.G, self.L, self.Y, f_i, f_h, f_e, self.B)
        
        
        done = self.B <= 0 or self.current_step >= self.simulated_gdp_paths.shape[0] - 1
        self.current_step += 1
        next_state = np.array([self.G, self.L, self.Y, self.B / self.initial_B], dtype=np.float32)
        
        return next_state, reward, done, {}

    def convert_action_to_funding(self, action):
        
        max_budget_pct_change = 0.1  # 10% of the total budget

    
        f_i = (action[0] * max_budget_pct_change + 1) * (self.B / 3)  # infrastructure
        f_h = (action[1] * max_budget_pct_change + 1) * (self.B / 3)  # healthcare
        f_e = (action[2] * max_budget_pct_change + 1) * (self.B / 3)  # education


        return f_i, f_h, f_e

    def calculate_reward(self, G_scaled, L_scaled, Y_scaled, f_i, f_h, f_e, B):
        # reverse scaling 
        G, L, Y = self.scaler.inverse_transform([[G_scaled, L_scaled, Y_scaled]])[0]
        
      
        # apply impact of the budget allocations on the indicators
        G_impact = 4.17e-3 * f_i / B
        L_impact = 35 * f_h / B
        Y_impact = -10.81 * f_e / B

        # include stochastic elements
        epsilon_G = np.random.normal(0, math.sqrt(gdp_data['GDP (£m)'].std()))
        epsilon_L = np.random.normal(0, math.sqrt(data1['LifeExpectancy'].std()))
        epsilon_Y = np.random.normal(0, math.sqrt(unemployment_data['Unemployment Rate (%)'].std()))

   
        G_real = G_scaled + G_impact + epsilon_G
        L_real = L_scaled + L_impact + epsilon_L
        Y_real = Y_scaled + Y_impact + epsilon_Y

        

        # calculate the reward
        reward =  G_real + L_real - Y_real

        # penalize for budget over-allocation or underutilization
        total_budget_allocated = f_i + f_h + f_e
        if total_budget_allocated > B:
            reward -= 2e-12 * (total_budget_allocated - B)  # Penalty for overspending
        elif total_budget_allocated < 0.9 * B:
            reward -= 1e-12*(0.9 * B - total_budget_allocated)  # Penalty for underutilization

        return reward


    
    def render(self, mode='human'):
        pass
import math

sigma_G = math.sqrt(gdp_data['GDP (£m)'].std()) 
sigma_L = math.sqrt(data1['LifeExpectancy'].std())
sigma_Y = math.sqrt(unemployment_data['Unemployment Rate (%)'].std()) 
initial_B = 300000000000 *5

env = BudgetAllocationEnv(simulated_gdp_paths, simulated_life_exp_paths, simulated_yur_paths, sigma_G, sigma_L, sigma_Y, initial_B)
env = DummyVecEnv([lambda: env])

# initialize the model with policy and environment
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00001, n_steps=2048, batch_size=128, gamma = 0.99)

# train model
model.learn(total_timesteps=20000)

# evaluate the policy
mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

model.save("ppo_budget_allocation")
model = PPO.load("ppo_budget_allocation")

