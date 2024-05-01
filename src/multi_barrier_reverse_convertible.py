'''
University of St.Gallen

8,194: Derivatives Modeling in Python
--- Case Study ---
5 May 2024

Dr. Mathis Moerke
'''

# IV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

# Function to calculate Black-Scholes price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to find implied volatility
def implied_volatility(option_price, S, K, T, r, option_type):
    try:
        option_price = float(option_price)
        S = float(S)
        K = float(K)
        T = float(T)
    except ValueError:
        return np.nan  # Handle incorrect types gracefully

    if option_type == 'C':
        price_func = lambda sigma: black_scholes_call(S, K, T, r, sigma) - option_price
    elif option_type == 'P':
        price_func = lambda sigma: black_scholes_put(S, K, T, r, sigma) - option_price
    else:
        return np.nan  # Invalid option type

    try:
        return brentq(price_func, 0.01, 3.0)
    except ValueError:
        return np.nan  # Handle cases where brentq does not converge


# Load data
stocks = pd.read_csv('data/stock_prices.csv')
options = pd.read_csv('data/option_prices.csv')

options = options[options['date'] != 'Currency']

# Process data
stocks['Date'] = pd.to_datetime(stocks['Date'])
options['date'] = pd.to_datetime(options['date'])
options['exdate'] = pd.to_datetime(options['exdate'])

stocks = stocks.rename(columns={'Date': 'date'})

print("Columns in options:", options.columns)
print("Columns in stocks:", stocks.columns)

# Assume 'date' is your datetime column and the rest are stock prices
stocks_long = stocks.melt(id_vars=['date'], var_name='underlying', value_name='price')

# Calculate moneyness, time to maturity, and implied volatilities
options = options.merge(stocks_long, left_on=['underlying', 'date'], right_on=['underlying', 'date'])
options['moneyness'] = (options['strike'] / options['price']).round(2)
options['time_to_maturity'] = (options['exdate'] - options['date']).dt.days / 365.25

# Define the risk free rate (Year Horizon as of 26.06.2018 see file "interest_rates.csv")
r = -0.005 

# Rename Columns for IV Calculation
options = options.rename(columns={'Date': 'date', 'strike': 'K', 'price': 'S', 'time_to_maturity': "T"})

# Calculate implied volatility for each option and add it as a new column
options['implied_volatility'] = options.apply(
    lambda row: implied_volatility(
        row['option_price'], row['S'], row['K'], row['T'], r, row['option_type']
    ), axis=1
)

# Filter data for options as of June 26, 2018
specific_date = pd.Timestamp('2018-06-26')
one_year_options = options[
    (options['date'] == specific_date) &
    (options['T'].between(0.9, 1.1))
]

atm_options = one_year_options[one_year_options['moneyness'].between(0.96, 1.04)]

# Function to get or interpolate the implied volatility
def get_atm_iv(options, stock_name):
    stock_options = options[options['underlying'] == stock_name]
    if not stock_options.empty:
        # Find the option closest to moneyness of 1
        stock_options['moneyness_diff'] = abs(stock_options['moneyness'] - 1)
        closest_atm_option = stock_options.loc[stock_options['moneyness_diff'].idxmin()]
        return closest_atm_option['implied_volatility']
    else:
        return None

# Get ATM implied volatility for each stock
adobe_iv = get_atm_iv(atm_options, 'Adobe')
apple_iv = get_atm_iv(atm_options, 'Apple')
microsoft_iv = get_atm_iv(atm_options, 'Microsoft')





def sim_correlated_path():
    '''
    Simulate correlated stock price paths.

    Parameters
    ---
    price : str, default None
        File name.
    vola : 
        Vola of underlying.
    rate : 
        Interest rate (cont. comp., annualized).
    t :
        Entire time.
    
    Returns
    ---
    price : Series
        Series of prices.
    '''
    # Insert code
    pass

def worst_of_down_and_in_put():
    '''
    Calcualte worst-of down-and-in put option price.
    
    Parameters
    ---
    ...

    Returns
    ---
    ...
    '''
    # Insert code
    pass

if __name__ == '__main__':

    # Simulate prices
    sim_correlated_path()

    # Calcualte worst-of down-and-in put option price
    worst_of_down_and_in_put()