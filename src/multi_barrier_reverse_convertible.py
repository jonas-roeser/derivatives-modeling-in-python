'''
University of St.Gallen

8,194: Derivatives Modeling in Python
--- Case Study ---
5 May 2024

Dr. Mathis Moerke
'''

# Load packages
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# Packages for IV calculation
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import norm
# from scipy.optimize import brentq
# from datetime import datetime

# # Function to calculate Black-Scholes price
# def black_scholes_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# def black_scholes_put(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# # Function to find implied volatility
# def implied_volatility(option_price, S, K, T, r, option_type):
#     try:
#         option_price = float(option_price)
#         S = float(S)
#         K = float(K)
#         T = float(T)
#     except ValueError:
#         return np.nan  # Handle incorrect types gracefully

#     if option_type == 'C':
#         price_func = lambda sigma: black_scholes_call(S, K, T, r, sigma) - option_price
#     elif option_type == 'P':
#         price_func = lambda sigma: black_scholes_put(S, K, T, r, sigma) - option_price
#     else:
#         return np.nan  # Invalid option type

#     try:
#         return brentq(price_func, 0.01, 3.0)
#     except ValueError:
#         return np.nan  # Handle cases where brentq does not converge

# # Load data
# stocks = pd.read_csv('data/stock_prices.csv')
# options = pd.read_csv('data/option_prices.csv')

# options = options[options['date'] != 'Currency']

# # Process data
# stocks['Date'] = pd.to_datetime(stocks['Date'])
# options['date'] = pd.to_datetime(options['date'])
# options['exdate'] = pd.to_datetime(options['exdate'])

# stocks = stocks.rename(columns={'Date': 'date'})

# print("Columns in options:", options.columns)
# print("Columns in stocks:", stocks.columns)

# # Assume 'date' is your datetime column and the rest are stock prices
# stocks_long = stocks.melt(id_vars=['date'], var_name='underlying', value_name='price')

# # Calculate moneyness, time to maturity, and implied volatilities
# options = options.merge(stocks_long, left_on=['underlying', 'date'], right_on=['underlying', 'date'])
# options['moneyness'] = (options['strike'] / options['price']).round(2)
# options['time_to_maturity'] = (options['exdate'] - options['date']).dt.days / 365.25

# # Define the risk free rate (1 Year US Forward Rate as of 26.06.2018 see file "interest_rates.csv")
# r = 0.0277094

# # Rename Columns for IV Calculation
# options = options.rename(columns={'Date': 'date', 'strike': 'K', 'price': 'S', 'time_to_maturity': "T"})

# # Calculate implied volatility for each option and add it as a new column
# options['implied_volatility'] = options.apply(
#     lambda row: implied_volatility(
#         row['option_price'], row['S'], row['K'], row['T'], r, row['option_type']
#     ), axis=1
# )

# # Filter data for options as of June 26, 2018
# specific_date = pd.Timestamp('2018-06-26')
# one_year_options = options[
#     (options['date'] == specific_date) &
#     (options['T'].between(0.9, 1.1))
# ]

# atm_options = one_year_options[one_year_options['moneyness'].between(0.96, 1.04)]

# # Function to get or interpolate the implied volatility
# def get_atm_iv(options, stock_name):
#     stock_options = options[options['underlying'] == stock_name]
#     if not stock_options.empty:
#         # Find the option closest to moneyness of 1
#         stock_options['moneyness_diff'] = abs(stock_options['moneyness'] - 1)
#         closest_atm_option = stock_options.loc[stock_options['moneyness_diff'].idxmin()]
#         return closest_atm_option['implied_volatility']
#     else:
#         return None

# # Get ATM implied volatility for each stock
# adobe_iv = get_atm_iv(atm_options, 'Adobe')
# apple_iv = get_atm_iv(atm_options, 'Apple')
# microsoft_iv = get_atm_iv(atm_options, 'Microsoft')


# # Calc Bond Price
# def price_bond(nominal, coupon_rate, years_to_maturity, r, credit_spread, frequency=2):

#     total_yield = r + credit_spread
#     coupon_payment = nominal * coupon_rate / frequency
#     num_payments = years_to_maturity * frequency
#     discount_factors = [(1 + total_yield / frequency) ** -n for n in range(1, num_payments + 1)]
    
#     # Calculate present value of future coupon payments
#     present_value_coupons = sum(coupon_payment * df for df in discount_factors)
    
#     # Calculate present value of the nominal value (paid at maturity)
#     present_value_nominal = nominal / (1 + total_yield / frequency) ** num_payments
    
#     # Total bond price is the sum of the present value of coupons and nominal
#     bond_price = present_value_coupons + present_value_nominal
#     return bond_price

def sim_correlated_paths(underlying_prices, volatilities, pricing_date=None, expiration_date=None, i_rate=0.01, exchange='SIX'):
    '''
    Simulate correlated stock price paths.

    Parameters
    ---
    underlying_prices : pd.DataFrame
        Data frame with underplying prices.
    volatilities : pd.DataFrame
        Option implied volatilities.
    pricing_date : datetime, default None
        Pricing date.
    expiration_date : datetime, default None
        Expiration date.
    i_rate : float, default 0.01
        Interest rate (cont. comp., annualized).
    exchange : str, default 'SIX'
        Exchange the product is traded on.
    
    Returns
    ---
    correlated_paths : data frame
        Data frame of correlated paths.
    '''
    # Compute number of days from pricing to expiration date
    n_days = (expiration_date - pricing_date).days + 1

    # Compute time delta in terms of years
    delta_years = n_days / 365.25

    # Create a calendar
    exchange = mcal.get_calendar(exchange)

    # Generate data range
    date_range = exchange.valid_days(start_date=pricing_date, end_date=expiration_date).tz_convert(None)
    
    # Compute time step per trading day
    time_step = delta_years / len(date_range)

    # Copute log returns
    log_returns = np.log(underlying_prices / underlying_prices.shift())

    # Compute correlations
    corr_matrix = log_returns.corr()
    corr_12 = corr_matrix.iloc[0,1]
    corr_13 = corr_matrix.iloc[0,2]
    corr_23 = corr_matrix.iloc[1,2]

    # Compute additional coefficients for correlated paths computation
    corr_23_star = (corr_23 - corr_12 * corr_13) / np.sqrt(1 - corr_12**2)
    corr_33_star = np.sqrt((1 - corr_12**2 - corr_23**2 - corr_13**2 + 2 * corr_12 * corr_13 * corr_23) / (1 - corr_12**2))

    # Create empty dataframe
    correlated_paths = pd.DataFrame(index=date_range, columns=underlying_prices.columns)

    # Generate correlated stock paths
    for i, date in enumerate(date_range):

        # For first date
        if (i == 0) & (date == pricing_date):

            # Set first price equal to underlying
            correlated_paths.loc[date] = underlying_prices.loc[pricing_date]

        # For all other dates
        else:

            # Set random errors
            e_1 = np.random.normal()
            e_2 = np.random.normal()
            e_3 = np.random.normal()

            # First underlying
            correlated_paths.loc[date, correlated_paths.columns[0]] = correlated_paths.iloc[i - 1, 0] * np.exp((i_rate - 0.5 * volatilities.iloc[0,0]**2) * time_step + volatilities.iloc[0,0] * np.sqrt(time_step) * e_1)

            # Second underlying
            correlated_paths.loc[date, correlated_paths.columns[1]] = correlated_paths.iloc[i - 1, 1] * np.exp((i_rate - 0.5 * volatilities.iloc[0,1]**2) * time_step + volatilities.iloc[0,1] * np.sqrt(time_step) * (corr_12 * e_1 + np.sqrt(1 - corr_12**2) * e_2))

            # Third underlying
            correlated_paths.loc[date, correlated_paths.columns[2]] = correlated_paths.iloc[i - 1, 2] * np.exp((i_rate - 0.5 * volatilities.iloc[0,2]**2) * time_step + volatilities.iloc[0,2] * np.sqrt(time_step) * (corr_13 * e_1 + corr_23_star * e_2 + corr_33_star * e_3))
    
    return correlated_paths

def worst_of_down_and_in_put(barrier_levels, strike_prices, conversion_ratios, *args, pricing_date=None, expiration_date=None, i_rate=0.01, sim_function=sim_correlated_paths, sim_runs=100, **kwargs):
    '''
    Calcualte worst-of down-and-in put option price.
    
    Parameters
    ---
    barrier_levels : pd.DataFrame
        Underlying barrier levels.
    strike_prices : pd.DataFrame
        Underlying strike prices.
    conversion_ratios : pd.DataFrame
        Underlying conversion ratios.
    pricing_date : datetime, default None
        Pricing date.
    expiration_date : datetime, default None
        Expiration date.
    i_rate : float, default 0.01
        Interest rate.
    sim_function : callable, deault sim_correlated_paths
        Function for computing correlated price paths.
    sim_runs : int, default 100
        Number of simulation runs.

    Returns
    ---
    option_price : float
        Price of the option.
    '''
    # Vector containing the payoff for each simulated price path
    payoffs = np.zeros(sim_runs)

    # Run monte carlo simulation
    for i in range(sim_runs):

        # Simulate price paths
        prices = sim_function(*args, pricing_date, expiration_date, i_rate, **kwargs)
        
        # Determine if the option knocks in for any stock
        knocks_in = (prices <= barrier_levels.iloc[0]).any().any()

        # If the option knocks in:
        if knocks_in:

            # Get expiration prices:
            expiration_prices = prices.loc[expiration_date]

            # Check if option is in the money:
            in_the_money = (expiration_prices < strike_prices).any().any()

            # If the option is in the money:
            if in_the_money:

                # Calculate rerlative performance
                performance = (expiration_prices - strike_prices) / strike_prices
                
                # Find worst performing stock
                worst_performing_stock = performance.idxmin()

                # Compute payoff
                payoffs[i] = (strike_prices[worst_performing_stock] - expiration_prices[worst_performing_stock]) * conversion_ratios[worst_performing_stock]
    
    # Compute number of days from pricing to expiration date
    n_days = (expiration_date - pricing_date).days + 1

    # Compute time delta in terms of years
    delta_years = n_days / 365.25

    # Compute option price from average payoffs
    option_price = np.exp(- i_rate * delta_years) * sum(payoffs) / sim_runs

    return option_price

if __name__ == '__main__':

    # Load stock prices
    stock_prices = pd.read_csv('../data/stock_prices.csv', parse_dates=['Date'], index_col='Date').convert_dtypes().rename(columns=str.lower)

    # Load option prices
    # option_prices = pd.read_csv('../data/option_prices.csv', parse_dates=['exdate']).convert_dtypes()
    # option_prices = option_prices[~((option_prices['date'] == 'Currency') & (option_prices['option_price']=='USD'))]
    # option_prices['date'] = pd.to_datetime(option_prices['date'])
    # option_prices.set_index('date', inplace=True)

    # Load interest rates
    # interest_rates = pd.read_csv('../data/interest_rates.csv', parse_dates=['date'], index_col='date').convert_dtypes()

    # Load CDS spreads
    # cds_spreads = pd.read_csv('../data/cds_spreads.csv', parse_dates=['date'], index_col='date').convert_dtypes()

    # Set random seed to ensure replicability
    np.random.seed(42)

    # Set product parameters
    NOMINAL_VALUE = 1000
    pricing_date = pd.to_datetime('2018-06-27')
    expiration_date = pd.to_datetime('2019-06-20')
    strike_prices = stock_prices.loc[pricing_date]
    KICK_IN_LEVEL = 0.65
    barrier_levels = strike_prices * KICK_IN_LEVEL
    conversion_ratios = round(NOMINAL_VALUE / strike_prices, 4)

    # These should be computed in code
    I_RATE = 0.01
    volatilities = pd.DataFrame([[0.2, 0.25, 0.15]], columns=stock_prices.columns)

    # nominal_value = 1000
    # annual_coupon_rate = 0.085
    # maturity_in_years = 1
    # r = 0.0277094 # 1 Year US Forward Rate as of 26.06.2018
    # credit_risk_spread = 0.00172599899999999 # 1 year Unsubordinated UBS cds as of 26.06.2018

    # Calculate bond price
    # bond_price = price_bond(nominal_value, annual_coupon_rate, maturity_in_years, r, credit_risk_spread)

    # Calcualte worst-of down-and-in put option price
    option_price = worst_of_down_and_in_put(barrier_levels, strike_prices, conversion_ratios,
                             stock_prices, volatilities, # *args (only inner)
                             pricing_date=pricing_date, expiration_date=expiration_date, i_rate=I_RATE,
                             exchange='SIX' # **kwargs (only inner)
                             )
    