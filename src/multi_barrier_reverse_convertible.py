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
from scipy.stats import norm
from scipy.optimize import minimize

def bond(nominal_value, coupon_rate, coupon_frequency, years_to_maturity, i_rate, credit_spread):
    '''
    Compute bond price.

    Parameters
    ---
    nominal_value : float
        Nominal value.
    coupon_rate : float
        Rate of coupon payments.
    coupon_frequency : int or float
        Number of coupon payments per year.
    years_to_maturity : int or float
        Time to maturity in years.
    i_rate : float
        Risk free rate (continuous & annualised).
    credit_spread : float
        Credit spread.
    
    Returns
    ---
    bond_price : float
        Price of the bond.
    '''
    # Compute total yield
    total_yield = i_rate + credit_spread

    # Compute coupon payment
    coupon_payment = nominal_value * coupon_rate / coupon_frequency

    # Compute number of payments
    n_payments = years_to_maturity * coupon_frequency

    # Compute discount factors
    discount_factors = [(1 + total_yield / coupon_frequency) ** -i for i in range(1, n_payments + 1)]
    
    # Compute present value of future coupon payments
    present_value_coupons = sum(coupon_payment * discount_factor for discount_factor in discount_factors)
    
    # Compute present value of the nominal value (paid at maturity)
    present_value_nominal = nominal_value * discount_factors[-1]
    
    # Compute total bond price as the sum of the present value of coupons and nominal
    bond_price = present_value_coupons + present_value_nominal

    return bond_price

def black_scholes(S, K, t, T, v, r, dividend_yield, option_type):
    '''
    Compute Black-Scholes option price.

    Parameters
    ---
    S : int or float
        Spot price of the underlying at time t.
    K : int or float
        Strike price of the option.
    t : pd.datetime
        Pricing date.
    T : pd.datetime
        Expiration date.
    volatility : int or float
        Volatility of the underlying (continuous & annualised).
    r : int or float
        Risk free rate to maturity (continuous & annualised).
    dividend_yield : float
        Dividend by share price.
    option_type : str
        Variable indicating call or put option, one of 'C' or 'P'.

    Returns
    ---
    option_price :
        Price of the option.
    '''
    # Create parameter dictionary
    parameters = {'S': S, 'K': K, 't': t, 'T': T, 'v': v, 'r': r, 'dividend_yield': dividend_yield, 'option_type': option_type}

    # Ensure all parameters except volatility and expiration date are of type Series
    parameters = {key: value if isinstance(value, pd.core.series.Series) else pd.Series(value) for key, value in parameters.items()}

    # Check if all parameters except volatility and expiration date have the same length
    if any(len(value) != len(parameters['S']) for key, value in parameters.items() if key not in {'T', 'v'}):
        raise ValueError('Option parameters are of different length')
    
    # Initialise list of option prices
    option_prices = []

    # Iteratre through all options
    for i in range(len(S)):

        # Set parameters for each option
        S = parameters['S'].iloc[i]
        K = parameters['K'].iloc[i]
        t = parameters['t'].iloc[i]
        T = parameters['T'].iloc[0]
        v = parameters['v'].iloc[0]
        r = parameters['r'].iloc[i]
        dividend_yield = parameters['dividend_yield'].iloc[i]
        option_type = parameters['option_type'].iloc[i]

        # Compute number of days from pricing to expiration date
        n_days = (T - t).days + 1 # Should this be in trading days?

        # Compute time to maturity T in years
        T_t = n_days / 365.25

        # Set call_put flag
        if option_type == 'C':
            call_put = 1
        elif option_type == 'P':
            call_put = -1
        else:
            raise ValueError('Invalid option type')

        # Check if option parameters are valid
        if S > 0 and K > 0 and T_t > 0 and v > 0 and r >= 0 and dividend_yield >= 0:

            # Compute Black-Scholes probability factor
            d = (np.log(S / K) + (r - dividend_yield + v**2 / 2) * T_t) / (v * np.sqrt(T_t))

            # Compute Black-Scholes option price
            option_price = call_put * S * np.exp(-dividend_yield * T_t) * norm.cdf(call_put * d) - call_put * K * np.exp(-r * T_t) * norm.cdf(call_put * (d - v * np.sqrt(T_t)))
        
        # Check if any of S, K, T_t or v are 0
        elif S == 0 or K == 0 or T_t == 0 or v == 0:
        
            # Set option price equal to ...
            option_price = max(call_put * (S * np.exp(-dividend_yield * T_t) - K * np.exp(-r * T_t)), 0)

        # Catch invalid option parameters
        else:
            raise ValueError('Invalid option parameters')
        
        # Append option price to list of option prices
        option_prices.append(option_price)

    return option_prices

def rmse(actuals, error_type='absolute', **kwargs):
    '''
    Compute root mean squared error.

    Parameters
    ---
    actuals : int or float
        Actual value.
    predicted : int or float
        Predicted value.
    error_type : str, default 'absolute'
        Type of error, one of 'absolute' or 'relative'

    Returns
    ---
    error :
        Root mean squared error.
    '''
    # Initialise list of errors
    errors = []

    # Compute Black-Scholes option prices
    predictions = black_scholes(**kwargs)

    # Compute error for each option
    for actual, prediction in zip(actuals, predictions):

        # Compute absolute error
        error = prediction - actual
        
        # For relative error type transform into relative terms
        if error_type == 'absolute':
            pass
        elif error_type == 'relative':
            error = error / actual
        else:
            raise ValueError('Invalid error type')
        
        errors.append(error)
    
    # Compute root mean squared error
    error = np.sqrt(np.mean(error**2))
    
    return error

def implied_volatility(error_function=rmse, starting_value=0.2, method='Nelder-Mead', bounds=(0, None), **kwargs):
    '''
    Compute option-implied volatility.

    Parameters
    ---
    error_function : callable, default rmse
        Function for computing the error that is to be minimised.
    starting_value : int or float, default 0
        Starting value of the estimator.
    
    Returns
    ---
    iv : float
        Option implied volatility.
    '''
    
    # Optimise function by minimising error
    optimisation_result = minimize(lambda x: error_function(v=x, **kwargs), starting_value, method=method, bounds=[bounds])
    
    # Extract implied volatility
    iv = optimisation_result.x[0]

    return iv

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

    # Define product parameters
    NOMINAL_VALUE = 1000
    MATURITY_IN_YEARS = 1
    ANNUAL_COUPON_RATE = 0.085
    KICK_IN_LEVEL = 0.65
    COUPON_RATE = 0.085
    COUPON_FREQUENCY = 2
    pricing_date = pd.to_datetime('2018-06-27')
    expiration_date = pd.to_datetime('2019-06-20')
    strike_prices = stock_prices.loc[pricing_date]
    KICK_IN_LEVEL = 0.65
    barrier_levels = strike_prices * KICK_IN_LEVEL
    conversion_ratios = round(NOMINAL_VALUE / strike_prices, 4)

    # These should be computed in code
    I_RATE = 0.01
    volatilities = pd.DataFrame([[0.2, 0.25, 0.15]], columns=stock_prices.columns)

    # Add stock price to option data
    stock_prices_long = stock_prices.melt(var_name='underlying', value_name='underlying_price', ignore_index=False)
    option_prices = pd.merge(option_prices, stock_prices_long, on=['date','underlying'])

    # Add risk free rate to option data
    option_prices['rate'] = 0.1

    # Add dividend yield to option data
    option_prices['dividend_yield'] = 0

    option_prices['volatility'] = option_prices.groupby('exdate').apply(lambda option: implied_volatility(
        error_function=rmse, starting_value=0.2, method='Nelder-Mead', bounds=(0, None), # kwargs for implied_volatility()
        actuals=option.option_price, error_type='absolute', # kwargs for rmse()
        S=option.underlying_price, K=option.strike, t=option.index, T=option.name, r=option.rate, dividend_yield=option.dividend_yield, option_type=option.option_type)) # kwargs for black_scholes()

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

    # # Get ATM implied volatility for each stock
    # adobe_iv = get_atm_iv(atm_options, 'Adobe')
    # apple_iv = get_atm_iv(atm_options, 'Apple')
    # microsoft_iv = get_atm_iv(atm_options, 'Microsoft')

    # Calculate bond price
    bond_price = bond(NOMINAL_VALUE, COUPON_RATE, COUPON_FREQUENCY, MATURITY_IN_YEARS, i_rate, credit_spread)

    # Calcualte worst-of down-and-in put option price
    option_price = worst_of_down_and_in_put(barrier_levels, strike_prices, conversion_ratios,
                             stock_prices, volatilities, # *args (only inner)
                             pricing_date=pricing_date, expiration_date=expiration_date, i_rate=I_RATE,
                             exchange='SIX' # **kwargs (only inner)
                             )
    
    # Calcualte structure product price (short the option)
    product_value = bond_price - option_price

    # Calculate the issue premium
    issue_premium = (NOMINAL_VALUE - product_value) / product_value

    # Sensitivity analysis
    # discount rate : risk free rate
    # correlations : 
    # and volatility :