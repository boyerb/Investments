import zipfile
import pandas as pd
import yfinance as yf
import requests
import io
import ssl
import urllib.request
import certifi
import urllib.request
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
from io import BytesIO
"""
This functions downloads and processes the Fama-French 5-Factor data from the Dartmouth website using the 'requests' library. 
"""
def get_ff5():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

    # Read the zipped CSV file from the URL
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response:
        ff_five_factors = pd.read_csv(url, skiprows=3)

    # Use str.match to identify rows with dates (6-digit YYYYMM format), and drop NaN values that could arise
    ff_five_factors = ff_five_factors[ff_five_factors['Unnamed: 0'].str.match(r'^\d{6}$', na=False)]

    # Rename the first column to 'Date'
    ff_five_factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    # Convert the Date column to datetime format (YYYY-MM)
    ff_five_factors['Date'] = pd.to_datetime(ff_five_factors['Date'], format='%Y%m')

    # Convert all columns except 'Date' to numeric types
    ff_five_factors.iloc[:, 1:] = ff_five_factors.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Multiply all columns except 'Date' by 0.01
    ff_five_factors.iloc[:, 1:] = ff_five_factors.iloc[:, 1:] * 0.01

    # reset the index
    ff_five_factors.set_index('Date', inplace=True)

    return ff_five_factors

"""
This functions downloads and processes the Fama-French 3-Factor data from the Dartmouth website using the 'requests' library. 
"""
#########################################################################################################################
def get_ff3(start_date=None, end_date=None):
    #http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
    #http.request("GET", "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")

    # Make the request using the session
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    response = requests.get(url)

    # Read the content of the file
    zip_content = response.content

    # Open the zip file from the content
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        with zf.open('F-F_Research_Data_Factors.CSV') as f:
            # Read the CSV file content (you can load it into pandas or process as needed)
            ff_three_factors = pd.read_csv(f, skiprows=3)

    # Use str.match to identify rows with dates (6-digit YYYYMM format), and drop NaN values that could arise
   # start_index = ff_three_factors[ff_three_factors.iloc[:, 0].dropna().str.contains("Annual Factors: January-December")].index[0]
    start_index = ff_three_factors[ff_three_factors.iloc[:, 0].str.contains("Annual Factors: January-December", na=False)].index[0]
    ff3_1 = ff_three_factors.iloc[:start_index]

    ff3_2=ff3_1.copy()
    ff3_2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

    # Convert first column to Period
    ff3_2['date'] = pd.to_datetime(ff3_2['date'].astype(str), format='%Y%m').dt.to_period()
    ff3_2.iloc[:, 1:] = ff3_2.iloc[:, 1:].astype(float) * 0.01

    # reset the index
    ff3_2.set_index('date', inplace=True)

    # Apply date range filtering if provided
    if start_date is not None:
        ff3_2 = ff3_2[ff3_2.index >= pd.Period(start_date, freq='M')]
    if end_date is not None:
        ff3_2 = ff3_2[ff3_2.index <= pd.Period(end_date, freq='M')]

    return ff3_2
#########################################################################################################################
def get_ff_beta_portfolios(dtype, start_date=None, end_date=None):
    #http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
    #http.request("GET", "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")

    # Make the request using the session
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_BETA_CSV.zip"
    response = requests.get(url)

    # Read the content of the file
    zip_content = response.content

    # Open the zip file from the content
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        with zf.open('Portfolios_Formed_on_BETA.csv') as f:
            # Read the CSV file content (you can load it into pandas or process as needed)
            dat = pd.read_csv(f, skiprows=15, header=0)

    start_index = dat[dat.iloc[:, 0].str.contains("Equal Weighted Returns -- Monthly", na=False)].index[0]
    dat1 = dat.iloc[:start_index]
    dat2=dat1.copy()
    dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

    # Convert first column to Period
    dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()
    dat2.iloc[:, 1:] = dat2.iloc[:, 1:].astype(float) * 0.01

    # reset the index
    dat2.set_index('date', inplace=True)

    if dtype == 'quintiles':
        cols = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20']
    elif dtype == 'deciles':
        cols = ['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5',
                'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']
    else:
        raise ValueError("Invalid data type. Choose 'quintiles' or 'deciles'.")

    dat3 = dat2[cols].copy()  # Select relevant columns
    dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)

    # Apply date range filtering if provided
    if start_date is not None:
        dat3 = dat3[dat3.index >= pd.Period(start_date, freq='M')]
    if end_date is not None:
        dat3 = dat3[dat3.index <= pd.Period(end_date, freq='M')]

    ff3=get_ff3()
    ff3['mkt']=ff3['Mkt-RF']+ff3['RF']
    ff3.rename(columns={'RF':'rf'},inplace=True)
    dat3=pd.merge(dat3,ff3[['mkt','rf']],left_index=True,right_index=True,how='left')
    return dat3
####################################################################################################
def get_ff_strategies(stype, start_date=None, end_date=None, details=None):

    if stype == 'beta':
        if details is True:
            print("----------------")
            print("Beta Strategy")
            print("----------------")
            print(print("Basic Strategy: stocks are sorted into deciles based on their historical betas."))
            print()
            print("Construction: The portfolios are formed on univariate market beta at the end of each June using NYSE breakpoints.")
            print("Beta for June of year t is estimated using the preceding five years (two minimum) of past monthly returns.")
            print()

            print("Stocks: All NYSE, AMEX, and NASDAQ stocks for which we have market equity data for June of t and good returns for the preceding 60 months (24 months minimum)."
                  )
            return

        # Make the request using the session
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_BETA_CSV.zip"
        response = requests.get(url)

        # Read the content of the file
        zip_content = response.content

        # Open the zip file from the content
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            with zf.open('Portfolios_Formed_on_BETA.csv') as f:
                # Read the CSV file content (you can load it into pandas or process as needed)
                dat = pd.read_csv(f, skiprows=15, header=0)

        start_index = dat[dat.iloc[:, 0].str.contains("Equal Weighted Returns -- Monthly", na=False)].index[0]
        dat1 = dat.iloc[:start_index]
        dat2=dat1.copy()
        dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

        # Convert first column to Period
        dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()
        dat2.iloc[:, 1:] = dat2.iloc[:, 1:].astype(float) * 0.01

        # reset the index
        dat2.set_index('date', inplace=True)
        dat2.rename(columns={'Lo 10': 'Dec 1', 'Hi 10': 'Dec 10'}, inplace=True)
        cols = ['Dec 1', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5',
                'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Dec 10']

        dat3 = dat2[cols].copy()  # Select relevant columns
        dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)

    #-------------------------------------------
    elif stype == "momentum":
        if details is True:
            print("----------------")
            print("Momentum Strategy")
            print("----------------")
            print("Basic Strategy: stocks are sorted into deciles based on their prior 12-month returns, excluding the most recent month.")
            print()
            print("Construction: The portfolios are constructed monthly using NYSE prior (2-12) return decile breakpoints.")
            print()
            print("Stocks: The portfolios constructed each month include NYSE, AMEX, and NASDAQ stocks with prior return data.")
            print("To be included in a portfolio for month t (formed at the end of month t-1), a stock must have a price for the")
            print("end of month t-13 and a good return for t-2. In addition, any missing returns from t-12 to t-3 must be -99.0,")
            print("CRSP's code for a missing price. Each included stock also must have ME for the end of month t-1.")

            return
        # Continue the momentum strategy implementation below

        # Make the request using the session
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_CSV.zip"
        response = requests.get(url)

        # Read the content of the file
        zip_content = response.content

        # Open the zip file from the content
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            with zf.open('10_Portfolios_Prior_12_2.CSV') as f:
                # Read the CSV file content (you can load it into pandas or process as needed)
                dat = pd.read_csv(f, skiprows=10, header=0)



        start_index = dat[dat.iloc[:, 0].str.contains("  Average Equal Weighted Returns -- Monthly", na=False)].index[0]
        dat1 = dat.iloc[:start_index]
        dat2 = dat1.copy()
        dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        dat2.rename(columns={'Lo PRIOR': 'Dec 1', 'PRIOR 2': 'Dec 2', 'PRIOR 3': 'Dec 3', 'PRIOR 4': 'Dec 4'}, inplace=True)
        dat2.rename(columns={'PRIOR 5': 'Dec 5', 'PRIOR 6': 'Dec 6', 'PRIOR 7': 'Dec 7', 'PRIOR 8': 'Dec 8'}, inplace=True)
        dat2.rename(columns={'PRIOR 9': 'Dec 9', 'Hi PRIOR': 'Dec 10'}, inplace=True)

        # Convert first column to Period
        dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()
        dat2.iloc[:, 1:] = dat2.iloc[:, 1:].astype(float) * 0.01

        # reset the index
        dat2.set_index('date', inplace=True)
        dat3=dat2.copy()  # Select relevant columns
        dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)

    else:
        raise ValueError("Invalid strategy type. Choose 'beta' or 'momentum'.")

    #------------------------------------------
    # Apply date range filtering if provided
    if start_date is not None:
        dat3 = dat3[dat3.index >= pd.Period(start_date, freq='M')]
    if end_date is not None:
        dat3 = dat3[dat3.index <= pd.Period(end_date, freq='M')]

    ff3=get_ff3()
    ff3['mkt']=ff3['Mkt-RF']+ff3['RF']
    ff3.rename(columns={'RF':'rf'},inplace=True)
    dat_final=pd.merge(dat3,ff3[['mkt','rf']],left_index=True,right_index=True,how='left')
    return dat_final




####################################################################################################
"""
    This function fetches monthly returns for a given ticker over a specified date range.

    Parameters:
    - ticker (str): The ticker symbol of the stock.
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame: A DataFrame with monthly returns and adjusted close prices.
"""
def get_monthly_returns(tickers, start_date, end_date, tbill_return=True):
    all_returns = pd.DataFrame()
    adjusted_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    for ticker in tickers:
        # Download daily data from Yahoo Finance
        data = yf.download(ticker, start=adjusted_start_date, end=end_date, interval="1d", auto_adjust=False)
        data['date'] = data.index

        # Resample data to get the last business day of each month
        month_end_data = data.groupby([data.index.year, data.index.month]).apply(lambda x: x.loc[x.index.max()])
        month_end_data = month_end_data.set_index('date')

        # Calculate monthly returns based on month-end data
        month_end_data = month_end_data.sort_index()
        month_end_data['Monthly Return'] = month_end_data['Adj Close'].pct_change()


        # Drop any missing values (the first row will have NaN for returns)
        month_end_data.dropna(inplace=True)

        all_returns[ticker] = month_end_data['Monthly Return']

    if tbill_return:
        ff3 = get_ff3()
        all_returns['YearMonth'] = all_returns.index.to_period('M')
        all_returns = pd.merge(all_returns, ff3[['RF']], left_on='YearMonth', right_index=True, how='left')
        all_returns.drop('YearMonth', axis=1, inplace=True)

    return all_returns

####################################################################################################
def intercept(y,x):
    x_with_intercept = sm.add_constant(x)

    # Fit the model: y = β0 + β1*x
    model = sm.OLS(y, x_with_intercept)
    results = model.fit()

    # Return the slope (coefficient of x)
    return results.params[0]  # The slope is the second parameter (after the intercept)


####################################################################################################
def slope(y,x):
    x_with_intercept = sm.add_constant(x)

    # Fit the model: y = β0 + β1*x
    model = sm.OLS(y, x_with_intercept)
    results = model.fit()

    # Return the slope (coefficient of x)
    return results.params[1]  # The slope is the second parameter (after the intercept)
####################################################################################################
def portfolio_volatility(weights, covariance_matrix):
    return np.sqrt(weights.T @ covariance_matrix @ weights)  # Essential Math Fact #5
####################################################################################################
def EFRS_portfolio(target_return, expected_returns, covariance_matrix):

    # Define constraints (tuple with two dictionaries)
    constraints = (# Constraint 1: Ensure portfolio return equals the target return
        # "type": "eq" specifies an equality constraint (must be exactly zero)
        # "fun": defines the function
        {"type": "eq", "fun": lambda w: w.T @ expected_returns - target_return},

        # Constraint 2: Ensure portfolio weights sum to 1 (fully invested)
        # "type": "eq" specifies an equality constraint (must be exactly zero)
        # "fun": defines the function
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    # starting values
    N = expected_returns.shape[0]
    initial_weights = np.ones(N) / N

    # Python's version of Excel Solver
    # the function to minimize is portfolio_volatility
    # the variables to change are defined by the first input to the function
    # in this case the first argument is the vector of portfolio weights
    result = minimize(fun=portfolio_volatility,  # function to minimize
        x0=initial_weights,  # starting values
        args=(covariance_matrix),  # additional arguments needed for function
        method="SLSQP",  # minimization method
        constraints=constraints,  # constraints
    )

    optimal_weights=result.x
    Ereturn = optimal_weights.T @ expected_returns
    volatility = portfolio_volatility(optimal_weights, covariance_matrix)

    return result.x, Ereturn, volatility

####################################################################################################
def portfolio_sharpe(weights: np.ndarray, expected_returns: np.ndarray,
                     covariance_matrix: np.ndarray, rf: float) -> float:
    """
    Computes the Sharpe ratio of a portfolio given the asset weights, expected returns,
    covariance matrix, and risk-free rate.

    Parameters:
    -----------
    weights : np.ndarray
        A 1D NumPy array of portfolio weights (shape: `(n_assets,)`).
    expected_returns : np.ndarray
        A 1D NumPy array of expected returns for each asset (shape: `(n_assets,)`).
    covariance_matrix : np.ndarray
        A 2D NumPy array representing the covariance matrix of asset returns (shape: `(n_assets, n_assets)`).
    rf : float
        The risk-free rate.

    Returns:
    --------
    float
        The Sharpe ratio of the portfolio.

    Raises:
    -------
    TypeError:
        If any of the inputs are not NumPy arrays.
    ValueError:
        If the dimensions of inputs do not match.
    """

    # Enforce that all inputs are NumPy arrays
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy array")
    if not isinstance(expected_returns, np.ndarray):
        raise TypeError("expected_returns must be a NumPy array")
    if not isinstance(covariance_matrix, np.ndarray):
        raise TypeError("covariance_matrix must be a NumPy array")

    # Ensure correct dimensions
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D vector")
    if expected_returns.ndim != 1:
        raise ValueError("expected_returns must be a 1D vector")
    if covariance_matrix.ndim != 2:
        raise ValueError("covariance_matrix must be a 2D array")

    # Get the number of assets from expected returns and weights
    n_assets = expected_returns.shape[0]

    # Check that weights and expected returns have the same length
    if weights.shape[0] != n_assets:
        raise ValueError(f"weights and expected_returns must have the same length ({n_assets})")

    # Check that the covariance matrix is square and matches the number of assets
    if covariance_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"covariance_matrix must be a square matrix of shape ({n_assets}, {n_assets})")

    # Compute portfolio return as the weighted sum of expected asset returns
    port_ret = weights.T @ expected_returns

    # Compute portfolio volatility using the given covariance matrix
    port_vol = portfolio_volatility(weights, covariance_matrix)

    # Compute and return the Sharpe ratio (excess return divided by risk)
    return (port_ret - rf) / port_vol

####################################################################################################
def tangent_portfolio(expected_returns, covariance_matrix, rf):
    """
    Calculates the weights, expected return, and volatility of the tangent portfolio.

    Parameters:
    expected_returns (np.array): Vector of expected returns for each asset.
    covariance_matrix (np.array): Covariance matrix of asset returns.
    rf (float): Risk-free rate of return.

    Returns:
    tuple: A tuple containing the tangent portfolio weights, expected return, and volatility.
    """
    N = expected_returns.shape[0]
    initial_weights = np.ones(N) / N  # Initialize as a 1D column vector
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Constraint: weights sum to 1

    # Define a lambda function to negate the output of portfolio_sharpe
    neg_portfolio_sharpe = lambda x, expected_returns, covariance_matrix, rf: -portfolio_sharpe(x, expected_returns, covariance_matrix, rf)

    # Perform optimization
    result = minimize(fun=neg_portfolio_sharpe, x0=initial_weights, args=(expected_returns, covariance_matrix, rf),
                      method="SLSQP", constraints=constraints)

    # Ensure result.x is reshaped as a column vector (N x 1)
    tangent_weights = result.x

    # Compute expected return and volatility for the tangent portfolio
    tangent_return = tangent_weights.T @ expected_returns
    tangent_volatility = portfolio_volatility(tangent_weights, covariance_matrix)

    return tangent_weights, tangent_return, tangent_volatility
