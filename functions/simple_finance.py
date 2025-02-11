import zipfile
from io import BytesIO
import pandas as pd
import yfinance as yf
import requests
import io
import ssl
import urllib.request
import certifi
import urllib.request
import os
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import urllib3
import statsmodels.api as sm
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
def get_ff3():
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
    ff_3 = ff_three_factors[ff_three_factors['Unnamed: 0'].str.match(r'^\d{6}$', na=False)] # gets rid of Copyright line
    ff_three_factors=ff_3.copy()
    ff_three_factors = ff_three_factors.astype(float)

    # Rename the first column to 'Date'
    ff_three_factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    # Convert the Date column to datetime format (YYYY-MM)
    ff_three_factors['Date'] = pd.to_datetime(ff_three_factors['Date'], format='%Y%m').dt.to_period()

    # Convert all columns except 'Date' to numeric types
    ff_three_factors.iloc[:, 1:] = ff_three_factors.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Multiply all columns except 'Date' by 0.01
    ff_three_factors.iloc[:, 1:] = (ff_three_factors.iloc[:, 1:] * 0.01)

    # reset the index
    ff_three_factors.set_index('Date', inplace=True)

    return ff_three_factors


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
