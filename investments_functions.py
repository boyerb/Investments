import pandas as pd
import yfinance as yf
import requests
import io

"""
This functions downloads and processes the Fama-French 5-Factor data from the Dartmouth website using the 'requests' library. 
"""
def get_ff5():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

    # Read the zipped CSV file from the URL
    ff_five_factors = pd.read_csv(url, skiprows=3)

    # Use str.match to identify rows with dates (6-digit YYYYMM format), and drop NaN values that could arise
    ff_five_factors = ff_five_factors[ff_five_factors['Unnamed: 0'].str.match(r'^\d{6}$', na=False)]

    # Rename the first column to 'Date'
    ff_five_factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    # Convert the Date column to datetime format (YYYY-MM)
    ff_five_factors['Date'] = pd.to_datetime(ff_five_factors['Date'], format='%Y%m').dt.to_period('M')

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

def get_ff3():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

    # Read the zipped CSV file from the URL
    ff_three_factors = pd.read_csv(url, skiprows=3)

    # Use str.match to identify rows with dates (6-digit YYYYMM format), and drop NaN values that could arise
    ff_three_factors = ff_three_factors[ff_three_factors['Unnamed: 0'].str.match(r'^\d{6}$', na=False)]

    # Rename the first column to 'Date'
    ff_three_factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    # Convert the Date column to datetime format (YYYY-MM)
    ff_three_factors['Date'] = pd.to_datetime(ff_three_factors['Date'], format='%Y%m').dt.to_period('M)

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
        data = yf.download(ticker, start=adjusted_start_date, end=end_date, interval="1d")

        # Resample data to get the last business day of each month
        month_end_data = data.resample('M').ffill()  # 'M' gives month-end, ffill to get last available price

        # Calculate monthly returns based on month-end data
        month_end_data['Monthly Return'] = month_end_data['Adj Close'].pct_change()

        # Drop any missing values (the first row will have NaN for returns)
        month_end_data.dropna(inplace=True)

        # Add returns to the main DataFrame with ticker as column name
        all_returns[ticker] = month_end_data['Monthly Return']

    all_returns.index = all_returns.index.to_period('M')

    if tbill_return:
        ff3 = get_ff3()
        all_returns = all_returns.join(ff3['RF'])

    return all_returns