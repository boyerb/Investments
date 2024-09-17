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

    return ff_five_factors
####################################################################################################

def get_monthly_returns(ticker):
    # Download daily data from Yahoo Finance
    data = yf.download(ticker, start="2000-01-01", end="2023-12-31", interval="1d")

    # Resample data to get the last business day of each month
    month_end_data = data.resample('ME').ffill()  # 'M' gives month-end, ffill to get last available price

    # Calculate monthly returns based on month-end data
    month_end_data['Monthly Return'] = month_end_data['Adj Close'].pct_change()

    # Drop any missing values (the first row will have NaN for returns)
    month_end_data.dropna(inplace=True)

    return month_end_data