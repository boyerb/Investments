import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import investments_functions as invf
import pandas as pd

ydat = invf.get_monthly_returns('AAPL')


fdat=invf.get_ff5()

# URL for the five-factor data
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

# Read the zipped CSV file from the URL
ff_five_factors = pd.read_csv(url, skiprows=3)

# Use str.match to identify rows with dates (6-digit YYYYMM format), and drop NaN values that could arise
ff_five_factors = ff_five_factors[ff_five_factors['Unnamed: 0'].str.match(r'^\d{6}$', na=False)]

# Rename the first column to 'Date'
ff_five_factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Convert the Date column to datetime format (YYYY-MM)
ff_five_factors['Date'] = pd.to_datetime(ff_five_factors['Date'], format='%Y%m').dt.to_period('M')

# Print the first few rows to verify
print(ff_five_factors.head())





f_data = invf.get_ff5()

print(f_data)