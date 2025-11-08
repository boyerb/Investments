import zipfile
import pandas as pd
import requests
import io
import ssl
import urllib.request
import certifi
import urllib.request
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
from typing import Iterable, List, Union

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width to fit the output
pd.set_option('display.max_colwidth', None)  # Show full column content without truncation

##############################################################################################################################
def get_crsp_msf_by_ids(
    db,
    identifiers: Iterable[Union[str, int]],
    start_date: str,
    end_date: str,
    chunk_size: int = 500
) -> pd.DataFrame:

    """
    Pull CRSP Monthly Stock File (msf) data for either a list of TICKERS or PERMNOs.

    Parameters
    ----------
    identifiers : iterable of str|int
        Either all tickers (e.g., ['AAPL','MSFT']) or all PERMNOs (e.g., [14593, 12079]).
        Mixed types are not allowed.
    start_date, end_date : 'YYYY-MM-DD'
        Inclusive date range to filter `a.date`.
    chunk_size : int
        Max identifiers per SQL IN() chunk to avoid overly long queries.

    Returns
    -------
    pandas.DataFrame
        Columns: date, permno, ticker, comnam, shrcd, exchcd, siccd, prc, ret, retx, vol, shrout
    """
    # --- Normalize date inputs ---
    try:
        start = pd.Period(start_date, freq="M").to_timestamp(how="start")
        end = pd.Period(end_date, freq="M").to_timestamp(how="end")
    except Exception:
        raise ValueError("start_date and end_date must be in 'YYYY-MM' format.")

    # Normalize identifiers and auto-detect type
    ids_list: List[Union[str, int]] = list(identifiers)
    if not ids_list:
        raise ValueError("identifiers list is empty.")

    # If everything looks numeric (ints or numeric strings), treat as PERMNOs; else, tickers.
    def is_numeric_like(x):
        if isinstance(x, int):
            return True
        if isinstance(x, str):
            return x.strip().isdigit()
        return False

    all_numeric_like = all(is_numeric_like(x) for x in ids_list)
    if all_numeric_like:
        id_type = "permno"
        # Cast all to int
        ids_list = [int(x) for x in ids_list]
    else:
        id_type = "ticker"
        # Upper-case tickers and strip spaces
        ids_list = [str(x).strip().upper() for x in ids_list]
        # Basic sanity: forbid quotes to keep the simple IN (...) builder safe
        if any("'" in t or '"' in t for t in ids_list):
            raise ValueError("Tickers must not contain quotes.")

    # Base SELECT/JOIN and date validity join to msenames
    base_sql = f"""
        SELECT 
            a.date, 
            a.permno,
            a.permco, 
            b.ticker, 
            b.comnam, 
            b.shrcd, 
            b.exchcd, 
            b.siccd, 
            a.prc, 
            a.ret, 
            a.retx, 
            a.vol, 
            a.shrout
        FROM crspm.msf a
        INNER JOIN crspm.msenames b
            ON a.permno = b.permno
        WHERE a.date >= b.namedt 
          AND a.date <= b.nameendt
          AND a.date BETWEEN '{start:%Y-%m-%d}' AND '{end:%Y-%m-%d}'
    """

    # Build WHERE clause chunks
    def chunk(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    dfs = []

    for sub in chunk(ids_list, chunk_size):
        if id_type == "permno":
            # numeric IN list
            id_str = ",".join(str(x) for x in sub)
            where_ids = f" AND a.permno IN ({id_str})"
        else:
            # quoted text IN list
            id_str = ",".join(f"'{x}'" for x in sub)
            where_ids = f" AND b.ticker IN ({id_str})"

        sql = base_sql + where_ids
        dfs.append(db.raw_sql(sql))

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    # Optional: sort the result
    if not out.empty:
        out = out.sort_values(["permno", "date"]).reset_index(drop=True)
        out['date'] = pd.to_datetime(out['date'])  # ensure that date is a datetime object
        out['date'] = out['date'].dt.to_period("M")
        out = out.set_index('date')
    return out

##############################################################################################################################
def format_alpha_vantage(r, start_date=None, end_date=None):
    """
    Convert an Alpha Vantage monthly adjusted time series API response
    into a clean Pandas DataFrame.

    Parameters
    ----------
    r : requests.Response
        The HTTP response object returned by requests.get(url).

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with datetime index (month-end dates) and
        columns: Open, High, Low, Close, Adjusted Close, Volume, Dividend Amount.
    """

    # Step 1: Convert the JSON response body into a Python dictionary.
    # At this point, 'data' will be a nested dictionary containing
    # both metadata and the monthly adjusted time series.
    data = r.json()

    # Step 2: Access the part of the API response corresponding
    # specifically to the monthly adjusted stock data.
    # This section uses dates as keys and stores open, high, low, close,
    # adjusted close, volume, and dividend info for each month.
    ts_data = data["Monthly Adjusted Time Series"]

    # Step 3: Convert the time series dictionary into a Pandas DataFrame.
    # Using orient="index" tells Pandas to use the dictionary keys (dates)
    # as the DataFrame index, so each row corresponds to one month.
    df = pd.DataFrame.from_dict(ts_data, orient="index")

    # Step 4: Rename columns to something cleaner and easier to read.
    df.columns = [
        "Open", "High", "Low", "Close",
        "Adjusted Close", "Volume", "Dividend Amount"
    ]

    # Step 5: Convert string values into numeric floats.
    # JSON encodes all numbers as strings, so they must be converted
    # for analysis, plotting, and calculations.
    df = df.astype(float)

    # Step 6: Convert the index to datetime and sort chronologically.
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M")
    df = df.sort_index()

    # Step 7: Apply optional date filtering.
    if start_date is not None:
        df = df[df.index >= pd.Period(start_date, freq="M")]
    if end_date is not None:
        df = df[df.index <= pd.Period(end_date, freq="M")]

    df.index.name = "date"

    return df

##############################################################################################################################
"""
This functions downloads and processes the Fama-French 5-Factor data from the Dartmouth website using the 'requests' library. 
"""
import io, zipfile, requests
import pandas as pd

def get_ff5(start_date=None, end_date=None):
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    r = requests.get(url)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open("F-F_Research_Data_5_Factors_2x3.csv") as f:
            raw = pd.read_csv(f, skiprows=3)

    # find where monthly section ends (right before the "Annual Factors..." line)
    stop = raw[raw.iloc[:, 0].astype(str).str.contains("Annual Factors: January-December", na=False)].index[0]
    df = raw.iloc[:stop].copy()

    # normalize first column name and convert to monthly PeriodIndex
    first_col = df.columns[0]
    df.rename(columns={first_col: "date"}, inplace=True)
    df["date"] = pd.PeriodIndex(df["date"].astype(str), freq="M")

    # convert remaining columns to numeric and scale from % to decimal
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.iloc[:, 1:] = df.iloc[:, 1:] * 0.01

    # set index, drop any non-date rows, sort
    df.set_index("date", inplace=True)
    df = df.dropna(how="all")  # drop any junk rows if present
    df = df.sort_index()

    # helper: coerce any input to a monthly Period
    def _to_period_m(x):
        if x is None:
            return None
        if isinstance(x, pd.Period):
            return x.asfreq("M")
        # accept 'YYYY-MM', 'YYYYMM', datetime, etc.
        return pd.Period(pd.to_datetime(str(x)).strftime("%Y-%m"), freq="M")

    start_p = _to_period_m(start_date)
    end_p   = _to_period_m(end_date)

    # clean slicing by Period
    if start_p or end_p:
        df = df.loc[start_p:end_p]

    return df

"""
This functions downloads and processes the Fama-French 3-Factor data from the Dartmouth website using the 'requests' library. 
"""
#########################################################################################################################
def get_ff3(start_date=None, end_date=None):

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    response = requests.get(url)

    # Read the content of the file
    zip_content = response.content

    # Open the zip file from the content
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        with zf.open('F-F_Research_Data_Factors.csv') as f:
            # Read the CSV file content (you can load it into pandas or process as needed)
            ff_three_factors = pd.read_csv(f, skiprows=3)

    start_index = ff_three_factors[ff_three_factors.iloc[:, 0].str.contains("Annual Factors: January-December", na=False)].index[0]
    ff3_1 = ff_three_factors.iloc[:start_index]

    ff3_2=ff3_1.copy()
    ff3_2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

    # Convert first column to Period
    ff3_2['date'] = pd.to_datetime(ff3_2['date'].astype(str), format='%Y%m').dt.to_period()
    # Convert all columns except 'date' to numeric types
    for col in ff3_2.columns[1:]:  # Skip the 'date' column
        ff3_2[col] = pd.to_numeric(ff3_2[col], errors='coerce')

    ff3_2.iloc[:, 1:] = ff3_2.iloc[:, 1:] * 0.01

    # reset the index
    ff3_2.set_index('date', inplace=True)

    # Apply date range filtering if provided
    if start_date is not None:
        ff3_2 = ff3_2[ff3_2.index >= pd.Period(start_date, freq='M')]
    if end_date is not None:
        ff3_2 = ff3_2[ff3_2.index <= pd.Period(end_date, freq='M')]

    return ff3_2
####################################################################################################
def get_ff_strategies(stype, start_date=None, end_date=None, details=None):

    if stype == 'beta':

        # Make the request using the session
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_BETA_csv.zip"
        response = requests.get(url)

        # Read the content of the file
        zip_content = response.content

        # Open the zip file from the content
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            with zf.open('Portfolios_Formed_on_BETA.csv') as f:
                # Read the CSV file content (you can load it into pandas or process as needed)
                dat = pd.read_csv(f, skiprows=15, header=0, encoding='utf-8', skipfooter=5, engine='python')

        start_index = dat[dat.iloc[:, 0].str.contains("Equal Weighted Returns -- Monthly", na=False)].index[0]
        dat1 = dat.iloc[:start_index]
        dat2 = dat1.copy()
        dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

        # Convert first column to Period
        dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()

        # Convert all columns except 'date' to numeric types
        for col in dat2.columns[1:]:  # Skip the 'date' column
            dat2[col] = pd.to_numeric(dat2[col], errors='coerce')

        dat2.iloc[:, 1:] = dat2.iloc[:, 1:] * 0.01

        # reset the index
        dat2.set_index('date', inplace=True)
        dat2.rename(columns={'Lo 10': 'Dec 1', 'Hi 10': 'Dec 10'}, inplace=True)
        cols = ['Dec 1', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5',
                'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Dec 10']

        dat3 = dat2[cols].copy()  # Select relevant columns
        dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)
        if details is True:
            print("----------------")
            print("Beta Strategy")
            print("----------------")
            print("Basic Strategy: stocks are sorted into deciles based on their historical betas.")
            print()
            print("Construction: The portfolios are formed on univariate market beta at the end of each June using NYSE breakpoints.")
            print("Beta for June of year t is estimated using the preceding five years (two minimum) of past monthly returns.")
            print()

            print("Stocks: All NYSE, AMEX, and NASDAQ stocks for which we have market equity data for June of t and good returns for the preceding 60 months (24 months minimum)."
                  )
            min_date = dat3.index.min()
            max_date = dat3.index.max()
            print()
            print(f"Min Date: {min_date}, Max Date: {max_date}")


    #-------------------------------------------
    elif stype == "momentum":


        # Continue the momentum strategy implementation below

        # Make the request using the session
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_csv.zip"
        response = requests.get(url)

        # Read the content of the file
        zip_content = response.content

        # Open the zip file from the content
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            with zf.open('10_Portfolios_Prior_12_2.csv') as f:
                # Read the CSV file content (you can load it into pandas or process as needed)
                dat = pd.read_csv(f, skiprows=10, header=0)



        start_index = dat[dat.iloc[:, 0].str.contains("Average Equal Weighted Returns -- Monthly", na=False)].index[0]
        dat1 = dat.iloc[:start_index]
        dat2 = dat1.copy()
        dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        dat2.rename(columns={'Lo PRIOR': 'Dec 1', 'PRIOR 2': 'Dec 2', 'PRIOR 3': 'Dec 3', 'PRIOR 4': 'Dec 4'}, inplace=True)
        dat2.rename(columns={'PRIOR 5': 'Dec 5', 'PRIOR 6': 'Dec 6', 'PRIOR 7': 'Dec 7', 'PRIOR 8': 'Dec 8'}, inplace=True)
        dat2.rename(columns={'PRIOR 9': 'Dec 9', 'Hi PRIOR': 'Dec 10'}, inplace=True)

        # Convert first column to Period
        dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()
        for col in dat2.columns[1:]:  # Skip the 'date' column
            dat2[col] = pd.to_numeric(dat2[col], errors='coerce')

        dat2.iloc[:, 1:] = dat2.iloc[:, 1:] * 0.01

        # reset the index
        dat2.set_index('date', inplace=True)
        dat3=dat2.copy()  # Select relevant columns
        dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)
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
            min_date = dat3.index.min()
            max_date = dat3.index.max()
            print()
            print(f"Min Date: {min_date}, Max Date: {max_date}")


    #------------------------------------------
    elif stype == 'shorttermreversal':

        # Make the request using the session
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_1_0_csv.zip"
        response = requests.get(url)

        # Read the content of the file
        zip_content = response.content

        # Read the content of the file


        # Open the zip file from the content
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            with zf.open('10_Portfolios_Prior_1_0.csv') as f:
                # Read the CSV file content (you can load it into pandas or process as needed)
                dat = pd.read_csv(f, skiprows=10, header=0, encoding='utf-8', skipfooter=5, engine='python')

        start_index = dat[dat.iloc[:, 0].str.contains("Equal Weighted Returns -- Monthly", na=False)].index[0]
        dat1 = dat.iloc[:start_index]
        dat2 = dat1.copy()
        dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

        # Convert first column to Period
        dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()

        # Convert all columns except 'date' to numeric types
        for col in dat2.columns[1:]:  # Skip the 'date' column
            dat2[col] = pd.to_numeric(dat2[col], errors='coerce')

        dat2.iloc[:, 1:] = dat2.iloc[:, 1:] * 0.01

        # reset the index
        dat2.set_index('date', inplace=True)
        dat2.rename(columns={'Lo PRIOR': 'Dec 1', 'PRIOR 2': 'Dec 2', 'PRIOR 3': 'Dec 3', 'PRIOR 4': 'Dec 4', 'PRIOR 5': 'Dec 5',
                             'PRIOR 6': 'Dec 6', 'PRIOR 7': 'Dec 7', 'PRIOR 8': 'Dec 8', 'PRIOR 9': 'Dec 9', 'Hi PRIOR': 'Dec 10'}, inplace=True)
        cols = ['Dec 1', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5',
                'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Dec 10']

        dat3 = dat2[cols].copy()  # Select relevant columns
        dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)
        if details is True:
            print("----------------")
            print("Short Term Reversal Strategy")
            print("----------------")
            print("Basic Strategy: stocks are sorted into deciles based on their prior 1-month return.")
            print("Construction: The portfolios are formed on the prior one-month return at the end of each month.")
            print("Each portfolio is value-weighted.")

            min_date = dat3.index.min()
            max_date = dat3.index.max()
            print()
            print(f"Min Date: {min_date}, Max Date: {max_date}")

    #------------------------------------------
    elif stype == 'accruals':

        # Make the request using the session
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_AC_csv.zip"
        response = requests.get(url)

        # Read the content of the file
        zip_content = response.content

        # Open the zip file from the content
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            with zf.open('Portfolios_Formed_on_AC.csv') as f:
                # Read the CSV file content (you can load it into pandas or process as needed)
                dat = pd.read_csv(f, skiprows=17, header=0, encoding='utf-8', skipfooter=5, engine='python')

        start_index = dat[dat.iloc[:, 0].str.contains("Equal Weighted Returns -- Monthly", na=False)].index[0]
        dat1 = dat.iloc[:start_index]
        dat2 = dat1.copy()
        dat2.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

        # Convert first column to Period
        dat2['date'] = pd.to_datetime(dat2['date'].astype(str), format='%Y%m').dt.to_period()

        # Convert all columns except 'date' to numeric types
        for col in dat2.columns[1:]:  # Skip the 'date' column
            dat2[col] = pd.to_numeric(dat2[col], errors='coerce')

        dat2.iloc[:, 1:] = dat2.iloc[:, 1:] * 0.01

        # reset the index
        dat2.set_index('date', inplace=True)
        dat2.rename(columns={'Lo 10': 'Dec 1', 'Hi 10': 'Dec 10'}, inplace=True)
        cols = ['Dec 1', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5',
                'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Dec 10']

        dat3 = dat2[cols].copy()  # Select relevant columns
        dat3.index = dat2.index  # Keep the index (assumed to be a PeriodIndex)
        if details is True:
            print("----------------")
            print("Accruals Strategy")
            print("----------------")
            print("The portfolios are formed on Accruals at the end of each June using NYSE breakpoints.")
            print("Accruals for June of year t is the change in operating working capital per split-adjusted share from")
            print("the fiscal year end t-2 to t-1 divided by book equity per share in t-1.")
            print("Stocks are ranked and sorted into deciles. Each decile portfolio is value-weighted.")

            min_date = dat3.index.min()
            max_date = dat3.index.max()
            print()
            print(f"Min Date: {min_date}, Max Date: {max_date}")

    else:
        raise ValueError("Invalid strategy type. Choose 'beta', 'momentum', or 'shortermreversal'.")

    #------------------------------------------
    # Apply date range filtering if provided
    if start_date is not None:
        dat3 = dat3[dat3.index >= pd.Period(start_date, freq='M')]
    if end_date is not None:
        dat3 = dat3[dat3.index <= pd.Period(end_date, freq='M')]

    ff3=get_ff3()
    ff3.rename(columns={'Mkt-RF':'mkt-rf', 'SMB':'smb', 'HML':'hml', 'RF':'rf'},inplace=True)

    dat_final=pd.merge(dat3,ff3[['mkt-rf','smb','hml','rf']],left_index=True,right_index=True,how='inner')
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
def describe(name, series):
    std = np.std(series)         # Compute the standard deviation (volatility)
    mean = np.mean(series)       # Compute the mean (average return)
    # Print formatted output with consistent spacing and precision
    print(f"{name:20s} |  Mean: {mean:.4f} | Std Dev: {std:.6f}")
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
                     covariance_matrix: np.ndarray, rf=None, zerocost=None) -> float:
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

    if zerocost==True:
        Sharpe=(port_ret) / port_vol
    else:
        Sharpe=(port_ret - rf) / port_vol

    # Compute and return the Sharpe ratio (excess return divided by risk)
    return Sharpe

####################################################################################################
def tangent_portfolio(expected_returns, covariance_matrix, rf=None, factors=None):
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

    if factors!=True:
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
    else:
        constraints = ({'type': 'eq', 'fun': lambda x: x[0] - 1})  # Constraint: weights sum to 1

        # Define a lambda function to negate the output of portfolio_sharpe
        neg_portfolio_sharpe = lambda x, expected_returns, covariance_matrix: -portfolio_sharpe(x, expected_returns, covariance_matrix, zerocost=True)

        # Perform optimization
        result = minimize(fun=neg_portfolio_sharpe, x0=initial_weights, args=(expected_returns, covariance_matrix),
                      method="SLSQP", constraints=constraints)

        # Ensure result.x is reshaped as a column vector (N x 1)
        tangent_weights = result.x

        # Compute expected return and volatility for the tangent portfolio
        tangent_return = tangent_weights.T @ expected_returns
        tangent_volatility = portfolio_volatility(tangent_weights, covariance_matrix)


    return tangent_weights, tangent_return, tangent_volatility
###################################################################################################
def black_scholes(type, S, K, T, rf, sigma):

    r=np.log(1+rf) # create continuous-time risk-free rate

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == "put":
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    return option_price
##################################################################################################
def implied_volatility(type, option_price, S, K, T, r):

    def objective(sigma):
        return black_scholes(type, S, K, T, r, sigma) - option_price

    try:
        return brentq(objective, 1e-6, 5.0)
    except ValueError:
        return np.nan