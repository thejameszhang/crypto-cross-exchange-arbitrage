import numpy as np
import pandas as pd
from functools import reduce
import datetime
from typing import List, Optional 
import pyarrow as pa
import pyarrow.parquet as pq

def postprocess_exchange_data(
    exchange_df: pd.DataFrame, 
    keep_single: Optional[bool] = False
) -> pd.DataFrame:
    """
    Rearrange the exchange dataframe such that the index is time
    and at any time, features of all coins on the exchange can be 
    determined.

    :param exchange_df: exchange data
    :return: postprocessed exchange data
    """
    # Assert the index of the given dataframe is "timestamp".
    assert "timestamp" == exchange_df.index.name 
    # Rename this timestamp column to avoid having column with the same name as index.
    exchange_df = exchange_df.rename(columns={"timestamp":"timestamp.1"})
    # Move timestamp to a column and localize it.
    exchange_df = exchange_df.reset_index()
    exchange_df["timestamp"] = pd.to_datetime(exchange_df["timestamp"], utc=True)
    # Get the name of the exchange and assert it is unique.
    exchange_id = exchange_df["exchange_id"].unique()
    assert len(exchange_id) == 1
    exchange_id = exchange_id[0]
    # Drop all irrelevant columns.
    cols_to_drop = ["timestamp.1", "knowledge_timestamp", "year", "month", "exchange_id"]
    exchange_df = exchange_df.drop(columns=cols_to_drop)
    # Group the dataframe by currency pair.
    currency_pair_dfs = [group for _, group in exchange_df.groupby("currency_pair")]
    # Calls helper function that also renames OHLCV columns.
    currency_pair_dfs = calculate_vwap(currency_pair_dfs, exchange_id)
    # Initialize the dataframe that we will return, which starts as just time.
    res_df = pd.DataFrame(exchange_df["timestamp"].unique())
    res_df = res_df.rename(columns={0:"timestamp"})
    res_df = res_df.sort_values(by="timestamp")
    # Merge all currency pair dataframes into the return dataframe. 
    for currency_pair in currency_pair_dfs:
        res_df = pd.merge_asof(res_df, currency_pair.sort_values(by="timestamp"), on="timestamp")
    # Set index as timestamp which was lost during merging.
    res_df = res_df.set_index("timestamp")
    # Sort by column name to the order is consistent.
    res_df = res_df.sort_index(axis=1)
    # Drop duplicate columns if there are any.
    res_df = res_df.loc[:,~res_df.columns.duplicated()].copy()
    # After postprocessing, create column names to convert this to multiindex.
    if keep_single:
        return res_df
    res_df = convert_to_multiindex(res_df)
    return res_df

def calculate_vwap(
    currency_pair_dfs: List[pd.DataFrame],
    exchange_id: str,
) -> List[pd.DataFrame]:
    """
    Calculates vwap for each dataframe in the list of currency pair dataframes.

    :param currency_pair_dfs: list of dataframes
    :param exchange_id: exchange id
    :return: list of dataframes with a vwap column
    """
    for df in currency_pair_dfs:
        # Get name of currency_pair for renaming purposes.
        currency_pair = df["currency_pair"].unique()[0]
        columns = {
            "vwap": f"vwap-{exchange_id}::{currency_pair}",
            "volume": f"volume-{exchange_id}::{currency_pair}",
            "open": f"open-{exchange_id}::{currency_pair}",
            "high": f"high-{exchange_id}::{currency_pair}",
            "low": f"low-{exchange_id}::{currency_pair}",
            "close": f"close-{exchange_id}::{currency_pair}"
        }
        # Calculate vwap.
        midprice = (df["high"] + df["low"]) / 2
        numerator = np.cumsum(np.multiply(df["volume"], midprice))
        denominator = np.cumsum(df["volume"])
        df["vwap"] = np.divide(numerator, denominator)
        # Now rename the OHLCV columns.
        df.rename(columns=columns, inplace=True)
        # Drop irrelevant columns and set timestamp as index.
        df.drop(columns=["currency_pair"], inplace=True)
        df.set_index("timestamp", inplace=True)
    return currency_pair_dfs

def convert_to_multiindex(single_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all of the column levels such that we can transform 
    the single_index_df into multi_index.
    
    :param single_df: dataframe returned by convert_to_multi_index
    :return: a multiindex dataframe
    """
    
    # Store the timestamp for later use.
    timestamp = single_df.index
    # Create a list of all column names.
    columns = single_df.columns
    # Create the outer feature level.
    feature_levels = [column.split("-")[0] for column in columns]
    # Create the inner currency pair levels.
    currency_pair_levels = [column.split("-")[-1] for column in columns]
    res_df = pd.DataFrame(single_df.values, columns=[feature_levels, currency_pair_levels])
    # Convert the given dataframe to multiindex.
    feature_string = " ".join([str(feature) for feature in feature_levels])
    currency_pair_string = " ".join([str(pair) for pair in currency_pair_levels])
    res_df = pd.DataFrame(np.array(single_df), columns=[feature_string.split(), currency_pair_string.split()])
    # Restore the initial timestamp.
    res_df.index = timestamp
    # Drop duplicate columns if there are any.
    res_df = res_df.loc[:,~res_df.columns.duplicated()].copy()
    return res_df

def merge_postprocess_exchange_data(
    exchange_dfs: List[pd.DataFrame]
) -> List[pd.DataFrame]:
    """
    Converts a list of exchange dataframes into one large
    multiindex dataframe.

    :param exchange_dfs: list of exchange dataframes
    :return: multiindex dataframe
    """
    # Postprocess each dataframe.
    converted_dfs = [postprocess_exchange_data(df, True) for df in exchange_dfs]
    # Merge dataframes.
    res_df = pd.concat(converted_dfs, axis=1)
    # Sort by time and columns before passing into convert_to_multiindex 
    res_df = res_df.sort_index()
    res_df = res_df.sort_index(axis=1)
    # Drop duplicate columns if there are any.
    res_df = res_df.loc[:,~res_df.columns.duplicated()].copy()
    # Now convert this merged dataframe to multiiindex.
    res_df = convert_to_multiindex(res_df)
    return res_df

def get_symbols(multindex_df: pd.MultiIndex) -> List[str]:
    """
    Extract all the unique currency pairs from multiindex exchange dataframe

    :param multiindex_df: multiindex dataframe 
    :return: list of symbols
    """
    symbols = multindex_df["close"].columns
    symbols = [symbol.split("::")[-1] for symbol in symbols]
    symbols = sorted(list(symbols))
    return symbols

def get_symbol_info(multiindex_df: pd.MultiIndex, symbol: str) -> pd.MultiIndex:
    """
    Returns a two-level dataframe with only the given symbol.

    :param multiindex_df: multiindex dataframe
    :param symbol: symbol
    :return: all data associated with the symbol
    """
    columns_list = multiindex_df.columns
    columns = [column for column in columns_list if symbol in column[1]]
    return multiindex_df[columns]