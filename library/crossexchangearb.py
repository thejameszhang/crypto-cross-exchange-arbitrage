import numpy as np
import pandas as pd
from functools import reduce
import datetime
from typing import List, Optional, Dict
import pyarrow as pa
import pyarrow.parquet as pq

### MAINTAIN LISTS OF ALL FEATUERS, EXCHANGES, CURRENCY PAIRS ###

features = ['close', 'high', 'low', 'open', 'volume', 'vwap']
exchanges = ['binance_futures', 'binance_spot', 'binanceus', 'okx']
currency_pairs = ['APE_USDT', 'AVAX_USDT', 'AXS_USDT', 'BAKE_USDT', 'BNB_USDT',
       'BTC_BUSD', 'BTC_USDT', 'CRV_USDT', 'CTK_USDT', 'DOGE_USDT', 'DOT_USDT',
       'DYDX_USDT', 'ETH_BUSD', 'ETH_USDT', 'FTM_USDT', 'GMT_USDT',
       'LINK_USDT', 'MATIC_USDT', 'NEAR_USDT', 'OGN_USDT', 'RUNE_USDT',
       'SAND_USDT', 'SOL_USDT', 'STORJ_USDT', 'UNFI_USDT', 'WAVES_USDT',
       'XRP_USDT']

### FUNCTIONS TO CONVERT DATA TO MULTINDEX ### 

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
    # After postprocessing, create column names to convert this to multindex.
    if keep_single:
        return res_df
    res_df = convert_to_multindex(res_df)

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

def convert_to_multindex(single_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all of the column levels such that we can transform 
    the single_index_df into multi_index.
    
    :param single_df: dataframe returned by convert_to_multi_index
    :return: a multindex dataframe
    """
    
    # Store the timestamp for later use.
    timestamp = single_df.index
    # Create a list of all column names.
    columns = single_df.columns
    # Create the feature level.
    feature_levels = [column.split("-")[0] for column in columns]
    # Create a temp level.
    temp = [column.split("-")[-1] for column in columns]
    # Create the exchange level. 
    exchange_levels = [column.split("::")[0] for column in temp]
    # Create the currency pair level.
    currency_pair_levels = [column.split("::")[-1] for column in temp]
    # Convert the given dataframe to multindex.
    feature_string = " ".join([str(feature) for feature in feature_levels])
    exchange_string = " ".join([str(exchange)for exchange in exchange_levels])
    currency_pair_string = " ".join([str(pair) for pair in currency_pair_levels])
    res_df = pd.DataFrame(np.array(single_df), columns=[feature_string.split(), 
                                                        exchange_string.split(), 
                                                        currency_pair_string.split()])
    # Restore the initial timestamp and sort.
    res_df.index = timestamp
    return res_df

def merge_postprocess_exchange_data(
    exchange_dfs: List[pd.DataFrame]
) -> List[pd.DataFrame]:
    """
    Converts a list of exchange dataframes into one large
    multindex dataframe.

    :param exchange_dfs: list of exchange dataframes
    :return: multindex dataframe
    """
    # Postprocess each dataframe.
    converted_dfs = [postprocess_exchange_data(df, True) for df in exchange_dfs]
    # Merge dataframes.
    res_df = pd.concat(converted_dfs, axis=1)
    # Sort by time and columns before passing into convert_to_multindex 
    res_df = res_df.sort_index()
    res_df = res_df.sort_index(axis=1)
    # Now convert this merged dataframe to multiiindex.
    res_df = convert_to_multindex(res_df)
    return res_df


def get_coins(multindex_df: pd.MultiIndex) -> List[str]:
    """
    Extract all the unique currency pairs from multindex exchange dataframe.

    :param multindex_df: multindex dataframe
    :return: list of coins 
    """
    for level in multindex_df.columns.levels:
        if level[0] in currency_pairs:
            return list(level)
    raise Exception("Coins level not found.")

def get_ncoins(multindex_df: pd.MultiIndex) -> int:
    """
    Returns the number of unique currency pairs in the dataframe.

    :param multindex_df: a multindex dataframe
    :returns: number of coins
    """
    return len(get_coins(multindex_df))

def find_type(multindex_df: pd.MultiIndex, desired: str) -> int:
    """
    Helper function that finds what level the desired type is located.

    :param multindex_df: a multindex dataframe
    :param desired: "feature", "exchange", or "coin"
    :return: where the feature, exchange, or coin level currently is
    """
    # Assert that we have are given a multindex dataframe.
    assert len(multindex_df.columns.levels) > 1
    # Find what the user is looking for.
    if desired == "feature":
        desired_list = features 
    elif desired == "exchange":
        desired_list = exchanges
    else:
        desired_list = currency_pairs
    # Loop through the column levels.
    for i, level in enumerate(multindex_df.columns.levels):
        if level[0] in desired_list:
            return i
    raise Exception("Type not found in the dataframe.")

def get_coins_info(multindex_df: pd.MultiIndex, coin: str) -> pd.MultiIndex:
    """
    Returns a two-level dataframe with only the given coin.

    :param multindex_df: multindex dataframe
    :param coin: coin
    :return: all data associated with the coin
    """

    # Assert we have three levels in our dataframe.
    assert len(multindex_df.columns.levels) == 3
    # Now find where the desired type currently is
    level = find_type(multindex_df, "coin")
    # If the type is located at level 0, we can just use dataframe indexing.
    if level == 0:
        return multindex_df[coin]
    # Else move level to the 0th level such that we can use indexing.
    else:
        multindex_df = multindex_df.swaplevel(0, level, 1)
    res_df = multindex_df[coin]
    res_df = res_df.sort_index(axis=1)
    return res_df

def get_exchanges(multindex_df: pd.MultiIndex) -> List[str]:
    """
    Extract all the exchanges from multindex exchange dataframe.

    :param multindex_df: multindex dataframe 
    :return: list of exchanges
    """

    for level in multindex_df.columns.levels:
        if level[0] in exchanges:
            return list(level)
    raise Exception("Exchanges level not found.")

def get_nexchanges(multindex_df: pd.MultiIndex) -> int:
    """
    Returns the number of unique exchanges in the dataframe.

    :param multindex_df: a multindex dataframe
    :returns: number of exchanges
    """
    return len(get_exchanges(multindex_df))
    

def get_exchange_info(multindex_df: pd.MultiIndex, exchange: str) -> pd.MultiIndex:
    """
    Returns a two-level dataframe with only the given exchange.

    :param multindex_df: multindex dataframe
    :param exchange: the desired exchange
    :return: all data associated with the exchange
    """

    # Assert we have three levels in our dataframe.
    assert len(multindex_df.columns.levels) == 3
    # Now find where the desired type currently is
    level = find_type(multindex_df, "exchange")
    # If the type is located at level 0, we can just use dataframe indexing.
    if level == 0:
        return multindex_df[exchange]
    # Else move level to the 0th level such that we can use indexing.
    else:
        multindex_df = multindex_df.swaplevel(0, level, 1)
    res_df = multindex_df[exchange]
    res_df = res_df.sort_index(axis=1)
    return res_df

def get_features(multindex_df: pd.MultiIndex) -> List[str]:
    """
    Extract all the features from the multindex dataframe.

    :param multindex_df: multindex dataframe 
    :return: list of features
    """

    for level in multindex_df.columns.levels:
        if level[0] in features:
            return list(level)
    raise Exception("Features level not found.")

def get_nfeatures(multindex_df: pd.MultiIndex) -> int:
    """
    Returns the number of unique features in the dataframe.

    :param multindex_df: a multindex dataframe
    :returns: number of features
    """
    return len(get_features(multindex_df))

def get_feature_info(multindex_df: pd.MultiIndex, feature: str) -> pd.MultiIndex:
    """
    Returns a two-level dataframe with only the given feature.

    :param multindex_df: multindex dataframe
    :param feature: the desired feature
    :return: all data associated with the feature
    """

    # Assert we have three levels in our dataframe.
    assert len(multindex_df.columns.levels) == 3
    # Now find where the desired type currently is
    level = find_type(multindex_df, "feature")
    # If the type is located at level 0, we can just use dataframe indexing.
    if level == 0:
        return multindex_df[feature]
    # Else move level to the 0th level such that we can use indexing.
    else:
        multindex_df = multindex_df.swaplevel(0, level, 1)
    res_df = multindex_df[feature]
    res_df = res_df.sort_index(axis=1)
    return res_df

### FUNCTIONS TO HELP PERFORM CROSS EXCHANGE ARBITRAGE ### 

def swaplevels(multindex_df: pd.MultiIndex) -> pd.MultiIndex:
    """
    Converts the dataframe to have a the currency pair as the outer level, 
    the feature as the middle level, and the exchange as the inner level.

    :param multindex_df: a multindex dataframe
    :return: a formatted multindex dataframe
    """

    # Assert that the dataframe has 3 columns, or else we cannot proceed.
    assert len(multindex_df.columns.levels) == 3
    assert multindex_df.columns.levels[0][0] in features
    assert multindex_df.columns.levels[1][0] in exchanges
    assert multindex_df.columns.levels[2][0] in currency_pairs
    # Our formula only needs volume and vwap.
    cols_to_drop = ["open", "high", "low", "close"]
    multindex_df = multindex_df.drop(columns=cols_to_drop)
    # Now make the appropriate swaps.
    multindex_df = multindex_df.swaplevel(0, 2, 1)
    multindex_df = multindex_df.swaplevel(1, 2, 1)
    multindex_df = multindex_df.sort_index(axis=1)
    return multindex_df

def arbitrage(multindex_df: pd.MultiIndex) -> pd.DataFrame:
    """
    Creates dataframe representing all possible profitable trades for
    the specified coin given the multindex dataframe.

    :param multindex_df: multindex dataframe with only the given coin.
    :return coin: a currency pair
    :return: total profit for this coin
    """
    # Index by the given coin.
    coin_df = multindex_df[coin]
    # Isolate the timestamp for later use.
    coin = pd.DataFrame(coin_df.index)
    coin = coin.set_index('timestamp')
    # Get a list of the exchanges.
    exchange_list = list(coin_df['vwap'].columns)
    # Create minimum and maximum vwap columns.
    coin['price_i'] = coin_df['vwap'][exchange_list].min(axis=1)
    coin['price_j'] = coin_df['vwap'][exchange_list].max(axis=1)
    # Create exchange columns that have the minimum and maximum price.
    coin['i'] = coin_df['vwap'][exchange_list].idxmin(axis=1)
    coin['j'] = coin_df['vwap'][exchange_list].idxmax(axis=1)
    # Create volume columns.
    for exchange in exchange_list:
        coin[exchange] = coin_df['volume'][exchange]
    # Only consider rows where there exists a price dislocation, i != j != NaN.
    # Save a copy of coin.
    coin2 = coin.loc[coin['i'] != coin['j']]
    coin2 = coin2.dropna(subset=['i','j'])
    # Group by the pairs.
    dfs = [group for _, group in coin2.groupby(['i', 'j'])]
    # Compute the min_volume column for each df. 
    for df in dfs:
        exchange1, exchange2 = df['i'].unique()[0], df['j'].unique()[0]
        df['min_volume'] = df[[exchange1, exchange2]].min(axis=1)
    # Concatenate them, sort by timestamp, and drop unnecessary columns
    if dfs:
        res_df = pd.concat(dfs)
        res_df = res_df.sort_index()
        res_df = res_df.drop(columns=exchange_list)
        res_df['instant_profit'] = res_df['min_volume'] * abs(res_df['price_i'] - res_df['price_j'])
    else:
        res_df = coin
        res_df['instant_profit'] = np.zeros(len(coin_df))
    return res_df

def arbitrage_all(multindex_df: pd.MultiIndex) -> Dict[str: pd.DateFrame]:
    """
    Computes total arbitrage for all coins in the given multindex dataframe.

    :param multindex_df: a multindex dataframe
    :return: a dictionary where the key is the coin and the value is the
    dataframe representing profitable trades
    """
    coins_list = multindex_df.columns.levels[0]
    return {coin: arbitrage(multindex_df, coin) for coin in coins_list}