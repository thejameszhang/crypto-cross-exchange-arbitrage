import numpy as np
import pandas as pd
from functools import reduce
import datetime
from typing import List, Optional 
import pyarrow as pa
import pyarrow.parquet as pq

features = ['close', 'high', 'low', 'open', 'volume', 'vwap']
exchanges = ['binance_futures', 'binance_spot', 'binanceus', 'okx']
currency_pairs = ['APE_USDT', 'AVAX_USDT', 'AXS_USDT', 'BAKE_USDT', 'BNB_USDT',
       'BTC_BUSD', 'BTC_USDT', 'CRV_USDT', 'CTK_USDT', 'DOGE_USDT', 'DOT_USDT',
       'DYDX_USDT', 'ETH_BUSD', 'ETH_USDT', 'FTM_USDT', 'GMT_USDT',
       'LINK_USDT', 'MATIC_USDT', 'NEAR_USDT', 'OGN_USDT', 'RUNE_USDT',
       'SAND_USDT', 'SOL_USDT', 'STORJ_USDT', 'UNFI_USDT', 'WAVES_USDT',
       'XRP_USDT']

def swaplevels_cross_exchange_arbitrage(multindex_df: pd.MultiIndex) -> pd.MultiIndex:
    """
    Converts the dataframe to have a the currency pair as the outer level, 
    the feature as the middle level, and the exchange as the inner level.

    :param multindex_df: a multindex dataframe
    :return: a formatted multindex dataframe
    """
    # Assert that the dataframe has 3 columns, or else we cannot proceed.
    assert len(multindex_df.columns.levels) == 3
    # Save the timestamp for later.
    timestamp = multindex_df.index
    # We have to restructure columns, so save the column length.
    length = len(multindex_df.columns)
    # Determine which level corresponds to feature, exchange, currency.
    for level in multindex_df.columns.levels:
        if level[0] in features:
            feature_level = sorted(list(level) * int(length / len(level)))
        elif level[0] in exchanges:
            exchange_level = sorted(list(level) * int(length / len(level)))
        else:
            currency_pair_level = sorted(list(level) * int(length / len(level)))
    # Now create the strings from the lists.
    feature_string = " ".join([str(feature) for feature in feature_level])
    exchange_string = " ".join([str(exchange)for exchange in exchange_level])
    currency_pair_string = " ".join([str(pair) for pair in currency_pair_level])
    # Now create the correctly formatted multindex dataframe.
    res_df = pd.DataFrame(np.array(multindex_df.values), 
            columns=[currency_pair_string.split(), feature_string.split(), exchange_string.split()])
    # Restore the initial timestamp and sort.
    res_df.index = timestamp
    return res_df