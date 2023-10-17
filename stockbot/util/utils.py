from sklearn.preprocessing import StandardScaler, MinMaxScaler
from iexfinance.refdata import get_symbols
from .. import config, reset, STOCKS, DATA, session
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm
import yfinance as yf
import os

@dataclass(frozen=True, unsafe_hash=True)
class Utility:

    '''
        inputs
            filename: location for StockVolume.csv
            threshold: volume based cutoff
        outputs:
            returns a current set.
    '''
    @staticmethod
    def get_volume_symbols(filename, threshold=200000):
        final_list   = pd.read_csv(filename)
        currency_set = Utility.get_currency(final_list)
        symbol_volume = set(final_list[(final_list.Volume > threshold)].sort_values('Volume').symbol)

        for sym in symbol_volume:
            currency_set.add(sym)

        return currency_set

    '''
        Get only those symbols that pertain to currencies.
    '''
    @staticmethod
    def get_currency(final_list):
        volume_zero = set(final_list[final_list.Volume == 0].symbol)
        currency_set = set()

        for currency in volume_zero:
            if '=X' in currency:
                currency_set.add(currency)
        return currency_set

    '''
        Pulls all the available symbols and merge with Yahoo finance.
    '''
    @staticmethod
    def get_all_symbols():
        try:
            IEX = get_symbols(output_format='pandas', token=config['TOKEN'])
            IEX = IEX[IEX.exchange.isin(['NAS', 'NYS', 'PSE'])].reset_index()
            stockcurrency_symbol = set(IEX.symbol)
            ycurrency = pd.read_html('https://finance.yahoo.com/currencies/')

            for x in set(ycurrency[0].Symbol):
                stockcurrency_symbol.add(x)

            return stockcurrency_symbol
        except:
            # TODO: Logging
            raise

    @staticmethod
    def run_symbol_volume(allSymbols, path = '.'):
        final_list = pd.DataFrame()

        for symbol in allSymbols:
            yf_stock = yf.Ticker(symbol)
            datapull = yf_stock.history().tail(1).reset_index()
            datapull['symbol'] = symbol
            final_list = pd.concat([final_list, datapull], axis=0)

        final_list.to_csv(os.path.join(path, 'StocksVolume.csv'))


    @staticmethod
    def createTensor(allSymbols, session, rewardName = 'TSLA', anchor_symbol = 'SPY'):
        anchor_date = pd.DatetimeIndex( 
                            pd.read_sql(sql = session.query(STOCKS).join(STOCKS.data)\
                                       .order_by(DATA.Date.asc()) \
                                       .filter(STOCKS.symbol==anchor_symbol)\
                                       .with_entities(STOCKS.symbol, \
                                                      DATA.Date).statement,\
                                        con = session.bind).Date)

        FullDataPull = pd.read_sql(sql = session.query(STOCKS).join(STOCKS.data).order_by(DATA.Date.asc()).with_entities(STOCKS.symbol,
                                                                                 DATA.Date,
                                                                                 DATA.Open,
                                                                                 DATA.High,
                                                                                 DATA.Low,
                                                                                 DATA.Volume,
                                                                                 DATA.Dividends,
                                                                                 DATA.Splits, 
                                                                                 DATA.Close).statement,
                                   con = session.bind)

        FullDataPullgrouped = FullDataPull.groupby('symbol')

        TensorArray  = []
        TensorSymbol = []
        RewardArray  = []

        for name, group_ in tqdm(FullDataPullgrouped): # parallelize this later.

            group = group_.drop_duplicates('Date', keep = 'last')

            len_cond = len(group) == len(anchor_date)

            if len_cond:
                date_cond = (pd.DatetimeIndex(group.Date) != anchor_date).sum()

                if date_cond > 0:
                    temp = group.set_index('Date').reindex(anchor_date, 
                                                           fill_value=0).drop('symbol',
                                                                               axis=1)

                else:
                    temp = group.set_index('Date').drop('symbol',axis=1)

            else:
                temp = group.set_index('Date').reindex(anchor_date, 
                                               fill_value=0).drop('symbol',
                                                                   axis=1)

            temp   = temp.to_numpy() 
            if name == rewardName:
                reward = temp.copy()
                RewardArray = reward

            scaler = MinMaxScaler(feature_range=(-1.0, 1.0)) #StandardScaler() # 
            temp   = scaler.fit_transform( temp )

            TensorArray.append( temp )
            TensorSymbol.append( name )

        TensorArray = np.array(TensorArray)
        TensorSymbol = np.array(TensorSymbol)

        return TensorArray, TensorSymbol, RewardArray, anchor_date