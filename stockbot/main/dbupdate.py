from .. import config, SECTOR, STOCKS, DATA
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import yfinance as yf
import pandas as pd

'''
    update class:
        database: loop through all the stock symbols
                  and update. Starts from 1/1/2016.
        add_datapoint: creates a database entry element.
'''
@dataclass(frozen=False, unsafe_hash=False)
class update:
    @staticmethod
    def database(all_symbols, session, basetime=datetime(2019, 8, 22)):
        # Anchor symbol is a stable stock (always available) that we can use
        # in order to align all the time points for all the other symbols.
        anchor_symbol = config['ANCHOR']

        # Count of all observed timestamps for the anchor stock.
        anchor_len = len(pd.DatetimeIndex(
            pd.read_sql(sql=session.query(STOCKS).join(STOCKS.data)\
                        .order_by(DATA.Date.asc())\
                        .filter(STOCKS.symbol == anchor_symbol)\
                        .with_entities(STOCKS.symbol,\
                                       DATA.Date).statement,\
                        con=session.bind).Date))

        # Loop through all symbols and provide an update
        # based on yahoo finance API.
        for symbol in tqdm(all_symbols):

            # Query the STOCKS database for the symbol.
            query_stock = session.query(STOCKS)
            stock = query_stock.filter(STOCKS.symbol == symbol)

            try:
                yf_stock = yf.Ticker(symbol)
            except:
                continue

            # If the symbol does not exist then pull the symbol
            # from yfinance starting at the base time.
            if stock.count() == 0:

                try:
                    data_pull = yf_stock.history(period="max",
                                                start=basetime,
                                                end=datetime.now()).reset_index().dropna()
                except:
                    raise # TODO consider using continue.

                # For each data point create a sqlalchemy element
                # and commit to the database.
                for i in range(len(data_pull)):
                    update.add_datapoint(datapt=data_pull.iloc[i],
                                  symbol=symbol,
                                  session=session,
                                  sector_name='DEFAULT')
            else:
                # If the symbol does exist query DATA, filter the symbol and order the DATA.
                get_all_joined = session.query(DATA).join(STOCKS.data).filter(STOCKS.symbol == symbol).order_by(
                    DATA.Date.asc()).all()

                # Get the last update date.
                recent_date = get_all_joined[-1].Date

                # Pull the new data points starting from recent_date.
                data_pull = yf_stock.history(period="max",
                                            start=recent_date,
                                            end=datetime.now()).reset_index().dropna().iloc[-1:]

                # Get all the unique dates.
                check_dates = set([x.strftime("%Y-%m-%d") for x in list(data_pull.Date)])

                # This check is in the event no new data is pulled and recent_date is returned
                # or the data_pull count is nothing (i.e., the stock was delisted).
                if (recent_date.strftime("%Y-%m-%d") in check_dates) or (len(data_pull) == 0):
                    continue

                # Update the database if we have data.
                if (len(data_pull) > 0):

                    current_len = len(get_all_joined)

                    # If the current length of timestamps matches that for the
                    # anchor then pop off the elements at the beginning and
                    # append the new data points at the end. This is a rolling
                    # window of stock data that starts at base time to the time
                    # when the first sync occurred!
                    if (current_len == anchor_len):
                        for elem in get_all_joined[0:len(data_pull)]:
                            session.delete(elem)
                            session.commit()
                    for i in range(len(data_pull)):
                        update.add_datapoint(datapt=data_pull.iloc[i],
                                      symbol=symbol,
                                      session=session,
                                      sector_name='DEFAULT')

                else:
                    continue

    @staticmethod
    def add_datapoint(datapt, symbol, session, sector_name='DEFAULT'):
        # For default all sectors to DEFAULT.
        # This will be changed soon when the sector
        # part is segmented into relevant parts.

        try:
            sector = SECTOR(name=sector_name)
            stock = STOCKS(symbol=symbol)

            data = DATA(Date=datapt['Date'],
                         Open=float(datapt['Open']),
                         High=float(datapt['High']),
                         Low=float(datapt['Low']),
                         Close=float(datapt['Close']),
                         Volume=float(datapt['Volume']),
                         Dividends=float(datapt['Dividends']),
                         Splits=float(datapt['Stock Splits']))

            session.add(sector)
            stock.sector.append(sector)
            stock.data.append(data)
            session.add(stock)
            session.commit()

        except:
            print(symbol)
            print(datapt)
            # TODO: Logging
            raise



