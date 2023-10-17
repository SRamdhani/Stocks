from sqlalchemy import Column, ForeignKey, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from tqdm import tqdm

config = {
    'TOKEN': "pk_dc9306f173d843288561267e71516164", # this is not used anymore
    'DBSTRING': "postgresql://localhost/stockcurrency",
    'ANCHOR': 'SPY'
}

'''
    Set up sqlalchemy engine and create session.
    Bind all the tables to be used to the Base.
'''

db        = create_engine(config['DBSTRING'])
Session   = sessionmaker()
Session.configure(bind=db)
session   = Session()
Base      = declarative_base()

'''
    STOCKS: Stores high-level information about the symbol.
        stocks_id: integer primary key.
        symbol: string for stock symbol.
        sector: sector for the stock -> relationship to SECTOR.
        data: data for the symbol -> relationship to DATA.
'''
class STOCKS(Base):
    __tablename__ = 'stocks'
    stocks_id = Column(Integer,        primary_key=True)
    symbol    = Column(String(250),    index=True)
    sector    = relationship('SECTOR', secondary='stock_sector', back_populates='stocks')
    data      = relationship('DATA',   secondary='stock_data',   back_populates='stock')

'''
    DATA: Contains the actual data to be joined when needed.
        Date_id: Integer primary key.
        Date: Date from yfinance date.
        Open: Float from yfinance open.
        High: Float from yfinance high.
        Low: Float from yfinance low.
        Close: Float from yfinance close.
        Volume: Float from yfinance volume.
        Dividends: Float from yfinance dividends.
        Splits: Float from yfinance splits. 
        stock: information about the symbol -> relationship to STOCKS.
'''
class DATA(Base):
    __tablename__ = 'data'
    Date_id = Column(Integer, primary_key=True)
    Date    = Column(Date(),  index=True)
    Open    = Column(Float())
    High    = Column(Float())
    Low     = Column(Float())
    Close   = Column(Float())
    Volume  = Column(Float())
    Dividends = Column(Float())
    Splits    = Column(Float())
    stock     = relationship('STOCKS', secondary='stock_data', back_populates='data')

'''
    STOCK_DATA: join id's for symbol and dates.
        date_id: Integer column primary key from DATA.Date_id.
        symbol_id: Integer column primary key from STOCKS.stocks_id.
'''
class STOCK_DATA(Base):
    __tablename__ = 'stock_data'
    date_id   = Column(Integer, ForeignKey('data.Date_id'),
                       primary_key=True)
    symbol_id = Column(Integer, ForeignKey('stocks.stocks_id'),
                       primary_key=True)

'''
    SECTOR: sector based information containing the stock not used yet.
        sector_id: Integer primary key.
        stocks: stock information -> relationship to STOCKS.
        name: String for the actual sector name.
'''
class SECTOR(Base):
    __tablename__ = 'sector'
    sector_id = Column(Integer,        primary_key=True)
    stocks    = relationship('STOCKS', secondary='stock_sector', back_populates='sector')
    name      = Column(String(250),    nullable=False)

'''
    STOCK_SECTOR: join id's for symbol and sector.
        sector_id: Integer foreign key from SECTOR.sector_id.
        symbol_id: Integer foreign key from STOCKS.stocks_id.
'''
class STOCK_SECTOR(Base):
    __tablename__ = 'stock_sector'
    sector_id = Column(Integer, ForeignKey('sector.sector_id'),
                       primary_key=True)
    symbol_id = Column(Integer, ForeignKey('stocks.stocks_id'),
                       primary_key=True)

# Resets or Deletes Table.
def reset(TABLE, session):
    session.query(TABLE).delete()
    session.commit()

# Resets or deletes all the elements from the table.
def resetiter(TABLE, session):
    getall = session.query(TABLE).all()
    for elem in tqdm(getall):
        session.delete(elem)
        session.commit()

Base.metadata.create_all(db)