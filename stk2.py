import datetime as dt
import pandas as pd
import pandas_datareader as web
from yahoo_finance import Share
from rtstock.stock import Stock
from twstock import Stock

today = dt.datetime.today()
two_yrs_ago = today - dt.timedelta(days=2 * 365)


def get_data(ticker):
    stock = Stock(ticker)
    index = Stock('3056')
    data = stock.fetch_from(two_yrs_ago.year, two_yrs_ago.month)
    data_index = index.fetch_from(two_yrs_ago.year, two_yrs_ago.month)
    s_close = []
    tpe_index = []
    for i in range(485):
        s_close.append(data[i][6])
        tpe_index.append(data_index[i][6] * 500)
    df_close = pd.DataFrame(s_close)
    df_index = pd.DataFrame(tpe_index)
    # df_close_n_index=pd.DataFrame(s_close)
    # df_close_n_index[1]=tpe_index
    print(df_close)
    print(df_index)
    # print(df_close_n_index)


print("2337")
print("2330")
print("6223")
print("6220")
choice = str(input("Enter the stock code : "))
get_data(choice)
