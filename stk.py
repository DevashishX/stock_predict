import datetime as dt
import pandas as pd
import pandas_datareader as web
from twstock import Stock
import os.path

#n=int(input("enter years : "))

today = dt.datetime.today()
n_yrs_ago = today - dt.timedelta(days=3 * 365)


def get_data(ticker):
    stock = Stock(ticker)
    index = Stock('3056')
    data = stock.fetch_from(n_yrs_ago.year, n_yrs_ago.month)
    data_index = index.fetch_from(n_yrs_ago.year, n_yrs_ago.month)
    s_close = []
    tpe_index = []
    for i in range(len(data)):
        s_close.append(data[i][6])
        tpe_index.append(data_index[i][6] * 500)
    df_close = pd.DataFrame(s_close)
    df_index = pd.DataFrame(tpe_index)
    df_close_index = pd.DataFrame(list(zip(s_close, tpe_index)))
    conv_to_csv(df_close, df_index, df_close_index, ticker)
    frame = [df_close, df_index]
    return frame


def conv_to_csv(df_close, df_index, df_close_index, ticker):
    state = "close"
    file_name_close = "{}_{}.csv".format(ticker, state)
    df_close.to_csv(file_name_close)
    df_index.to_csv("index.csv")
    file_name_ci = "{}_{}_{}.csv".format(ticker, state, "index")
    df_close_index.to_csv(file_name_ci)
    csv_to_df(ticker)


def csv_to_df(name):
    state = "close"
    file_name_close = "{}_{}.csv".format(name, state)
    # file_name_close = "{}_{}.csv".format(name, state)
    close = pd.read_csv(file_name_close, usecols=[1])
    index = pd.read_csv("index.csv", usecols=[1])
    file_name_ci = "{}_{}_{}.csv".format(name, state, "index")
    close_index = pd.read_csv(file_name_ci, usecols=[1, 2])
    # print(close)
    # print(index)
    # print(close_index)
    return close, index, close_index


def select_ticker():
    print("2337 : Macronix International")
    print("2330 : Taiwan Semiconductor Manufacturing")
    print("6223 : MPI Corp")
    print("6220 : BonEagle Electric Co Ltd")
    choice = input("Enter the stock code : ")
    file_name_close = "{}_{}.csv".format(choice, "close")
    if(os.path.exists(file_name_close) == True):
        csv_to_df(choice)  # if we want real time, replace by get_data(choice)
    else:
        get_data(choice)


def main():
    select_ticker()


if __name__ == "__main__":
    main()
