import datetime as dt
import pandas as pd
import pandas_datareader as web
from twstock import Stock	

n=int(input("enter years : "))
today=dt.datetime.today()
n_yrs_ago=today-dt.timedelta(days=n*365)
def get_data(ticker):
	stock=Stock(ticker)
	index=Stock('3056')
	data=stock.fetch_from(n_yrs_ago.year, n_yrs_ago.month)
	data_index=index.fetch_from(n_yrs_ago.year, n_yrs_ago.month)
	s_close=[]
	tpe_index=[]
	for i in range(len(data)):
		s_close.append(data[i][6])
		tpe_index.append(data_index[i][6]*500)
	df_close=pd.DataFrame(s_close)
	df_index=pd.DataFrame(tpe_index)
	df_close_index=pd.DataFrame(list(zip(s_close, tpe_index)))
	conv_to_csv(df_close, df_index, df_close_index)
	frame=[df_close, df_index]
	return frame
	
def conv_to_csv(df_close, df_index, df_close_index):
	df_close.to_csv("close.csv")
	df_index.to_csv("index.csv")
	df_close_index.to_csv("close_index.csv")
	csv_to_df()
def csv_to_df():
	close=pd.read_csv("close.csv", usecols=[1])
	index=pd.read_csv("index.csv", usecols=[1])
	close_index=pd.read_csv("close_index.csv", usecols=[1, 2])
	print(close)
	print(index)
	print(close_index)
	return close
	return index
	return close_index
	
print("2337")
print("2330")
print("6223")
print("6220")
choice=input("Enter the stock code : ")
get_data(choice)

