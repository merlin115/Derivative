import pandas as pd
import numpy as np
import tushare as ts
import datetime
import math

class download_data():

    def __init__(self, SC, sta_date, end_date):
        self.SC = SC
        self.sta_date = sta_date
        self.end_date = end_date
        self.token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
        self.pro = ts.pro_api(self.token)

    def get_stock_code(self):
        return self.SC

    def set_stock_code(self, SC):
        self.SC = SC

    def get_stock_data(self):
        # pro = ts.pro_api("d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c")
        df = self.pro.daily(ts_code=self.SC,
                            start_date = self.sta_date,
                            end_date=self.end_date)
        df.index = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.sort_index(inplace=True)
        return df

    def get_r(self, type = "shibor"):
        if type == "libor":
            df = self.pro.libor(curr_type='RMB',
                                start_date = self.sta_date,
                                end_date=self.end_date)
        if type == "shibor":
            df = self.pro.shibor(curr_type='USD',
                                start_date=self.sta_date,
                                end_date=self.end_date)
        if type == "hibor":
            df = self.pro.hibor(curr_type='USD',
                                start_date=self.sta_date,
                                end_date=self.end_date)

        df.index = pd.to_datetime(df["date"], format="%Y%m%d")
        df.sort_index(inplace=True)



# def get_stock_data(stock_code:str = "000001.SZ", sta_date = "20100101", end_date = ""):
#
#     pro = ts.pro_api("d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c")
#     df = pro.daily(ts_code=stock_code, end_date='202001031')
#     df.index = pd.to_datetime(df["trade_date"], format="%Y%m%d")
#     df.sort_index(inplace = True)
#
#     return df
#
#
# df.columns
#
# close = df["close"]

def get_volatility(df, method = "percentage"):
    # df = close
    length = df.shape[0]
    Xi_list = []
    for i in range(0, length-1):
        # i = 0

        # 百分比价格变动法
        if method == "percentage":
            Xi = (df.iloc[i+1] - df.iloc[i]) / df.iloc[i]

        # 对数价格变动法
        elif method == "logarithm":
            Xi = math.log(df.iloc[i + 1]) - math.log(df.iloc[i])

        Xi_list.append(Xi)
    Xi_list = np.array(Xi_list)
    X_bar = Xi_list.mean()
    sigma = math.sqrt(sum([(x - X_bar)**2 for x in Xi_list]))
    return sigma

# get_volatility(close)



def main():
    print("The stock code is: ")
    SC = "000001.SZ"
    print("The start date and the end date are: ")
    sta_date = "20100101"
    end_date = "20201010"

    project = download_data(SC=SC, sta_date=sta_date, end_date=end_date)
    project.get_stock_code()
    data = project.get_stock_data()
    df_r = project.get_r()

    pro.shibor(curr_type="RMB", start_date=sta_date, end_date=end_date)


    d_sta_date = datetime.datetime.strptime(sta_date, "%Y%m%d")
    d_end_date = datetime.datetime.strptime(end_date, "%Y%m%d")




