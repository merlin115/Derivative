import pandas as pd
import numpy as np
import tushare as ts
import datetime
import math
from scipy.special import comb

from math import log, sqrt, exp

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

    def get_dividend(self):
        df = self.pro.daily(ts_code=self.SC,
                            start_date = self.sta_date,
                            end_date=self.end_date)
        df.index = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.sort_index(inplace=True)


    def get_r(self, type = "shibor"):
        if type == "libor":
            df = self.pro.libor(curr_type='USD',
                                start_date=self.sta_date,
                                end_date=self.end_date)
        if type == "shibor":
            df = self.pro.shibor(start_date=self.sta_date,
                                 end_date=self.end_date)
        if type == "hibor":
            df = self.pro.hibor(curr_type='RMB',
                                start_date=self.sta_date,
                                end_date=self.end_date)

        df.index = pd.to_datetime(df["date"], format="%Y%m%d")
        df.sort_index(inplace=True)
        return df["1y"]





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


def Price_by_BT(r, sigma, delta, K, S0, h, T):
    pass
    # r = 0.035
    # delta = 0.02
    # sigma = 1.2
    # h = 1/365
    # K = 120
    # S0 = 100
    #
    # n = int(T/h)
    # u = exp((r-delta) * h + sigma*sqrt(h))
    # d = exp((r-delta) * h - sigma*sqrt(h))
    #
    # c_u = max(0, u*S0 - K)
    # c_d = max(0, d*S0 - K)
    # p = ((1 + r - sigma) - d)/(u - d)
    #
    # comb_list = []
    # option_value_list = []
    # for i in range(0, n+1):
    #     comb_value = int(comb(n, i))
    #     u_coef = i
    #     d_coef = n + 1 - i
    #     option_value_pred = K - S0 * u**u_coef * d**d_coef
    #     option_value_adju = max()
    #     comb_list.append(comb_value)
    # max(0, K - S0 * u )
    # for i in range((0, n+1):
    #     c = (p * c_u + (1 - p) * c_d)/(1 + r - delta)
    # return c

def get_option_by_s(ST, X, alpha = 0.3):
    if alpha * ST >= ST - X >= 0:
        option = ST - X
    if ST - X > alpha * ST:
        option = alpha * ST
    if ST - X < 0:
        option = 0
    return option





def Price_by_BT_2(r, sigma, delta, X, S0, h, alpha = 0.3):

    u = math.exp((r-delta) * h + sigma*math.sqrt(h))
    d = math.exp((r-delta) * h - sigma*math.sqrt(h))
    p = (exp((r-delta) * h) - d)/(u-d)
    q = 1 - p
    uS = u * S0
    dS = d * S0
    uOp = get_option_by_s(uS, X, alpha)
    dOp = get_option_by_s(dS, X, alpha)
    sanjiao = exp(-delta*h) * (uOp - dOp)/(S0*(u-d))
    B = exp(-r*h) * (u * dOp - d * uOp)/(u-d)
    output_dic = {"Call_Up":uOp,
                  "Call_down":dOp,
                  "Stock_Up":uS,
                  "Stock_Down":dS,
                  "Up_factor":u,
                  "Down_factor":d,
                  "p_star":p,
                  "delta":sanjiao,
                  "B":B}
    return output_dic

r = 0.035
delta = 0.02
sigma = 1.2
h = 1/365
X = 120
S0 = 100
alpha = 0.3

Price_by_BT_2(r, sigma, delta, X, S0, h, alpha)
get_option_by_s(100, 100, 0.3)



class Option:
    def __init__(self, r, sigma, delta, X, S0, h, alpha):
        self.r = r
        self.sigma = sigma
        self.delta = delta
        self.X = X
        self.S0 = S0
        self.h = h
        self.alpha = alpha

    def One_step_BT(self):
        pass



def Build_Tree_E(n, sigma, delta, X, S0_P, h, alpha):
    option_matrix = []
    # item = {"Stock":100, "Option":0}
    option = [{"Stock":100, "Call":0}]
    option_matrix.append(option)
    for i in range(1, n):
        option_list = []
        for j in range(0, i):
            S0_P = option_matrix[i - 1][j]["Stock"]
            if j != i-1:
                SU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
                CU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Call_Up"]
                option = {"Stock":SU, "Call":CU}
                option_list.append(option)
            if j == i-1:
                SU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
                CU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Call_Up"]
                option = {"Stock":SU, "Call":CU}
                option_list.append(option)

                SD = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Down"]
                CD = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Call_down"]
                option = {"Stock":SD, "Call":CD}
                option_list.append(option)

        option_matrix.append(option_list)
    return option_matrix.append(option_list)


def Build_Tree_A(n, sigma, delta, X, S0_P, h, alpha):
    option_matrix = []
    # item = {"Stock":100, "Option":0}
    option = [{"Stock":100, "Call":0}]
    option_matrix.append(option)
    for i in range(1, n):
        option_list = []
        for j in range(0, i):
            S0_P = option_matrix[i - 1][j]["Stock"]
            if j != i-1:
                SU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
                CU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Call_Up"]
                C_cdt = X - SU
                if C_cdt > CU:
                    option = {"Stock":SU, "Call":C_cdt}
                else:
                    option = {"Stock":SU, "Call":CU}
                option_list.append(option)

            if j == i-1:
                SU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Up"]
                CU = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Call_Up"]
                C_cdt = X - SU
                if C_cdt > CU:
                    option = {"Stock":SU, "Call":C_cdt}
                else:
                    option = {"Stock":SU, "Call":CU}
                option_list.append(option)


                SD = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Stock_Down"]
                CD = Price_by_BT_2(r, sigma, delta, X, S0_P, h, alpha)["Call_down"]
                C_cdt = X - SD
                if C_cdt > CD:
                    option = {"Stock":SD, "Call":C_cdt}
                else:
                    option = {"Stock":SD, "Call":CD}
                option_list.append(option)

        option_matrix.append(option_list)
    return option_matrix.append(option_list)



Build_Tree_A(n, sigma, delta, X, S0_P, h, alpha)


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

    # pro.shibor(curr_type="RMB", start_date=sta_date, end_date=end_date)


    d_sta_date = datetime.datetime.strptime(sta_date, "%Y%m%d")
    d_end_date = datetime.datetime.strptime(end_date, "%Y%m%d")



# V = exp(-r*(T-t)) * max((ST-X), 0)
# St = S0 * e(mu-sigma-sigma**2/2 + sigma*sqrt(t))

K = X/(1 - alpha)
d1 = (log(ST/X) + r - delta + 0.5*sigma**2)/(sigma*sqrt(T-t))
d2 = d1 - sigma*sqrt(T-t)
d3 = (log(ST/K) + r - delta + 0.5*sigma**2)/(sigma*sqrt(T-t))
d4 = d3 - sigma*sqrt(T-t)

