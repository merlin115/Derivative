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
        """
        得到无风险利率
        :param type: 采用的是上海银行拆借利率shibor
        :return: 返回的是年利率，一个DataFrame
        """
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
    """
    得到一直股票的历史波动率
    :param df: 股票的历史数据，最好采用每天的收盘价
    :param method: 有两种方法，默认的是百分比法
    :return:
    """
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
    """
    这个期权与股票S的关系，分为三种情况
    :param ST: 股票在t=T的时候的价格
    :param X: Strike Price
    :param alpha: 定价因子alpha，在(0, 1)之间
    :return: 返回的是期权的价格
    """
    if alpha * ST >= ST - X >= 0:
        option = ST - X
    if ST - X > alpha * ST:
        option = alpha * ST
    if ST - X < 0:
        option = 0
    return option





def Price_by_BT_2(r, sigma, delta, X, S0, h, alpha = 0.3):
    """
    :param r: 无风险利率
    :param sigma: 历史波动率
    :param delta: 股息dividend
    :param X: Strike Price
    :param S0: 股票在t=0的价格
    :param h: 一次二叉树的时间
    :param alpha: 奇异期权的定价因子
    :return: 返回的是一个dictionary
    """

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



def Build_Tree_E(n, r, sigma, delta, X, S0_P, h, alpha):
    """
    欧式期权的二叉树
    :param n: 进行n次回归
    :param r: 无风险利率
    :param sigma: 历史波动率
    :param delta: dividend的
    :param X: Strike Price
    :param S0_P: 之前股票的价格
    :param h: 每次二叉树的间隔
    :param alpha: 这个是奇异期权的定价中的变量alpha
    :return: 返回是一个嵌套list
    """

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
    return option_matrix


def Build_Tree_A(n, sigma, delta, X, S0_P, h, alpha):
    """
    美式期权的二叉树
    :param n: 进行n次回归
    :param r: 无风险利率
    :param sigma: 历史波动率
    :param delta: dividend的
    :param X: Strike Price
    :param S0_P: 之前股票的价格
    :param h: 每次二叉树的间隔
    :param alpha: 这个是奇异期权的定价中的变量alpha
    :return: 返回是一个嵌套list
    """

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
    return option_matrix

r = 0.035
n = 10
delta = 0.02
sigma = 1.2
h = 1/365
X = 120
S0 = 100
alpha = 0.3
Build_Tree_A(n, sigma, delta, X, S0, h, alpha)


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


def BS_Formular(X, alpha, ST, T, t):
    K = X/(1 - alpha)
    d1 = (log(ST/X) + r - delta + 0.5*sigma**2)/(sigma*sqrt(T-t))
    d2 = d1 - sigma*sqrt(T-t)
    d3 = (log(ST/K) + r - delta + 0.5*sigma**2)/(sigma*sqrt(T-t))
    d4 = d3 - sigma*sqrt(T-t)

    
    
#######greeks
#European case
TL=[0.3,1,3]
plt.figure(figsize=(12,3))
plt.title("delta")
S0=St[0]
for T in TL:
    plt.plot(list(np.arange(S0*0.1,S0*1.5,S0/500)),[BS_Formular(X, alpha, x, T, 0)[0] for x in np.arange(S0*0.1,S0*1.5,S0/500)])
plt.legend(TL)

plt.figure(figsize=(12,3))
plt.title("gamma")
S0=St[0]
for T in TL:
    plt.plot(list(np.arange(S0*0.1,S0*1.5,S0/500)),[BS_Formular(X, alpha, x, T, 0)[1] for x in np.arange(S0*0.1,S0*1.5,S0/500)])
plt.legend(TL)

plt.figure(figsize=(12,3))
plt.title("theta")
S0=St[0]
for T in TL:
    plt.plot(list(np.arange(S0*0.1,S0*1.5,S0/500)),[BS_Formular(X, alpha, x, T, 0)[2] for x in np.arange(S0*0.1,S0*1.5,S0/500)])
plt.legend(TL)


#American case. Use binomial tree
def greeks_America(S):
    u=Price_by_BT_2(r, sigma, delta, X, S, h, alpha = 0.3)['Up_factor']
    d=Price_by_BT_2(r, sigma, delta, X, S, h, alpha = 0.3)['Down_factor']
    V=Build_Tree_A(3, sigma, delta, X, S0, h, alpha)
    V0=V[0][0]['Call']
    V11,V10=V[1][0]['Call'],V[1][1]['Call']
    V22,V21,V20=V[2][0]['Call'],V[2][1]['Call'],V[2][2]['Call']
    Δ=(V11-V10)/(S*(u-d))
    #有限差分法
    #gamma=((V22-V21)/(S*u*u-S)-(V21-V20)/(S-S*d*d))/(0.5*S*(u**2-d**2))
    #theta=(V21-V0)/(2)
    #binomial tree
    gamma=((V22-V21)/(S*u*u-d*S)-(V21-V20)/(S-S*d*d))/(0.5*S*(u**2-d**2))
    theta=(V21-V0)/2
    return [Δ,gamma,theta]

#American case
TL=[0.3,1,3]
plt.figure(figsize=(12,3))
plt.title("delta")
S0=St[0]
for T in TL:
    plt.plot(list(np.arange(S0*0.5,S0*1.5,S0/500)),[greeks_America(x)[0] for x in np.arange(S0*0.5,S0*1.5,S0/500)])
plt.legend(TL)

plt.figure(figsize=(12,3))
plt.title("gamma")
S0=St[0]
for T in TL:
    plt.plot(list(np.arange(S0*0.5,S0*1.5,S0/500)),[greeks_America(x)[1] for x in np.arange(S0*0.5,S0*1.5,S0/500)])
plt.legend(TL)

plt.figure(figsize=(12,3))
plt.title("theta")
S0=St[0]
for T in TL:
    plt.plot(list(np.arange(S0*0.5,S0*1.5,S0/500)),[greeks_America(x)[2] for x in np.arange(S0*0.5,S0*1.5,S0/500)])
plt.legend(TL)

#delta hedging 收益
def deltaHedging(St,option,T,opType):
    '''
    St: time series格式，股价
    option: time series格式，期权价格
    S0: t=0的时候股价
    n: 到expiration共多少天
    opType: str,"e"为欧式,"a"为美式
    '''
    n=T*192
    df=[]
    for i in range(n):
        t=i/192
        S=St[i]
        V=option[i]
        if opType=="e":
            delta,gamma,theta=BS_Formular(X, alpha, S, T, t)
        if opType=="a":
            delta,gamma,theta=greeks_America(S)
        if i>0:
            gain_on_shares=100*delta*(S-St[i-1])
            gain_on_written_option=100*(option[i-1]-V)
            interest=-net_inv*(np.exp(r/365)-1) #使用的是昨天的net_inv
            overnight_profit=gain_on_shares+gain_on_written_option+interest
        else:
            gain_on_shares=0
            gain_on_written_option=0
            interest=0
            overnight_profit=0
        net_inv=100*(delta*S-V)
        df.append([S,V,delta,gamma,theta,net_inv,gain_on_shares,gain_on_written_option,interest,overnight_profit])
    df=pd.DataFrame(df,index=St.index[:n],columns=['S','V','delta','gamma','theta','net_inv','gain_on_shares'
                                               ,'gain_on_written_option','interest','overnight_profit'])
    df['net_inv']=df['net_inv'].shift(1)
    print("total profit",np.sum(df["overnight_profit"]))
    return df.apply(lambda x:round(x,4))
T=1
df=deltaHedging(St,option,T,"e")
df









