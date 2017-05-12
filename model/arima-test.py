# coding=utf-8
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

####股票时间序列分析####
#参数初始化
datafile='../data/stock.csv'
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#读取数据
data=pd.read_csv(datafile,sep='\t',index_col='dates',parse_dates=True)
data=DataFrame(data,dtype=np.float64)


#时序图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
data.plot()
plt.title('三只股票时序图')


#自相关数
from statsmodels.graphics.tsaplots import plot_acf
#AAPL 取前100条数据
plot_acf(data['AAPL'].iloc[:100]).show()
plt.title('AAPL自相关图')
#MSFT 取前100条数据
plot_acf(data['MSFT'].iloc[:100]).show()
plt.title('MSFT自相关图')
#XOM 取前100条数据
plot_acf(data['XOM'].iloc[:100]).show()
plt.title('XOM自相关图')
#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
#AAPL的平稳性检测，看pvalue值
print 'AAPL的pvalue:',ADF(data['AAPL'])[1] #pvalue值为0.99,大于0.05,不稳定，考虑差分数值建模
#MSFT的平稳性检测，看pvalue值
print 'MSFT的pvalue:',ADF(data['MSFT'])[1] #pvalue值为0.04,小于0.05,稳定
#XOM的平稳性检测，看pvalue值
# print 'MSFT的pvalue:',ADF(data['XOM'])[1]  #pvalue值为0.45,大于0.05,不稳定

###差分
D_data=data.diff().dropna()
D_data.columns=[u'AAPL差分',u'MSFT差分',u'XOM差分']

##差分后的结果
#差分时序图
D_data.plot()
plt.title('三只股票差分时序图')
#差分自相关数
from statsmodels.graphics.tsaplots import plot_acf
#AAPL差分 取前100条数据
plot_acf(D_data['AAPL差分'].iloc[:100]).show()
plt.title('AAPL差分自相关图')
#MSFT差分 取前100条数据
plot_acf(D_data['MSFT差分'].iloc[:100]).show()
plt.title('MSFT差分自相关图')
#XOM差分 取前100条数据
plot_acf(D_data['XOM差分'].iloc[:100]).show()
plt.title('XOM差分自相关图')
#偏自相关图
from statsmodels.graphics.tsaplots import plot_pacf
#AAPL差分偏自相关 取前100条数据
plot_pacf(D_data['AAPL差分'].iloc[:100]).show()
plt.title('AAPL差分偏自相关图')
#MSFT差分偏自相关 取前100条数据
plot_pacf(D_data['MSFT差分'].iloc[:100]).show()
plt.title('MSFT偏差分自相关图')
#XOM差分偏自相关 取前100条数据
plot_pacf(D_data['XOM差分'].iloc[:100]).show()
plt.title('XOM偏差分自相关图')
#差分平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
#AAPL差分的平稳性检测，看pvalue值
print 'AAPL的pvalue:',ADF(D_data['AAPL差分'])[1] #远小于0.05,稳定！
#MSFT差分的平稳性检测，看pvalue值
print 'MSFT的pvalue:',ADF(D_data['MSFT差分'])[1] #远小于0.05,稳定！
#XOM差分的平稳性检测，看pvalue值
print 'MSFT的pvalue:',ADF(D_data['XOM差分'])[1]  #远小于0.05,稳定！

#白噪声检测
from statsmodels.stats.diagnostic import acorr_ljungbox
#AAPL差分白噪声检测
acorr_ljungbox(D_data['AAPL差分'],lags=1) #返回统计量和p值  p值大于0.05，不排除白噪声
#MSFT差分白噪声检测
acorr_ljungbox(D_data['MSFT差分'],lags=1) #返回统计量和p值  p值小于0.05，排除白噪声
#XOM差分白噪声检测
acorr_ljungbox(D_data['XOM差分'],lags=1) #返回统计量和p值   p值小于0.05，排除白噪声



#AAPL  ARIMA建模
# from statsmodels.tsa.arima_model import  ARIMA
# pmax=3
# qmax=3
# bic_matrix=[] #bic矩阵
# for p in range(pmax+1):
#     tmp=[]
#     for q in range(qmax+1):
#         try: #存在部分报错，所以用try来跳过报错。
#             tmp.append(ARIMA(data['AAPL'], (p,1,q)).fit().bic)
#         except:
#             tmp.append(None)
#     bic_matrix.append(tmp)
#
# bic_matrix=pd.DataFrame(bic_matrix) #从中可找出最小值
# p,q=bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小位置。
# print u'BIC最小的p值和q值为: %s、%s'%(p,q) #0,0
# model=ARIMA(data['AAPL'],(p,1,q)).fit() #建立ARIMA(0,1,0)模型
# model.summary() #股票AAPL给出一份模型报告

#MSFT  ARIMA建模
from statsmodels.tsa.arima_model import  ARIMA
pmax=3
qmax=3
bic_matrix=[] #bic矩阵
for p in range(1,pmax+1):
    tmp=[]
    for q in range(1,qmax+1):
        try: #存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data['MSFT'],(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix) #从中可找出最小值
p,q=bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小位置。
print u'BIC最小的p值和q值为: %s、%s'%(p,q) #0,1
model=ARIMA(data['MSFT'],(p,1,q)).fit() #建立ARIMA(0,1,1)模型
model.summary() #给出一份模型报告


#XOM  ARIMA建模
# from statsmodels.tsa.arima_model import  ARIMA
# pmax=3
# qmax=3
# bic_matrix=[] #bic矩阵
# for p in range(pmax+1):
#     tmp=[]
#     for q in range(qmax+1):
#         try: #存在部分报错，所以用try来跳过报错。
#             tmp.append(ARIMA(data['XOM'],(p,1,q)).fit().bic)
#         except:
#             tmp.append(None)
#     bic_matrix.append(tmp)
# bic_matrix=pd.DataFrame(bic_matrix) #从中可找出最小值
# p,q=bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小位置。
#
# print u'BIC最小的p值和q值为: %s、%s'%(p,q) #2,3
# model=ARIMA(data['XOM'],(p,1,q)).fit() #建立ARIMA(2,1,3)模型
# model.summary() #给出一份模型报告result: