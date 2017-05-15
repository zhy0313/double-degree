# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import sys
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings("ignore")
import arrow
# 读取数据
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
data = pd.read_csv('../data/history-szzs.csv',sep='\t',parse_dates='date',index_col='date',date_parser=dateparse)
data = pd.DataFrame.sort_index(data) #从小到大排序
# print data.head()
ts = data['close']

senti_data = pd.read_csv('../data/szzs-sentiment2.csv',sep='\t',parse_dates='date',index_col='date',date_parser=dateparse)
senti_data = pd.DataFrame.sort_index(senti_data) #从小到大排序
senti_ts = senti_data['bi_change']
# 滚动统计
def rolling_statistic(timeseries):
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Debiation')
    plt.show()

# ADF检验
def adf_test(timeseries):
    rolling_statistic(timeseries)
    print 'Results of Fickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput

# 分解
def decompose(ts_log):
    decomposition = seasonal_decompose(ts_log)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413);
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

# 自相关图和偏自相关图
def plot_acf_pacf(ts_log_diff):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sm.graphics.tsa.plot_acf(ts_log_diff, lags=40, ax=ax1)  # ARIMA,q
    sm.graphics.tsa.plot_pacf(ts_log_diff, lags=40, ax=ax2)  # ARIMA,p

# p,q调参
def _proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_bic = sys.maxint
    best_model=None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(ts_log_diff, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1,method='css')
            except:
                continue
            bic = results_ARMA.bic
            print bic, best_bic
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
    print best_p,best_q
    return best_p,best_q

# 差分操作
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list #这个序列在恢复过程中需要用到
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print last_data_shift_list
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

# 还原操作，和差分操作配套使用
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data # return np.exp(tmp_data)也可以return到最原始，tmp_data是对原始数据取对数的结果

# 训练模型
def fit(ts_log,exog=None):
    d = [1]
    ts_log_diff = diff_ts(ts_log, d) # 差分操作
    # adf_test(ts_log_diff)
    # p,q = _proper_model(ts_log_diff, 8)
    p=2
    q=1
    exog = exog[ts_log.index]
    model = ARIMA(ts_log, (p, 1, q),exog)
    results_ARIMA = model.fit(disp=-1)
    plt.plot(ts_log_diff,color='blue')
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_log_diff) ** 2))
    plt.show()
    return results_ARIMA

# 数据还原操作
def predict(results_ARIMA):
    # 上述方法比较复杂，现在使用封装的方法
    diff_recover_ts = predict_diff_recover(results_ARIMA.predict(), d=[1])  # 恢复数据
    predicted = np.exp(diff_recover_ts)  # 还原对数前数据
    # 绘图
    original = ts[predicted.index]  # 排除空的数据
    plt.plot(original, color="blue", label='Original')
    plt.plot(predicted, color='red', label='Predicted')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % np.sqrt(sum((predicted - original) ** 2) / len(original)))  # RMSE,残差平方和开根号，即标准差
    plt.show()

# 获取时间范围
def get_date_range(start, limit, level='month',format='YYYY-MM-DD'):
    start = arrow.get(start, format)
    result =  (list(map(lambda dt: dt.format(format) , arrow.Arrow.range(level, start, limit=limit))))
    dateParse = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
    return map(dateParse, result)


def forecast(resultARIMA, start, n, exog=None):
    new_index = get_date_range(start, n, level='day')
    forecast_ARIMA_log = resultARIMA.forecast(n,exog=exog[start:new_index[-1]])
    forecast_ARIMA_log = forecast_ARIMA_log[0]
    forecast_ARIMA_log = pd.Series(forecast_ARIMA_log, copy=True, index=new_index)
    forecast_ARIMA = np.exp(forecast_ARIMA_log)
    plt.plot(ts, label='Original', color='blue')
    plt.plot(forecast_ARIMA, label='Forcast', color='red')
    plt.legend(loc='best')
    plt.title('forecast')
    plt.show()

# 对数变换
ts_log = np.log(ts)
ARIMA_result = fit(ts_log,senti_ts)
predict(ARIMA_result)
forecast(ARIMA_result, '2017-03-13', 2, senti_ts)




