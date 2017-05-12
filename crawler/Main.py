# -*- coding: utf-8 -*-
from CustomThread import *
from Config import *
import tushare as ts
if __name__ == '__main__':
    #股评数据获取
    # thread1 = CustomThread('http://guba.eastmoney.com/list,szzs,f_%d.html', 1030, 3555)
    # thread1.start()
    # 股价数据获取
    # ts.get_hist_data('sh',start='2016-08-25',end='2017-03-14').to_csv('../data/history-szzs.csv',sep='\t')
    print ts.get_hist_data('sh', start='2017-03-15',end='2017-03-23')['close']