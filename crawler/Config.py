# -*- coding: utf-8 -*-
import threading

URL = 'http://guba.eastmoney.com/list,000880,f_%d.html'

LOCK = threading._allocate_lock()

THREAD_NUM = 10

TOTALPAGE = 50

PARAMTERS = {}

HEADERS = {
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language':'zh-CN,zh;q=0.8',
    'Cache-Control':'max-age=0',
    'Connection':'keep-alive',
    'Host':'guba.eastmoney.com',
    'Upgrade-Insecure-Requests':1,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
}



