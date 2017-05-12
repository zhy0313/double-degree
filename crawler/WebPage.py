# -*- coding: utf-8 -*-
import re
import logging
import requests
import urllib2
import urllib
import time
from Config import *
import sys
log = logging.getLogger('Main.WebPage')

class WebPage(object):
    def __init__(self,url, headers=None, parameters=None):
        self.url = url
        self.pageSource = None
        self.customHeaders(headers)
        self.customParameters(parameters)


    def customHeaders(self,headers):
        global HEADERS
        self.headers = HEADERS
        if headers != None:
            self.headers.update(headers)

    def customParameters(self,parameters):
        global PARAMTERS
        self.paramters = PARAMTERS
        if parameters != None:
            self.paramters.update(parameters)

    # 定制url，%代表占位符
    def customUrl(self):
        pageIndex = self.paramters['page']
        self.url = self.url %pageIndex


    def customRequest(self,isGet):
        if isGet:
            self.customUrl()
            request = urllib2.Request(url=self.url, headers=self.headers)  # get方式
        else:
            encodedParamters = urllib.urlencode(self.paramters)
            request = urllib2.Request(url=self.url, data=encodedParamters, headers=self.headers)  # post方式
        return request

    def fetch(self,isGet=False):
        try:
            request = self.customRequest(isGet)
            response = urllib2.urlopen(request)
            self.pageSource = response.read()
        except Exception,e:
            print e

