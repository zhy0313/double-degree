# -*- coding: utf-8 -*-
from pymysql import Connection

#抽象的存储接口
class Storage(object):
    def __init__(self):
        pass
    #添加一条数据
    def add(self, row):
        pass
    #批量添加数据
    def batchAdd(self,rows):
        pass
