# -*- coding: utf-8 -*-
from Storage import *
from Config import *

class DatabaseStorage(Storage):
    def __init__(self, config=None):
        if config == None:
            config = self._default()
        self.connection = Connection(host=config['host'], user=config['user'], password=config['password'],
                                     database=config['database'], port=config['port'], charset=config['charset'])
        self.cursor = self.connection.cursor()

    def add(self, row):
        global LOCK
        LOCK.acquire()
        sql = 'insert into east_money values(%s,%s,%s,%s,%s,%s)'
        self.cursor.execute(sql, row)
        self.connection.commit()
        LOCK.release()

    def batchAdd(self,rows):
        for row in rows:
            self.add(row)

    def _default(self):
        __default = {
            'host': '121.192.191.70',
            'user': 'root',
            'password': 'cdYYadmin',
            'database': 'policyDB',
            'table': 'east_money',
            'port': 3306,
            'charset': 'utf8'
        }
        return __default