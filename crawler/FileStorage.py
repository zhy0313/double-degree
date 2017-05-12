# -*- coding: utf-8 -*-
from Storage import *
from Config import *
import pandas as pd
class FileStorage(Storage):
    def __init__(self, config=None):
        if config == None:
            config = self._default()
        self.file = config['file']

    def add(self, row):
        pass


    def batchAdd(self,rows):
        pd.DataFrame(rows).to_csv(self.file,encoding='UTF-8',sep="\t",index=False, header=None, mode='a+')

    def _default(self):
        __default = {
            'file': '../data/comment-szzs.csv'
        }
        return __default