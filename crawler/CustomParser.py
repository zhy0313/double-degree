from bs4 import BeautifulSoup
from Config import *


class CustomParser(object):
    def __init__(self,content):
        self.result = list()
        bsoup = BeautifulSoup(content, 'lxml')
        self.bsoup = bsoup

    def parse(self):pass
