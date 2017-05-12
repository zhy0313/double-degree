# -*- coding: utf-8 -*-
from CustomParser import *

# 解析html格式，提取数据
class CommentParser(CustomParser):
    def __init__(self,content):
        CustomParser.__init__(self,content)
    def parse(self):
        article_list = self.bsoup.find_all('div', 'articleh')
        for article in article_list:
            read_count = article.find('span', 'l1').string
            comment_count =  article.find('span', 'l2').string
            title = article.find('span', 'l3').find('a').string
            author = article.find('span','l4').find('a').string if article.find('span','l4').find('a') != None else article.find('span','l4').find('span').string
            release_date = '2016-' + article.find('span','l6').string
            print read_count, comment_count, title, author, release_date
            self.result.append((read_count, comment_count, title, author, release_date))

