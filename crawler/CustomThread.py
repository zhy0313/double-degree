import threading,thread
from WebPage import *
from CommentParser import *
from DatabaseStorage import *
from FileStorage import *
class CustomThread(threading.Thread):
    def __init__(self, url, fromIndex, toIndex):
        threading.Thread.__init__(self)
        self.storage = FileStorage()
        self.range = xrange(fromIndex, toIndex)
        self.url = url
    def run(self):
        for i in self.range:
            try:
                webpage = WebPage(url=self.url, parameters={'page':i})
                webpage.fetch(isGet=True)
                if webpage.pageSource != None:
                    parser = CommentParser(webpage.pageSource)
                    parser.parse()
                    self.storage.batchAdd(parser.result)
            finally:
                print "page ", str(i), " SUCCESS!"

        thread.exit_thread()