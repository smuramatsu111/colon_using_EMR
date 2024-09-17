import re
import unicodedata
from html.parser import HTMLParser
from mylib.format_data import format_data

class XMLparser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.body_flag  = False
        self.title_flag = False
        self.text_flag  = False
        self.main_title = ""
        self.title = []
        self.text  = []
        self.pid = {}

    def handle_starttag(self, tag, attrs):
        if tag == "structuredbody": self.body_flag  = True
        elif tag == "title":        self.title_flag = True
        elif tag == "text":         self.text_flag  = True
        elif tag == "administrativegendercode": self.pid.update({"PID-8": attrs[0][1]}) #性別情報を抽出
        elif tag == "birthtime":                self.pid.update({"PID-7": attrs[0][1]}) #年齢情報を抽出

    def handle_endtag(self, tag):
        if tag == "structuredbody": self.body_flag  = False
        elif tag == "title":        self.title_flag = False
        elif tag == "text":         self.text_flag  = False

    def handle_data(self, data):  
        data = unicodedata.normalize("NFKC", data) #正規化
        if self.body_flag:
            if self.title_flag:
                if   data == 'SUBJECTIVE DATA'   or data == '主訴':           data = 'S'
                elif data == 'OBJECTIVE DATA'    or data == '所見':           data = 'O'
                elif data == 'ASSESSMENTS'       or data == '現疾患(診断内容)': data = 'A'
                elif data == 'PLAN OF TREATMENT' or data == '計画':           data = 'P'
                elif data == 'Jspeed DOCUMENTATION':                          data = 'J-SPEED'
                self.title.append(data)

            elif self.text_flag:
                data = format_data(data) #正規化・置換処理
                self.text.append(data)
        else:
            if self.title_flag:  self.main_title = data

    def getData(self):
        return self.main_title, self.title, self.text, self.pid

