import sys
import json
import unicodedata

segments = json.load(open('./mylib/IDcode/HL7_segment.json', 'r'))

class SSMIXparser():
    def __init__(self, names):
        self.names = names
        keys = [
            'pID',
            'date',
            'type',
            'key',
            'high time',
            'department',
            'condition'
            ]
        self.dic = dict(file=dict(zip(keys, self.names))) #インスタンス化の際に受け取った(分割された)ファイル名(複数の情報を保持するリスト)を辞書のvalueとし、その辞書をvalueとして持つ辞書を作成
    
    def parse(self, line):
        data = line.strip().split("|")
        data = [unicodedata.normalize("NFKC", d) for d in data] #正規化

        if data[0] in self.dic: dic_list = self.dic.get(data[0]) #解析対象セグメント(行の冒頭)が既に辞書中に登録されていれば、新しい情報を追加
        else:                   dic_list = [] #登録されていなければ、空のdic_listを作成

        seg = segments.get(data[0]) #解析対象セグメントが持ち得る情報をsegmentsから取得
        sub_dic = {}
        if len(seg) >= len(data): #解析している行のデータサイズが、該当セグメントが持つ情報のサイズ上限を超えていなければ
            for i in range(1, len(data)):
                key_name = data[0] + "-" + str(i) #セグメントの値に添え字を付けて
                sub_dic.update({key_name: data[i]}) #sub_dicに追加、これをデータ数分繰り返す(内容例：[セグメント名-1:データ1, セグメント名-2:データ2, ...])
            dic_list.append(sub_dic) #dic_listにsub_dicを追加
            self.dic.update({data[0]: dic_list}) #"セグメント：dic_list"という形式で更新(key：セグメント、value：その内容)
        else: #解析している行のデータサイズが、該当セグメントが持つ情報のサイズ上限を超えていたらエラー出力
            print("\t", self.dic['file']['type'],  len(seg), len(data), file=sys.stderr)
            print(f'{line}\n{data}', file=sys.stderr)

    def getData(self):
        if self.dic['file']['type'] == 'PPR-01': 
            self.dic['PRB'] = sorted(self.dic['PRB'], key=lambda x: x['PRB-7'])  # プロブレム設定日時(診断日時)順にsort
        return self.dic
