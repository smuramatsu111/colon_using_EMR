import os
import json
import warnings
import zipfile

dataset_path = '抽出・分類したデータの保存先を指定' #parser_zip.pyのdataset_pathと同じパスを指定

class statistic_Dict:
    def __init__(self, pair_id) -> None:
        self.pair_id = pair_id # ファイル名の一部を利用したもの(渡邊さんの頃のファイル構成に倣っているため、要変更)
        self.xml_pset   = set()
        self.ssmix_pset = set()
        self.PPR_pset   = set()
        self.statistic_dic = {
            "patient": 0,
            "pair":{"patient": 0, "pidList": []},
            "xml": {"file": 0, "type":{}, "xml_only": []},
            "ssmix": {"file": 0, "type":{}, "ssmix_only": [], "PPR-01": [], "no_PPR-01": []}
        }
        self.colon_dic = {"patient": 0}


    def fnum_update(self, id): #ファイル数をxml, hl7についてそれぞれカウントする
        if   id == 0: self.statistic_dic['xml']['file'] += 1  # xml
        elif id == 1: self.statistic_dic['ssmix']['file'] += 1 # ssmix
            
    def patient_add(self, pid, id): #xml, hl7それぞれについて、ファイルを持つ患者の患者IDを記録する
        if   id == 0: self.xml_pset.add(pid)   # pid in xml
        elif id == 1: self.ssmix_pset.add(pid) # pid in ssmix

    def type_update(self, ftype, id, **kwargs): #ファイルの種類を記録する
        if   id == 0: f_dic = self.statistic_dic['xml']['type']   # xml file type
        elif id == 1: f_dic = self.statistic_dic['ssmix']['type'] # ssmix file type

        if 'title' in kwargs: ftype = f"{ftype}_{kwargs['title']}" #titleがkwargs内にあれば、「ftype」を「ftype_タイトル」に変更(例：経過記録_診察記事)
        elif 'pid' in kwargs: self.PPR_pset.add(kwargs['pid']) #pidがkwargsにあれば、PPR_psetにpidを追加

        if ftype in f_dic: f_dic[ftype] += 1 #f_dicの中に既にftypeがあれば(過去に同じ種類のファイルを処理していれば)、カウントを行う
        else:              f_dic.update({ftype: 1}) #なければf_dicを更新する

        if   id == 0: self.statistic_dic['xml']['type'] = f_dic # 更新処理
        elif id == 1: self.statistic_dic['ssmix']['type'] = f_dic # 更新処理

    def fname_update(self, pid, date, fname, **kwargs):
        p_path = f"{dataset_path}/pid/{pid}.json"
        new_zip = f'{dataset_path}/{pid}.zip'

        if not os.path.isfile(p_path): #p_pathにファイルが無ければ
            sub_dic = {date: [fname]} #sub_dicを作って
            with open(p_path, 'w', encoding='utf-8') as json_file:
                json.dump(sub_dic, json_file, indent="\t", ensure_ascii=False) #sub_dicの中身をjson_fileに書き出す(ensure_asciiで文字化けを防止)
        else: #ファイルがあれば
            with open(p_path, 'r', encoding='utf-8') as json_file: # 読み込んで
                pid_dic = json.load(json_file) # その内容をpid_dicに格納
            
            if date in pid_dic: pid_dic[date].append(fname) #もしpid_dic内にdateがあれば、pid_dicのdateにfnameを追加
            else:               pid_dic.update({date: [fname]}) #なければ、date:[fname]で更新

            with open(p_path, 'w', encoding='utf-8') as json_file: #p_pathにあるファイルを書き出し用で読み込んで
                json.dump(pid_dic, json_file, indent="\t", ensure_ascii=False) #pid_dicの内容を書き出し
                

        try: #ここから下でzipファイル作成
            warnings.simplefilter('error')

            #l.acquire()
            if 'type' in kwargs: 
                if kwargs['type'] == 'ADT-00': #ssmixのftypeがADT-00なら(ADT-00：患者基本情報の削除・更新)
                    return
            tmp_f = f'{dataset_path}/tmp/{pid}/{date}/' + fname #parser_zip.pyでtmpディレクトリ内に書き込んだファイルを参照
            afname = f'/{date}/' + fname # ファイル名作成
            with zipfile.ZipFile(new_zip, "a", compression=zipfile.ZIP_DEFLATED) as nzf: #第二引数a：既存のzipに新規ファイル追加、compressionは圧縮方式、第一引数は書き込むファイルの場所
                nzf.write(tmp_f, arcname=afname) #第一引数のファイルを、arcnameで指定した名前でzipファイルに書き込む
            os.remove(tmp_f)

        except UserWarning: #UserWarningが出た場合
            warnings.resetwarnings()
            UW_path = f'{dataset_path}/UserWarning_file.txt'
            with open(UW_path, 'w') as f:
                print(tmp_f, file=f)
            with zipfile.ZipFile(new_zip, "a", compression=zipfile.ZIP_DEFLATED) as nzf:
                nzf.write(tmp_f, arcname=fname)

        


    def colon_update(self, pid, **kwargs):
        if pid in self.colon_dic: self.colon_dic[pid].append(kwargs) #colon_dicの中にpidがあるかを確認、あればkwargsの内容を対象のpid部分に追記
        else:                     self.colon_dic.update({pid: [kwargs]}) #無ければ、colon_dicをpid:[kwargs]で更新
    
    # 統計情報の出力
    def output(self):
        pair = self.pair_id.split('-')[0] #pair_idを「-」で分割して、前半部分を取得(要変更)
        pair_path = f'{dataset_path}/{pair}.json' # 統計情報を出力するファイルまでのパス

        if os.path.isfile(pair_path): #ファイルがあれば
            with open(pair_path, 'r') as json_file: #読み込み用でファイルを開いて
                s_dic= json.load(json_file) #s_dicに読み込み
            self.xml_pset   |= set(s_dic['xml']['xml_only']) #既存のsetと新しいsetの和集合を取る
            self.ssmix_pset |= set(s_dic['ssmix']['ssmix_only'])
            self.PPR_pset   |= set(s_dic['ssmix']['PPR-01'])

            if self.statistic_dic['xml']['file'] == 0: 
                self.statistic_dic['xml']['file'] = s_dic['xml']['file'] 
            if self.statistic_dic['xml']['type'] == {}:
                self.statistic_dic['xml']['type'] = s_dic['xml']['type'] 
            if self.statistic_dic['ssmix']['file'] == 0:
                self.statistic_dic['ssmix']['file'] = s_dic['ssmix']['file'] 
            if self.statistic_dic['ssmix']['type'] == {}:
                self.statistic_dic['ssmix']['type'] = s_dic['ssmix']['type'] 
    

        # 患者の属性ごとに集計
        comb = self.xml_pset | self.ssmix_pset
        pair = self.xml_pset & self.ssmix_pset
        self.statistic_dic['patient'] = len(comb)
        self.statistic_dic['pair']['patient'] = len(pair)
        self.statistic_dic['pair']['pidList']     = sorted(pair)
        self.statistic_dic['xml']['xml_only']     = sorted((self.xml_pset - self.ssmix_pset))
        self.statistic_dic['ssmix']['ssmix_only'] = sorted((self.ssmix_pset - self.xml_pset))
        self.statistic_dic['ssmix']['PPR-01']     = sorted(self.PPR_pset)
        self.statistic_dic['ssmix']['no_PPR-01']  = sorted((self.ssmix_pset - self.PPR_pset))
        
        # 統計情報をjsonファイルに出力
        with open(pair_path, 'w') as json_file:
            json.dump(self.statistic_dic, json_file, indent="\t", ensure_ascii=False)
        
        colon_path = f'{dataset_path}/colon.json'
        if os.path.isfile(colon_path):
            with open(colon_path, 'r') as json_file:
                self.colon_dic.update(json.load(json_file))

        self.colon_dic['patient'] = len(self.colon_dic) - 1
        with open(colon_path, 'w') as json_file:
            json.dump(self.colon_dic, json_file, indent="\t", ensure_ascii=False)        
