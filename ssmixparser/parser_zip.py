import csv
import os
import re
import sys
import glob
import json
import shutil
from time import time
import zipfile
import unicodedata

from datetime import datetime
from tqdm import tqdm
from stream_unzip import stream_unzip

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from mylib.XMLparser import XMLparser
from mylib.SSMIXparser import SSMIXparser
from mylib.Statistic import statistic_Dict
from mylib.Decoder import byte_decode
from mylib.logger_time import setup_logger, timer

# ----------
# global
# ----------
colon_path = '電子カルテファイルが保存されているディレクトリまでのパスを指定'
dataset_path = '抽出・分類したデータの保存先を指定'
segments = json.load(open('./mylib/IDcode/HL7_segment.json', 'r')) #HL7ファイルのセグメント情報を取得

# ----------
# method
# ----------
def xml_contents(pid, ftype, names, chunks, statistic):
    # XML -> 抽出
    data = byte_decode(chunks) #バイト型のデータをデコード(復元)
    data = re.sub('<ClinicalDocument.*>', '<ClinicalDocument>', data) #文字列変換
    parser = XMLparser()
    parser.feed(data)
    main_title, titles, texts, personal = parser.getData() #XMLparserで解析した内容を取得(主題(例：診察記事)、SOAP(タイトル)とそれぞれに対応するテキストなど)

    # 統計
    statistic.fnum_update(0) #引数0➡xmlファイル数をカウント
    statistic.patient_add(pid, 0) #xmlファイルを持つ患者のリストに追記
    statistic.type_update(ftype, 0, title=main_title) #ファイルタイプと主題の情報を組み合わせて辞書を更新

    return pid, ftype, main_title, titles, texts, personal

def ssmix_contents(pid, names, chunks, statistic):
    # SSMIX -> Dict
    data = byte_decode(chunks) #エンコード
    parser = SSMIXparser(names) #インスタンス化

    for line in data.splitlines(): #splitlines:改行コード毎に区切る、改行コードは削除される
        while len(line) >= 4: #各行の文字数を確認、4文字以上なら処理
            check = True
            for skey in segments: #segments：プログラム冒頭で定義したHL7_segment.jsonの内容、辞書形式でhl7ファイルのセグメント構成がまとめてある(skeyの例：MSH,EVN,PIDなど)
                segid = "|" + skey + "|"
                index = line[3:].find(segid) #各行について、冒頭のskey(4文字目まで)以降に別のskeyが有るか探す。有ったらその開始位置を取得、無かったら-1を返す(一行に複数のセグメント情報が含まれているパターンを考慮)
                if index != -1: #skeyがあれば、
                    check = False
                    if line[0:3] in segments: #最初のセグメントがskeyかどうかを確認
                        parser.parse(line[:(index+3)]) #最初のセグメントがskeyなら(=1行にskeyが2つあれば)、前半部分のみを解析
                    else: 
                        print(f'{names}\n{line[:(index + 3)]}', file=sys.stderr) #そうでなければ、エラー出力

                    line = line[(index+3):] #後半部分をlineに代入
                    break

            if check: #4文字目以降にセグメントが見つからない場合
                if line[0:3] in segments: #冒頭4文字がセグメントかどうかを確認
                    parser.parse(line) #セグメントなら解析
                else: 
                    print(f'{names}\n{line}', file=sys.stderr) #そうでなければ、エラー出力
                break
    
    dic = parser.getData() #解析結果を取得

    # 統計
    statistic.fnum_update(1) #引数1➡ssmixファイルをカウント
    statistic.patient_add(pid, 1) #ssmixファイルを持つ患者のリストに追記

    if names[2] == 'PPR-01': #PPR-01ファイルであれば
        statistic.type_update(names[2], 1, pid=pid) #PPR-01ファイルを持つ患者のリストに追記
        # 病歴確認
        for byoreki in dic['PRB']:
            icd10 = byoreki['PRB-10'].split('^')[0] #PRB-10の一つ目の値：ICD10コード
            if re.match(r'C18[0-9]', icd10) or icd10 == 'C19' or icd10 == 'C20':
                code = byoreki['PRB-3'].split('^')[0] #病名管理番号の抽出

                if icd10 == 'C20' and (code not in ['20070958', '20070961']): #20070958：直腸がん、20070961：直腸癌術後再発、これらでなければ
                    continue

                if byoreki['PRB-13'] == "": suspicion = 0 #PRB-13：病気などの検証状態、疑わしさが記載、「1」が疑いあり
                else:                       suspicion = int(byoreki['PRB-13'].split('^')[0])
                # 大腸がん患者をまとめたリストの更新
                statistic.colon_update(
                    pid, 
                    icd10=icd10, 
                    code=code, 
                    suspicion=suspicion, 
                    conclusion=byoreki['PRB-14'].split('^')[0], #PRB-14：特定の日付/時刻における病気などの状態(アクティブ、アクティブで改善中、解決された、アクティブで悪化など)、転機区分として用いられる
                    rank=int(byoreki['PRB-18'].split('^')[0])
                )
    else:
        statistic.type_update(names[2], 1)
        
    return dic
    

#以下2つはstream_unzipのgithubを基に記述
# open zip
def get_zipped_chunks(zip_file):
    with open(zip_file, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            yield chunk

# extract zip
def get_unzipped_chunks(zipped_chunks, statistic): #ファイル情報と統計インスタンスを受け取る
    with ThreadPoolExecutor(max_workers=1000, thread_name_prefix="thread") as executor:
        for file_name, file_size, unzipped_chunks in stream_unzip(zipped_chunks): #stream_unzipでファイルを展開、ファイル名、サイズ、unzipped_chunksを受け取る
            if file_size > 0:
                file_name = file_name.decode('cp932') #decode(解読)
                file_name = unicodedata.normalize("NFKC", file_name) #正規化

                chunks = b''
                for chunk in unzipped_chunks: #stream_unzipで取得した内容をchunksに追加
                    chunks += chunk

                ext = os.path.splitext(file_name)[1] #拡張子を取得
                if ext == '.XML' or ext == '.xml': #拡張子がxmlなら
                    # 命名規則による抽出
                    names = os.path.dirname(file_name).split("/")[-2].split("_") #ディレクトリを"/"で区切って、後ろから2番目を取得し、"_"で区切る
                    if names[-1] != '1': continue # ファイルのコンディションフラグを参照(1が最新、0は旧ファイル)
                    pid = names[0].zfill(10) #ゼロパディングしてpidとする
                    ftype = names[2].split("^")[1] # ftypeを取得(例：経過記録など)

                    # skipするデータ
                    if ftype == 'カルテ2号紙': ftype = '経過記録' 
                    elif ftype == '災害時JSPEED記録': continue

                    future = executor.submit(xml_contents, pid, ftype, names, chunks, statistic)

                    # zip化(出力)
                    if future.result():
                        pid, ftype, main_title, titles, texts, personal = future.result() #xml_contentsの結果を取得
                        tmp_dir = f'{dataset_path}/tmp/{pid}/{names[1]}/' #names[1]：日付
                        os.makedirs(tmp_dir, exist_ok=True)
                        # 正規化
                        for i, (title, text) in enumerate(zip(titles, texts)):
                            main_title=main_title.replace(' ', '')#正規化
                            main_title=main_title.replace('/', '')
                            main_title=main_title.replace('_', '-')
                            title=title.replace(' ', '')
                            title=title.replace('/', '')
                            title=title.replace('_', '-')
                            path = f'{names[-3]}_{i}_{ftype}_{main_title}_{title}.txt' #names[-3]:日付、i：添え字、ftype(経過記録など)、main_title(プログレスノートなど)、title(SOAPなど)
                            tmp_f = tmp_dir + path
                            if text != '':
                                with open(tmp_f, 'w') as f: 
                                    f.write(text)
                                statistic.fname_update(pid, names[1], path)

                        personal_path = f'{dataset_path}/tmp/{pid}/-/personal.json'
                        os.makedirs(f'{dataset_path}/tmp/{pid}/-/', exist_ok=True)
                        if not os.path.isfile(personal_path):
                            with open(personal_path, 'w') as f:  
                                json.dump(personal, f)

                elif not ext: #ssmixなら
                    # 命名規則による抽出 
                    names = os.path.basename(file_name).split("_") #ファイル名を取得して"_"で区切る
                    pid = names[0].zfill(10) #ゼロパディングしてpidとする

                    # skipするデータ
                    if names[-1] != '1': continue #末尾が1じゃない(=旧ファイル)なら飛ばす
                    if names[2] \
                        in ["ADT-01", "ADT-21", "ADT-31", "ADT-32", "ADT-41", "ADT-42", "ADT-51",  #01:担当医の変更・取り消し 21：入院予定、またはその取消 31:外出泊実施、またはその取消 32:外出泊帰院実施、またはその取消 41:転科・転棟(転室・転床)予定、または取消 42:転科・転棟(転室・転床)実施、または取消 51:退院予定、または取消
                                        "OML-01", "OMG-02", "OMG-03"]:
                        continue
                    
                    future = executor.submit(ssmix_contents, pid, names, chunks, statistic)
                    
                    if future.result():
                        dic = future.result()
                        tmp_dir = f'{dataset_path}/tmp/{pid}/{names[1]}/' #names[1]:日付
                        os.makedirs(tmp_dir, exist_ok=True)
                        
                        i = 0
                        while True:
                            path = f'{names[-3]}_{i}_{names[2]}.json' #names[-3]：日付, path例：日付_{i}_ftype(PPR-01など).json
                            tmp_f = tmp_dir + path
                            if os.path.isfile(tmp_f): i += 1 #同じ添え字のファイルが有った場合は添え字を増やす
                            else:                     break
                            
                        with open(tmp_f, 'w') as json_file:
                            json.dump(dic, json_file, indent="\t", ensure_ascii=False) #{dataset_path}/tmp/pid/{日付}以下にjsonファイル形式で出力

                        statistic.fname_update(pid, names[1], path, id=1, type=names[2])
    return                 

def main_pair(inputs):
    zf, statistic = inputs #zf：zipfileのパス、statistic：統計インスタンス
    t0 = datetime.now()
    print(f'{statistic.pair_id}\t{t0}', file=sys.stderr)

    zipped_chunks = get_zipped_chunks(zf)
    get_unzipped_chunks(zipped_chunks, statistic)

    t1 = datetime.now()
    print(f'{statistic.pair_id}\t{t1} for {t1-t0}', file=sys.stderr)

    statistic.output()

    return


def basic_info_update(pid):
    for p in glob.glob(f'{dataset_path}/tmp/{pid}/-/*.json', recursive=True): #患者の基本情報をまとめた「-」ディレクトリ以下のjsonファイルにアクセス
        if os.path.isfile(p):
            fname = os.path.basename(p)
            if fname == 'personal.json':
                with open(p, 'r') as f:
                    personal = json.load(f)
            elif re.fullmatch(r'^.*\_ADT\-00\.json$', fname): #ファイル名の確認
                tmp_f = p
                with open(p, 'r') as f:
                    adt = json.load(f) #読み込み

    if personal and adt: #personalもadtもあれば
        if "PID" in adt: adt['PID'][0].update(personal) #personalに入っているPID-7,PID-8の値でadtを更新
        else: adt.update({"PID": [personal]})
        with open(tmp_f, 'w') as f:
            json.dump(adt, f, indent="\t", ensure_ascii=False) #更新した情報で上書き
        
        zf_path = f'{dataset_path}/{pid}.zip'
        zf_adt = f'/-/{os.path.basename(tmp_f)}'
        if os.path.isfile(zf_path):
            with zipfile.ZipFile(zf_path, "a", compression=zipfile.ZIP_DEFLATED) as nzf:
                nzf.write(tmp_f, arcname=zf_adt) #zf_pathに、zf_adtという名前で、tmp_fを圧縮書き込み


# ----------
# main
# ----------
with Pool(3) as process:
    values = []
    for i in range(3):
        for p in glob.glob(f'{colon_path}/pair_{i}/ka*.zip', recursive=True): # 渡邊さんの頃に利用していたデータのパスに対応しているため要変更
            if os.path.isfile(p):
                nm = p.split('/')
                s = statistic_Dict(f'{nm[-2]}-{nm[-1]}') # ファイル名の一部を利用して統計インスタンスを作成(要変更)
                values.append((p, s)) # ファイルパスと統計インスタンスをタプルでリストに追加
    process.map(main_pair, values) # main_pairに与えて実行

with Pool(3) as process: # 上と同様の処理を別のzipファイルに実行
    values = []
    for i in range(3):
        for p in glob.glob(f'{colon_path}/pair_{i}/an*.zip', recursive=True): # 上と同様に渡邊さんの頃と同じ形式になっているため要変更
            if os.path.isfile(p):
                nm = p.split('/')
                s = statistic_Dict(f'{nm[-2]}-{nm[-1]}') # ファイル名の一部を利用して統計インスタンスを作成(要変更)
                values.append((p, s)) # main_pairに与えて実行
    process.map(main_pair, values)


# 以下についても渡邊さんの頃のファイル構成に倣った処理となっているため、要変更
# データの分類処理(大腸がん患者、大腸がん疑い、大腸がん患者でない)(hl7かxmlのどちらかだけを持つ患者 or 両方を持つ患者 という点でも分類)
for i in range(3):
        with timer(f"pair{i} basic info"):
            with ThreadPoolExecutor(max_workers=1000, thread_name_prefix="thread") as executor:
                # 統計情報をまとめたファイルの読み込み
                pair = json.load(open(f'{dataset_path}/pair_{i}.json', 'r'))
                pid_list = pair['pair']['pidList'] #患者リストを作成

                for pid in tqdm(pid_list):
                    executor.submit(basic_info_update, pid)

        patient_list = json.load(open(f'{dataset_path}/colon.json', 'r'))

        pair_i = json.load(open(f'{dataset_path}/pair_{i}.json', 'r'))
        pair_pid_list = pair_i['pair']['pidList']
        all_pid_list = sorted(pair_pid_list + pair_i['xml']['xml_only'] + pair_i['ssmix']['ssmix_only'])
        
        # ディレクトリ作成
        os.makedirs(f'{dataset_path}/not_patient/', exist_ok=True) #大腸がん患者ではない患者のデータ
        os.makedirs(f'{dataset_path}/suspicion/', exist_ok=True) #大腸がん患者だが疑いの可能性がある
        os.makedirs(f'{dataset_path}/dataset/', exist_ok=True) #大腸がん患者(pairでないものも入れる(2系だけ or 3系だけ の患者も))
        os.makedirs(f'{dataset_path}/dataset_pair/', exist_ok=True) #上のファイルの中で、特にpairのものを入れる(2系と3系どちらもある患者のみ)

        # 患者を分類➡対応するディレクトリに配置
        for pid in all_pid_list:
            src = f'{dataset_path}/{pid}.zip' #その患者のzipファイルに注目
            if os.path.isfile(src):
                if pid in patient_list: #その患者が大腸がん患者のリストに入っていれば
                    suspicion = 1 #疑いではないか確認
                    for info_dict in patient_list[pid]:
                        suspicion *= info_dict['suspicion'] #患者のsuspicionと乗算する
                    if suspicion == 1: # 大腸癌疑いなら=suspicionが1なら
                        shutil.move(src,f'{dataset_path}/suspicion/')
                    else: #疑いでなかったら、
                        if pid in pair_pid_list: #pairかどうかを確認、pairならコピー後にdatasetに移動
                            shutil.copy2(src,f'{dataset_path}/dataset_pair/')
                            shutil.move(src,f'{dataset_path}/dataset/')
                        else: #pairじゃなくてもdatasetに移動
                            shutil.move(src,f'{dataset_path}/dataset/')
                        
                else: #大腸がん患者のリストに入っていなければ
                    shutil.move(src,f'{dataset_path}/not_patient/')
