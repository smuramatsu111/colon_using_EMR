import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import csv
import sys
import json
import glob
import torch
import pprint
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from mylib.nn_model_regre import MyDataset_CA, MLregression, NeuralNetwork


#プログラム実行時に入力を受取り、それに応じて入力データ(の規模や種類)を変更する
#予測対象である腫瘍マーカー値の種類を選択し(CEAかCA19-9)、同時に入力データのサイズを選択する(S、M、Lの3種類)
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, default='fullL', help='data')
args = parser.parse_args()

#利用するtextデータの種類を指定
#textデータは、カルテの記述における代表的な記法であるSOAPの枠組みを考慮して抽出されている(S:Subjective, O:Objective, A:Assessment, P:Plan)(カルテ中で、それぞれに対応した記述があり、データ抽出段階でそれらを区別して抽出した)
#本研究では上述のSOAPに当てはまらない、患者の基本情報(既往歴・生活歴・家族歴)等を抽出したOtherデータのみを用いているため以下ではotherみを指定
text_label = ['other']

features = 'MS'

# training
# NHO現地で調整している、ここでの値は適当な初期値
num_workers = 5   # data loader num workers
train_epochs = 50 # train epochs
batch_size =   64 # batch size of train input data
learning_rate = 1e-3  # optimizer learning rate
dropout = 0.3

#隠れ層のサイズを指定
#入力条件として1. 数値データのみ と 2. 数値データ + textデータ の2種類を用意している
#数値データのみ(num)の場合は、入力次元数が少なくなるため、隠れ層のサイズを小さくしている
#層のサイズも現地で適宜調整している
hidden1_num = 1024

if 'num' in args.data:
    hidden2_num = 256
else:
    #hidden2_num = 256
    hidden2_num = 512

hidden3_num = 64

#実行時の入力に応じて、ターゲットとなる腫瘍マーカー値の種類を選択・入力次元数とラベル数を指定
data_parser = {
    'CEA_fullS':   {'T': ['CEA'], 'M':[882,882], 'MS':[882,1]}, # text: 768 + num): 114
    'CEA_fullM':   {'T': ['CEA'], 'M':[902,902], 'MS':[902,1]}, # text: 768 + num: 134
    'CEA_fullL':   {'T': ['CEA'], 'M':[918,918], 'MS':[918,1]}, # text: 768 + num: 150
    'CEA_numS':    {'T': ['CEA'], 'M':[114,114],   'MS':[114,1]},   # num: 114
    'CEA_numM':    {'T': ['CEA'], 'M':[134,134], 'MS':[134,1]},   # num: 134
    'CEA_numL':    {'T': ['CEA'], 'M':[150,150], 'MS':[150,1]},   # num: 150
    'CA199_fullS': {'T': ['CA19-9'], 'M':[882,882], 'MS':[882,1]}, # text: 768 + num: 114
    'CA199_fullM': {'T': ['CA19-9'], 'M':[902,902], 'MS':[902,1]}, # text: 768 + num: 134
    'CA199_fullL': {'T': ['CA19-9'], 'M':[918,918], 'MS':[918,1]}, # text: 768 + num: 150
    'CA199_numS':  {'T': ['CA19-9'], 'M':[114,114],   'MS':[114,1]},   # num: 114
    'CA199_numM':  {'T': ['CA19-9'], 'M':[134,134], 'MS':[134,1]},   # num: 134
    'CA199_numL':  {'T': ['CA19-9'], 'M':[150,150], 'MS':[150,1]},   # num: 150
}


# -------------
#  func
# -------------
print("func")

#各特徴量は、それが補間された値であるか否かを表すフラグ情報を持っている。このフラグ情報を削除する際に、特徴量のカラム名に「_f」を付与する関数
def add_f(text):
    return f'{text}_f'
    
#pklファイルをdfに変換する関数
def f2df(fname):
    fnames = re.split('[\.\_]', os.path.basename(fname))
    ext = fnames[-1]
    if ext == 'pkl':
        df = pd.read_pickle(fname)
    else:
        print(f"*********** File Error ***************\n{fname}", file=sys.stderr) #対応ファイルで無ければエラー表示して何もしない
        return None
    
    return df


# --------
#  ready
# --------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#入力に応じて予測対象とする腫瘍マーカーに対応するデータセットpathを獲得
if 'CEA' in args.data:                              
    #CEA専門のpklファイルを指定
    data_path = 'NHO現地でpathを指定'
elif 'CA199' in args.data:
    #CA19-9専門のpklファイルを指定
    data_path = 'NHO現地でpathを指定'

if args.data in data_parser.keys(): #コマンド実行時の入力を確認し、
    data_info = data_parser[args.data] #対応する辞書をdata_infoに格納
    target = data_info['T'] #予測対象(Target)情報の獲得
    input_dim, label_num = data_info[features] #入力次元数とラベル数の獲得

#入力データのサイズとしてS,M,Lの3種類を設定、コマンド実行時に指定する
#入力されたサイズ情報に対応して、特徴量の削減処理を行う
del_label = [] #削除する特徴量のカラム名を格納するリスト
if 'M' in args.data or 'S' in args.data: #SかMだった時の処理(特徴量削減)
    with open('/~特徴量情報をまとめたcsvファイルまでのpath~/Label.csv', 'r') as csv_file:
        csv_reader = list(csv.reader(csv_file))
        if 'M' in args.data: del_label += csv_reader[8] #del_labelに['Eosin', 'Baso', 'MPV', 'AMY', 'TG', 'Na', 'K', 'Cl']を追記
        if 'S' in args.data: del_label += csv_reader[7] + csv_reader[8] #上の要素 + ['Neutro', 'St', 'Seg', 'HCT', 'MCV', 'MCH', 'MCHC', 'BUN', 'IP', 'γ-GTP']を追記
        del2 = map(add_f, del_label) 
        del_label += list(del2) #fを付けたものをdel_labelに追記(例：['Eosin_f', 'Baso_f', 'MPV_f', 'AMY_f', 'TG_f', 'Na_f', 'K_f', 'Cl_f'])
#数値のみを入力とする場合、textデータを削除する
if 'num' in args.data:
    for column in text_label: #text_label = [other]
        del_label += [f'{column}{i}' for i in range(768)] #textデータは768次元のベクトルとして表現されているため、それら全てを削除する


# ----------
#   main
# ----------

print("main") #デバッグ

model = NeuralNetwork(
    input_dim=input_dim,
    hidden_size1=hidden1_num,
    hidden_size2=hidden2_num,
    label_num=label_num,
    device=device
    )

model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate) #最適化手法：adam

df = f2df(data_path)
train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=0)
valid_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=0)


print("trainloader作成開始")

#以下で訓練データの読み込み及びバッチ化を行う
if not train_df is None:
    train_set = MyDataset_CA(train_df, target, del_label)
    batch = batch_size #64
    size = 0
    while size == 0:
        train_loader = DataLoader(
            train_set,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        size = len(train_loader)
        
    print("train_loader_size",size)
print("trainLoade作成r終了")

print("validloader作成開始")
if not valid_df is None:
    valid_set = MyDataset_CA(valid_df, target, del_label)
    batch = batch_size #64
    size = 0
    while size == 0:
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        size = len(valid_loader)
        
    print("valid_loader_size",size)
print("validLoader作成終了")

print("testloader作成開始")
if not test_df is None:
    test_set = MyDataset_CA(test_df, target, del_label)
    batch = batch_size
    size = 0
    while size == 0:
        test_loader = DataLoader(
            test_set,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        size = len(test_loader)
    print("train_loader_size",size)
print("testLoader作成終了")


ml = MLregression(
    training_loader=train_loader,
    validating_loader=valid_loader,
    testing_loader=test_loader,
    model=model,
    device=device,
    optimizer=optimizer,
    target=target)

print("学習開始")
print("Train:", len(train_loader), "Test:", len(test_loader))
for epoch in range(train_epochs):
    ml.train(epoch, train_epochs)
    ml.validation(epoch, train_epochs)

    path = f"./NN_log/regre/epoch/epoch{epoch}.pth"
    os.makedirs(f"./NN_log/regre/epoch/", exist_ok=True)
    ml.save(path)

ml.test()





