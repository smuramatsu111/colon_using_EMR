#LSTMによるregre(回帰)用のプログラム
#評価指標：MSE
#loss：MSE

import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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

from mylib.myLSTM_regre import SequenceDataset, LSTMClass, MLregression

parser = argparse.ArgumentParser() #プログラム実行時に引数を与える→引数に応じて実験条件を変更
parser.add_argument('--data', type=str, required=True, default='fullL', help='data')
args = parser.parse_args() #引数解析

# パス、次元数などを指定
text_label = ['S', 'O', 'A', 'P', 'other']
data_path = 'データセットまでのパス'
features = 'MS'

checkpoints = './LSTM/single/'

seq_len   = 21  # input sequence length
pred_len  = 1   # prediction sequence length

# 引数に応じて次元数を変更
if 'num' in args.data:
    d_model = 64
elif 'small' in args.data:
    d_model = 512
elif 'min' in args.data:
    d_model = 256
else:
    d_model = 256

dropout = 0.3

# training
num_workers = 5   # data loader num workers
train_epochs = 10 # train epochs
batch_size = 64   # batch size of train input data
learning_rate = 1e-5  # optimizer learning rate

use_gpu = True  # use gpu
gpu = 2  # gpu
use_multi_gpu = False
devices = '0,1,2,3'  # device ids of multile gpus


data_parser = {
    'fullH':  {'T': ['CEA','CA19-9'], 'M':[3929,3929], 'MS':[3929,2]}, # text: 768*5 + num: 89
    'fullM':  {'T': ['CEA','CA19-9'], 'M':[3949,3949], 'MS':[3949,2]}, # text: 768*5 + num: 109
    'fullL':  {'T': ['CEA','CA19-9'], 'M':[3965,3965], 'MS':[3965,2]}, # text: 768*5 + num: 125
    'smallH': {'T': ['CEA','CA19-9'], 'M':[601,601],    'MS':[601,2]},   # text: 512 + num: 89
    'smallM': {'T': ['CEA','CA19-9'], 'M':[621,621],    'MS':[621,2]},   # text: 512 + num: 109
    'smallL': {'T': ['CEA','CA19-9'], 'M':[637,637],    'MS':[637,2]},   # text: 512 + num: 125
    'minH':   {'T': ['CEA','CA19-9'], 'M':[345,345],    'MS':[345,2]},   # text: 256 + num: 89
    'minM':   {'T': ['CEA','CA19-9'], 'M':[365,365],    'MS':[365,2]},   # text: 256 + num: 109
    'minL':   {'T': ['CEA','CA19-9'], 'M':[381,381],    'MS':[381,2]},   # text: 256 + num: 125
    'numH':   {'T': ['CEA','CA19-9'], 'M':[89,89],       'MS':[89,2]},   # num: 89
    'numM':   {'T': ['CEA','CA19-9'], 'M':[109,109],    'MS':[109,2]},   # num: 109
    'numL':   {'T': ['CEA','CA19-9'], 'M':[125,125],    'MS':[125,2]},   # num: 125
}

# -------------
#  func
# -------------
def add_f(text):
    return f'{text}_f'
    
def f2df(fname):
    fnames = re.split('[\.\_]', os.path.basename(fname))
    ext = fnames[-1]
    if ext == 'pkl':
        df = pd.read_pickle(fname)
    elif ext == 'csv':
        df = pd.read_csv(fname, index_col=0)
    else:
        print(f"*********** File Error ***************\n{fname}", file=sys.stderr)
        return None

    # 実験で用いない特徴量の除去
    if 'Mg' in df.columns.values: #Mgがあればその列を削除
        df = df.drop(['Mg', 'Mg_f'], axis=1)
    
    if '蛋白定量' in df.columns.values: #蚕白定量があればその列を削除
        df = df.drop(['蛋白定量', '蛋白定量_f'], axis=1) 

    if 'TTR' in df.columns.values: #TTRについても同様に削除
        df = df.drop(['TTR', 'TTR_f'], axis=1)

    if len(df) < seq_len+pred_len:
        return None
    
    return df


# --------
#  ready
# --------
use_gpu = True if torch.cuda.is_available() and use_gpu else False

if use_gpu and use_multi_gpu:
    devices = devices.replace(' ','')
    device_ids = devices.split(',')
    device_ids = [int(id_) for id_ in device_ids]
    gpu = device_ids[0]

if use_gpu: device = torch.device('cuda:{}'.format(gpu))

# 実行時の引数から実験条件を獲得
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    target = data_info['T'] 
    input_dim, label_num = data_info[features] #input_dimは特徴量の次元数、label_numはラベルの数(CEA, CA19-9という2つの腫瘍マーカー値を同時に予測するため2)


if 'small' in args.data: data_path += '/512'
elif 'min' in args.data: data_path += '/256' 

del_label = []
if 'M' in args.data or 'H' in args.data: #実験で用いる特徴量の規模がM or H(中規模か小規模)だった場合,除去する特徴量のラベル情報をまとめる処理
    with open('ラベル情報をまとめたcsvファイルまでのパス', 'r') as csv_file:
        csv_reader = list(csv.reader(csv_file))
        if 'M' in args.data: del_label += csv_reader[8] #del_labelに['Eosin', 'Baso', 'MPV', 'AMY', 'TG', 'Na', 'K', 'Cl']を追記
        if 'H' in args.data: del_label += csv_reader[7] + csv_reader[8] #上の要素 + ['Neutro', 'St', 'Seg', 'HCT', 'MCV', 'MCH', 'MCHC', 'BUN', 'IP', 'γ-GTP']を追記
        del2 = map(add_f, del_label) 
        del_label += list(del2) #fを付けたものをdel_labelに追記(例：['Eosin_f', 'Baso_f', 'MPV_f', 'AMY_f', 'TG_f', 'Na_f', 'K_f', 'Cl_f'])

# 実験条件がnum(数値のみ)の場合、テキストデータを除去
if 'num' in args.data:
    for column in text_label: #S,O,A,P,otherのこと
        del_label += [f'{column}{i}' for i in range(768)]



# ----------
#   main
# ----------
file_list = [f for f in glob.glob(f'{data_path}/*.pkl') if os.path.isfile(f)]
file_list.sort()
sample_num = 10000 #扱うファイル数を指定
file_list = file_list[:sample_num]
test_num = int(len(file_list) * 0.1)
train_list = file_list[:-test_num] #file_listのうち、9割をtrain＆valid
test_list  = file_list[-test_num:] #file_listのうち、1割をtest

test_loaders = []

for ftest in test_list: 
    df = f2df(ftest) #f2dfでtest_list内のファイルをdf化＆学習に適さないデータの削除
    if df is None: continue
    test_set = SequenceDataset(df, target, del_label, seq_len, pred_len) #入力形式に変換
    batch = batch_size #64
    size = 0
    while size == 0:
        test_loader = DataLoader(
            test_set,
            batch_size=batch,
            shuffle=True,
            num_workers=num_workers, #ディープラーニングを高速化
            drop_last=True #余りのデータが出来たときに、それを除去する(しないとそのデータだけ重みが強くなる)
        )
        
        size = len(test_loader)
        batch -= 1

    test_loaders.append(test_loader)

model = LSTMClass(
    input_dim=input_dim,
    hidden_size=d_model,
    dropout=dropout,
    list_num=label_num
    )

if use_multi_gpu and use_gpu:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
else:
    model.to(device)


optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate) #最適化手法にadamを選択

# 5-クロスバリデーション
fold5 = list(np.array_split(train_list, 5))

for fold in range(5):
    train_list = sum([fold5[idx].tolist() for idx in range(5) if idx != fold], [])
    valid_list = fold5[fold].tolist()
    train_steps = len(train_list)

    train_loaders, valid_loaders = [], []

    #訓練データの読み込み及びバッチ化
    for ftrain in train_list:
        df = f2df(ftrain)
        if df is None: continue
        train_set = SequenceDataset(df, target, del_label, seq_len, pred_len)
        batch = batch_size
        size = 0
        while size == 0:
            train_loader = DataLoader(
                train_set,
                batch_size=batch,
                shuffle=batch_size,
                num_workers=num_workers,
                drop_last=True
            )
            size = len(train_loader)
            batch -= 1
        train_loaders.append(train_loader)

    #評価データの読み込み及びバッチ化
    for fvalid in valid_list:
        df = f2df(fvalid)
        if df is None: continue
        valid_set = SequenceDataset(df, target, del_label, seq_len, pred_len)
        batch = batch_size
        size = 0
        while size == 0:
            valid_loader = DataLoader(
                valid_set,
                batch_size=batch,
                shuffle=batch_size,
                num_workers=num_workers,
                drop_last=True
            )
            size = len(valid_loader)
            batch -= 1
        valid_loaders.append(valid_loader)

ml = MLregression(
    training_loader=train_loaders,
    validate_loader=valid_loaders,
    testing_loader=test_loaders,
    model=model,
    device=device,
    optimizer=optimizer,
    target=target,
    test_term=pred_len)


print("Train:", len(train_loaders), "Valid:", len(test_loaders), "Test:", len(test_loaders))
for epoch in range(train_epochs):
    ml.train(epoch, train_epochs)
    ml.validation(epoch, train_epochs)

    path = f"./LSTM_131/epoch/epoch{epoch}.pth"
    os.makedirs(f"./LSTM_131/epoch/", exist_ok=True)
    ml.save(path)

ml.test()

