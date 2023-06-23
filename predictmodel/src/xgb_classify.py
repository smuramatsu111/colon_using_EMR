#患者の現在の(検査情報、薬剤投与情報を基にした)状態から、未来の患者の状態を分類するプログラム
#具体的には患者の腫瘍マーカー値の変動に対して閾値を定め、上昇、下降、横ばいに分類し、この3クラス分類を行う
import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import csv
import sys
import json
import glob
import pprint
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


#プログラム実行時に入力を受取り、それに応じて入力データ(の規模や種類)を変更する
#具体的には、予測対象である腫瘍マーカー値の種類を選択し(CEAかCA19-9)、同時に入力データの次元数を選択する(S、M、Lの3種類)
#別の研究で利用した仕組みであり、本研究では利用する必要がないが、時間がなかったため仕組みが中途半端に残っている
#本研究では入力条件の規模として常にLを指定する、腫瘍マーカー値の種類は実行時に指定する
#今後修正予定
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, default='fullL', help='data') #-dataの-はオプション引数であることを示している、required=trueで必須指定にしている、defaultはデフォルト値の指定、helpは引数の説明
args = parser.parse_args()

data_parser = {
    'CEA_S':    {'T': ['CEA']},
    'CEA_M':    {'T': ['CEA']},
    'CEA_L':    {'T': ['CEA']},
    'CA199_S':  {'T': ['CA19-9']},
    'CA199_M':  {'T': ['CA19-9']},
    'CA199_L':  {'T': ['CA19-9']},
}

# -------------
#  func
# -------------
#特徴量のカラム名に「_f」を付与するメソッド(フラグ情報の削除を行う際に利用)
def add_f(text):
    return f'{text}_f'

#データを受け取って、特徴量と予測対象のデータに分けるメソッド
def make_feature_target(dataframe, target_label):
    targets = dataframe[target_label].copy()
    inputs = dataframe.drop(target_label,axis=1)
    return inputs, targets

# --------
#  ready
# --------
#実行時の入力に応じてデータの種類を選択
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    target_markaer = data_info['T']

#データパスの指定
#本研究ではCEAとCA19-9の2種類の腫瘍マーカー値を予測対象としており、それぞれについて別で実行するような実行方法を取っているためこのように記述している
if 'CEA' in args.data:                              
    #CEAを予測対象とするpklファイルを指定、実データがある現地での実行時にはこの部分を変更する必要がある
    data_path = 'all_cea_data_path(仮)'
elif 'CA199' in args.data:
    #CA19-9を予測対象とするpklファイルを指定、実データがある現地での実行時にはこの部分を変更する必要がある
    data_path = 'all_CA19-9_data_path(仮)'

#学習に利用しないカラムの削除
del_label = [] #削除するカラムのカラム名を格納するリスト
flag_source =[] #flag:検査項目の値が補間された値であるか否かを表すフラグ情報、本プログラムで特徴量として利用しないため削除する、flag_sourceに格納したカラム名に対して先述のadd_fメソッドを適用することでカラム名に対応したフラグ情報のリストを取得する。
#del_labelに削除するカラム名を格納していく処理
with open('/home/smuramatsu/EMR/MyProject/ssmixparser/mylib/Label.csv', 'r') as csv_file: #特徴量をまとめたcsvファイルの読み込み
    csv_reader = list(csv.reader(csv_file))
    #全てのフラグ情報を消すための処理、今後修正予定
    flag_source += csv_reader[2] + csv_reader[3] + csv_reader[5] + csv_reader[6] + csv_reader[7] + csv_reader[8] + csv_reader[9]
    del = map(add_f, flag_source)
    del_label += list(del)

#テキストのembedding情報を以前の研究で利用したが、本研究では利用しないため除去する
del_label += [f'other{i}' for i in range(768)] #768次元のテキストembedding情報を削除するためにdel_labelに追加

# ----------
#   main
# ----------
#データの読み込み
df  = pd.read_pickle(data_path)

#日付カラムを削除するため、del_labelに追加
del_label += ['date']

#予測対象とする腫瘍マーカー値の未来の値を削除()
future_marker = [f'future_{target_markaer[0]}']
del_label += future_marker

#del_labelにまとめたカラムのデータを削除
df = df.drop(del_label, axis=1)

#分類問題の予測対象とするカラム名を変数に格納
target_class = target_markaer[0] + '_class'
print(f"target:{target_class}")

#特徴量一覧を出力
featuers = [f for f in df.columns if f != target_class]
print(f"featuers:{featuers}")

#stratifyで、target_classの割合がtrainとvalidで同じになる様に分割
train_df, valid_test_df = train_test_split(df, test_size=0.2, random_state=0, stratify = df[target_class])
#valid_dfをtestとvalidに分ける
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5, random_state=0, stratify=valid_test_df[target_class])

#train, valid, testを特徴量(feature)と予測対象(target)に分ける
x_train, y_train = make_feature_target(train_df, target_class)
x_valid, y_valid = make_feature_target(valid_df, target_class)
x_test, y_test = make_feature_target(test_df, target_class)

#データのサイズを出力
print(f"Train:{len(x_train)} Vallid:{len(x_valid)} Test:{len(x_test)}")

#XGBoostのデータ型に変換
dtrain = xgb.DMatrix(x_train, y_train)
dvalid = xgb.DMatrix(x_valid, y_valid)
dtest = xgb.DMatrix(x_test, y_test)

#多クラス分類、クラス数は3
params = {
    "objective" : "multi:softprob",
    "num_class" : 3,
}

#学習過程を保存する辞書
results_dict = {}

#モデルの定義
model = xgb.train(
    params = params,
    dtrain = dtrain,
    evals = [(dtrain, "train"), (dvalid, "valid")],
    num_boost_round = 100,
    early_stopping_rounds = 10,
    evals_result = results_dict
)

#評価データで予測精度確認
pred = model.predict(dtest).argmax(axis = 1) #predictで0,1,2それぞれに対する確率が出力されるためargmaxで最大値のインデックス取得

#予測結果出力
#print("pred:",pred[:5])
#print("true:",y_valid[:5])

#事例分析のため、正解と不正解の実データを出力
"""
for i ,(pred, true) in enumerate(zip(pred, y_valid)):
    if pred == true:
        print(f"pred:{pred} ture:{true} 正解！")
        print(x_valid.iloc[i])

    else:
        print(f"pred:{pred} ture:{true} 不正解！")
        print(x_valid.iloc[i])
"""

#スコア算出
print("classification_report")
print(classification_report(y_test, pred))


#ログに追記するメモ(入力に利用したデータ条件に応じてメモの内容を変更)
if "NotAllZero" in data_path:
    memo = "NotAllZero"
elif "UnderDiffRemove" in data_path:
    memo = "RemoveError"
else:
    memo = "All"

#ロスの推移確認
plt.figure()
plt.plot(results_dict["train"]["mlogloss"], color = "red", label = "train")
plt.plot(results_dict["valid"]["mlogloss"], color = "blue", label = "valid")
plt.legend()
plt.savefig(f"./output/xgb/class/graph/{target_markaer[0]}/loss_move_{args.data}_{memo}.png")

#予測に貢献する特徴量確認
plt.figure()
title_info = f"{target_markaer[0]}_{memo}"
xgb.plot_importance(model, title = title_info, xlabel = 'gain', ylabel = 'feature', importance_type = "gain", max_num_features=20)
plt.savefig(f"./output/xgb/class/graph/{target_markaer[0]}/importance_{args.data}_{memo}.png")
