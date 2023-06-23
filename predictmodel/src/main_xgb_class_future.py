#git_test用のコメント
import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #GPUの指定を応答順➡PCI BUSのIDに紐づけて指定できるように変更
os.environ["CUDA_VISIBLE_DEVICES"]="0" #GPU指定
import argparse
import csv
import sys
import json
import glob
import pprint
import pandas as pd
#from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None) #データフレームの表示設定ヲ行う、全てのデータを表示
pd.set_option('display.max_columns', None)

#future-now を予測するプログラム、データ指定と削除するカラムが異なるだけ

parser = argparse.ArgumentParser() #プログラム実行時に引数を与える
parser.add_argument('--data', type=str, required=True, default='fullL', help='data') #-dataの-はオプション引数であることを示している、required=trueで必須指定にしている、defaultはデフォルト値の指定、helpは引数の説明
args = parser.parse_args() #引数を解析

text_label = ['other']

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

def add_f(text):
    return f'{text}_f'

def make_feature_target(dataframe, target_label):
    targets = dataframe[target_label].copy()
    inputs = dataframe.drop(target_label,axis=1)

    return inputs, targets

# --------
#  ready
# --------

if args.data in data_parser.keys(): #コマンド実行時のオプションに一致するkeyを確認し、
    data_info = data_parser[args.data] #それに対応するvalueの値をdata_infoに格納
    target_markaer = data_info['T']

#データパスの指定
if 'CEA' in args.data:                              
    #CEA専門のpklファイルを指定
    #全データ
    data_path = '/home/smuramatsu/EMR/MyProject/data/Training_data/dataset/pkl_dataset/unite/class/class_xgb_future_now_All_CEA_14581.pkl'
    #全て0の患者を除去したデータ
    #data_path = '/home/smuramatsu/EMR/MyProject/data/Training_data/dataset/pkl_dataset/unite/class/class_xgb_future_now_NotAllZero_CEA_8515.pkl'
    #差が誤差以下のものを除去したデータ
    #data_path = '/home/smuramatsu/EMR/MyProject/data/Training_data/dataset/pkl_dataset/unite/class/class_xgb_future_now_UnderDiffRemove_CEA_921.pkl'
elif 'CA199' in args.data:
    #CA19-9専門のpklファイルを指定
    #全データ
    #data_path = '/home/smuramatsu/EMR/MyProject/data/Training_data/dataset/pkl_dataset/unite/class/class_xgb_future_now_All_CA19-9_13370.pkl'
    #全て0の患者を除去したデータ
    #data_path = '/home/smuramatsu/EMR/MyProject/data/Training_data/dataset/pkl_dataset/unite/class/class_xgb_future_now_NotAllZero_CA19-9_4700.pkl'
    #差が誤差以下のものを除去したデータ
    data_path = '/home/smuramatsu/EMR/MyProject/data/Training_data/dataset/pkl_dataset/unite/class/class_xgb_future_now_UnderDiffRemove_CA19-9_991.pkl'

del_label = []
flag_source =[]
with open('/home/smuramatsu/EMR/MyProject/ssmixparser/mylib/Label.csv', 'r') as csv_file:
    csv_reader = list(csv.reader(csv_file))
    if 'S' in args.data or 'M' in args.data: #真ん中か一番下だった時の処理
        if 'M' in args.data: del_label += csv_reader[8] #del_labelに['Eosin', 'Baso', 'MPV', 'AMY', 'TG', 'Na', 'K', 'Cl']を追記
        if 'S' in args.data: del_label += csv_reader[7] + csv_reader[8] #上の要素 + ['Neutro', 'St', 'Seg', 'HCT', 'MCV', 'MCH', 'MCHC', 'BUN', 'IP', 'γ-GTP']を追記
        #del2 = map(add_f, del_label) #元のM,Sに対応して消すコード
        #del_label += list(del2) #fを付けたものをdel_labelに追記(例：['Eosin_f', 'Baso_f', 'MPV_f', 'AMY_f', 'TG_f', 'Na_f', 'K_f', 'Cl_f']) #元のM,Sに対応して消すコード
    
    #M,Sなどに関係なく、全てのfを消すための処理を一次的に記述(今後データセット構築の段階で除去したものを用意)
    #fを消す理由：xgbosstは、欠損値を許容して学習できる➡フラグの情報は逆に邪魔
    flag_source += csv_reader[2] + csv_reader[3] + csv_reader[5] + csv_reader[6] + csv_reader[7] + csv_reader[8] + csv_reader[9] #腫瘍マーカーのフラグも除去
    del2 = map(add_f, flag_source)
    del_label += list(del2)


        

#テキスト部は要らないから常に除去
#for column in text_label: #otherのこと
#    del_label += [f'{column}{i}' for i in range(768)] #S0, S1, ~ ,S767, O0,...というようにそれぞれに0~767の添え字を付けたものをdel_labelに追記
del_label += [f'other{i}' for i in range(768)] #S0, S1, ~ ,S767, O0,...というようにそれぞれに0~767の添え字を付けたものをdel_labelに追記

# ----------
#   main
# ----------

df  = pd.read_pickle(data_path)

del_label += ['date']

#(最新の)腫瘍マーカー値は使うことが出来ないため削除➡今回は予測対象が未来のため、除去しない(フラグは上で消している)
#marker_list = ['CEA', 'CA19-9']
#marker_f_list = map(add_f, marker_list)
#del_label += marker_list + list(marker_f_list)

#未来の腫瘍マーカー値は使うことが出来ないため削除
future_marker_list = [f'future_{target_markaer[0]}']
del_label += future_marker_list

#del_labelに記されているlabelのデータを除去
df = df.drop(del_label, axis=1)

#分類問題の予測対象となるカラム名を変数に格納
target_class = target_markaer[0] + '_class'
print(f"target:{target_class}")

#特徴量確認用
featuers = [f for f in df.columns if f != target_class]
print(f"featuers:{featuers}")

#stratifyで、target_classの割合がtrainとvalidで同じになる様に分割
train_df, valid_test_df = train_test_split(df, test_size=0.2, random_state=0, stratify = df[target_class])
#valid_dfをtestとvalidに分けてみる
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5, random_state=0, stratify=valid_test_df[target_class])

x_train, y_train = make_feature_target(train_df, target_class)
x_valid, y_valid = make_feature_target(valid_df, target_class)
x_test, y_test = make_feature_target(test_df, target_class)

print(f"Train:{len(x_train)} Vallid:{len(x_valid)} Test:{len(x_test)}")

#nanを置換する処理
#xgboostはnanを対処するアルゴリズムが組み込まれているから無くても良い
#x_train = x_train.fillna(0)
#x_valid = x_valid.fillna(0)

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
#pred = model.predict(xgb.DMatrix(x_valid)) #下の書き方とこちらのどっちが正しいか謎
#pred = model.predict(dvalid).argmax(axis = 1) #predictだけだと、0,1,2それぞれに対する確率が出力されるためargmaxで最大値のインデックス取得
pred = model.predict(dtest).argmax(axis = 1)

#予測結果出力
#print("pred:",pred[:5])
#print("true:",y_valid[:5])

#正解と不正解の実データ出力
"""
for i ,(pred, true) in enumerate(zip(pred, y_valid)):
    if pred == true:
        print(f"pred:{pred} ture:{true} 正解！")
        print(x_valid.iloc[i])

    else:
        print(f"pred:{pred} ture:{true} 不正解！")
        print(x_valid.iloc[i])
"""
#手動でmacro_accuracy求める:accuracyにはmacroもmicroもない！
"""
up = 0
up_correct = 0
keep = 0
keep_correct = 0
down = 0
down_correct = 0

for tes, pre in zip(y_test, pred):
    if tes == 0:
        down += 1
        if tes == pre:
            down_correct += 1
    if tes == 1:
        keep += 1
        if tes == pre:
            keep_correct += 1
    elif tes == 2:
        up += 1
        if tes == pre:
            up_correct += 1

down_acc = down_correct/down
keep_acc = keep_correct/keep
up_acc = up_correct/up

print("手動でacc計算")
print(f"up_acc:{up_acc} keep_acc:{keep_acc} down_acc:{down_acc}")
print("macro_acc")
print((up_acc + keep_acc + down_acc)/3)
"""



#正解率算出
#print(f"acc:{accuracy_score(y_valid, pred)}")
print(f"acc:{accuracy_score(y_test, pred)}")
print("classification_report")
print(classification_report(y_test, pred))

#ベースライン算出
print("about_baseline")
y_base = y_test.values
#modeについて
print("base_mode")
mode = y_test.mode().tolist()[0]
mode_list = np.full_like(y_base, mode)
print(f"acc:{accuracy_score(y_base, mode_list)}")
print("classification_report")
print(classification_report(y_base, mode_list))

#rollについて
print("base_roll")
y_roll = np.roll(y_base, -1)
print(f"acc:{accuracy_score(y_base, y_roll)}")
print("classification_report")
print(classification_report(y_base,  y_roll))



#ログに追記するメモ
if "NotAllZero" in data_path:
    memo = "NotAllZero"
elif "UnderDiffRemove" in data_path:
    memo = "RemoveError"
else:
    memo = "All"


#ロスの推移確認
"""
plt.figure()
plt.plot(results_dict["train"]["mlogloss"], color = "red", label = "train")
plt.plot(results_dict["valid"]["mlogloss"], color = "blue", label = "valid")
plt.legend()
plt.savefig(f"./output/xgb/class/future_now/graph/{target_markaer[0]}/loss_move_{args.data}_{memo}_test.png")

#予測に貢献する特徴量確認
plt.figure()
title_info = f"{target_markaer[0]}_{memo}"
xgb.plot_importance(model, title = title_info, xlabel = 'gain', ylabel = 'feature', importance_type = "gain", max_num_features=20)
plt.savefig(f"./output/xgb/class/future_now/graph/{target_markaer[0]}/importance_{args.data}_{memo}_test.png")
"""