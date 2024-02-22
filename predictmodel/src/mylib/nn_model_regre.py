import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
from torch import nn
import math
from math import sqrt


from genericpath import exists
import logging
import sys
import numpy as np
import pandas as pd

from decimal import *
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanAbsolutePercentageError

dir_name = "NHOで指定" #logファイルの出力フォルダ名、実験ごとに変更
label = "CA19-9" #CEA or CA19-9を指定 #呼び出し時に自動で指定するように変更予定

log_path = "./output/regre/" + label + "/" + dir_name + "/"
os.makedirs(log_path,exist_ok=True)
writer = SummaryWriter(log_dir = log_path)

pd.set_option('display.max_columns', None)
logging.basicConfig(level=logging.ERROR)

#入力形式にデータを変換
class MyDataset_CA(Dataset):
    def __init__(self, dataframe, target_label, del_label):
        del_label += ['date']
        target_f = target_label[0] + '_f'
        del_label += [target_f]

        data = dataframe.drop(del_label, axis=1) #del_labelに追記されているlabelのデータを除去
        self.targets = data[target_label].copy()

        data = data.drop(target_label,axis=1)
        self.inputs = data

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_df = self.inputs.iloc[index, :]
        target_df = self.targets.iloc[index, :]
        
        input_np = input_df.values
        target_np = target_df.values

        return {
            'inputs': torch.tensor(input_np, dtype=torch.float),
            'targets': torch.tensor(target_np, dtype=torch.float)
        } 

#モデル構造の定義
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, label_num, device):
        super().__init__()
        self.myl = MyLinear_new(input_dim, hidden_size1, device)
        self.l2 = torch.nn.Linear(hidden_size1,hidden_size2)
        self.l3 = torch.nn.Linear(hidden_size2, label_num)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, input):
        output = self.act1(self.myl(input))
        output = self.act2(self.l2(output))
        output = self.l3(output)
        return output

# 学習・検証・テストの実行→結果の出力まで
class MLregression():
    def __init__(self, training_loader, validating_loader, testing_loader, model, device, optimizer, target):
        self.training_loader= training_loader
        self.validating_loader = validating_loader
        self.testing_loader = testing_loader
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_label = target

    def loss_fn(self, outputs, targets):
        return torch.nn.MSELoss()(outputs, targets)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def metrics_mse(self, pred, true):
        return np.mean((true-pred)**2)

    def metrics_mae(self, pred, true):
        mae = np.mean(np.abs(pred - true))
        return mae

    def metrics_mape(self, pred, true):
        pred = pred + 1
        true = true + 1
        mape = np.mean(np.abs((pred - true)/(true)))*100 #1を誤差だとみなし、加えたもの
        #mape = np.mean(np.abs((pred - true)/(true + 1e-1)))*100 #限りなく小さい値の0.1を加えたもの
        return mape 

    def train(self, epoch, EPOCHS):
        self.model.train()
        with tqdm(total=len(self.training_loader), unit='batch', position=0, leave=True) as pbar:
            pbar.set_description(
                f"Epoch[{'{0:2d}'.format(epoch + 1)}/{EPOCHS}](train)")
                
            for data in self.training_loader:
                inputs = data['inputs'].to(self.device, dtype=torch.float)
                targets = data['targets'].to(self.device, dtype=torch.float)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
        print(f"Epoch [{epoch+1}] train Loss:{loss.item()}")
        writer.add_scalar("Train/Loss", loss.item(), epoch+1)
    
    writer.close()


    def validation(self, epoch, EPOCHS, *args):
        self.model.eval()
        ids_list = []
        target_list = []
        output_list = []
        total_mse = {}
        total_mae = {}
        total_mape = {}
        with torch.no_grad():
            with tqdm(total=len(self.validating_loader), unit='batch') as pbar:
                pbar.set_description(f"Epoch[{'{0:2d}'.format(epoch + 1)}/{EPOCHS}](valid)")
                for data in self.validating_loader:
                    fin_targets = []
                    fin_outputs = []
                
                    inputs = data['inputs'].to(self.device, dtype=torch.float)
                    targets = data['targets'].to(self.device, dtype=torch.float)
                    outputs = self.model(inputs)

                    loss = self.loss_fn(outputs, targets)
                    pbar.set_postfix({"Loss": loss.item()})

                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                
                    pbar.update(1)

                    preds = np.array(fin_outputs)
                    trues = np.array(fin_targets)

                    for label_num, label in enumerate(self.target_label):
                        pred = preds[:, label_num]
                        true = trues[:, label_num]
                        mse = self.metrics_mse(pred, true)
                        mae = self.metrics_mae(pred,true)
                        mape = self.metrics_mape(pred,true)
                        
                        #mseの辞書更新
                        if label in total_mse:
                            total_mse[label]['MSE'] += [mse]
                        else:
                            sub = {'MSE': [mse]}
                            total_mse.update({label: sub})
                        
                        #maeの辞書更新
                        if label in total_mae:
                            total_mae[label]['MAE'] += [mae]
                        else:
                            sub = {'MAE': [mae]}
                            total_mae.update({label: sub})
                        
                        #mapeの辞書更新
                        if label in total_mape:
                            total_mape[label]['MAPE'] += [mape]
                        else:
                            sub = {'MAPE': [mape]}
                            total_mape.update({label: sub})
                        
            print(f"Epoch [{epoch+1}] valid Loss:{loss.item()}")
            writer.add_scalar("Valid/valid_Loss", loss.item(), epoch+1)

            for label in total_mse:
                total_mse = np.average(total_mse[label]['MSE'])
                print('Valid Total: T:{} -> MSE:{}'.format(label, total_mse))
                if label == "CEA":
                    writer.add_scalar("Valid/CEA_MSE", total_mse, epoch+1)
                elif label == "CA19-9":
                    writer.add_scalar("Valid/CA19_9_MSE", total_mse, epoch+1)
            
            for label in total_mae:
                total_mae = np.average(total_mae[label]['MAE'])
                print('Valid Total: T:{} -> MAE:{}'.format(label, total_mae))
                if label == "CEA":
                    writer.add_scalar("Valid/CEA_MAE", total_mae, epoch+1)
                elif label == "CA19-9":
                    writer.add_scalar("Valid/CA19_9_MAE", total_mae, epoch+1)
            
            for label in total_mape:
                total_mape = np.average(total_mape[label]['MAPE'])
                print('Valid Total: T:{} -> MAPE:{}'.format(label, total_mape))
                if label == "CEA":
                    writer.add_scalar("Valid/CEA_MAPE", total_mape, epoch+1)
                elif label == "CA19-9":
                    writer.add_scalar("Valid/CA19_9_MAPE", total_mape, epoch+1)
    
    writer.close()

    
    def test(self):
        self.model.eval()
        total_mse = {}
        total_mae = {}
        total_mape = {}
        for i, data in enumerate(self.testing_loader, 0):
            fin_targets = []
            fin_outputs = []

            inputs = data['inputs'].to(self.device, dtype=torch.float)
            targets = data['targets'].to(self.device, dtype=torch.float)
            outputs = self.model(inputs)

            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
        
        preds = np.array(fin_outputs)
        trues = np.array(fin_targets)
        
        
        folder_path = './NN_log/regre/result/'
        os.makedirs(folder_path, exist_ok=True)

        #MSE, MAE, MAPEを計算・出力
        for label_num, label in enumerate(self.target_label):
            pred = preds[:, label_num]; true = trues[:, label_num]
            mse = self.metrics_mse(pred, true)
            mae = self.metrics_mae(pred, true)
            mape = self.metrics_mape(pred, true)

            if label in total_mse:
                total_mse[label]['MSE'] += [mse]
            else:
                sub = {'MSE': [mse]}
                total_mse.update({label: sub})
            
            if label in total_mae:
                total_mae[label]['MAE'] += [mae]
            else:
                sub = {'MAE': [mae]}
                total_mae.update({label: sub})
            
            if label in total_mape:
                total_mape[label]['MAPE'] += [mape]
            else:
                sub = {'MAPE': [mape]}
                total_mape.update({label: sub})

            print('{} T:{} -> MSE:{}'.format(label_num, label, mse), end="  ")
            # mse(平均2乗誤差): 0に近いほど良い # R^2(決定係数): 相関係数の2乗
            print('{} T:{} -> MAE:{}'.format(label_num, label, mae), end="  ")
            print('{} T:{} -> MAPE:{}'.format(label_num, label, mape), end="  ")
            if label == "CEA":
                writer.add_scalar("Test/CEA_MSE", mse, i+1)
            elif label == "CA19-9":
                writer.add_scalar("Test/CA19_9_MSE", mse, i+1)

            if label == "CEA":
                writer.add_scalar("Test/CEA_MAE", mae, i+1)
            elif label == "CA19-9":
                writer.add_scalar("Test/CA19_9_MAE", mae, i+1)
            
            if label == "CEA":
                writer.add_scalar("Test/CEA_MAPE", mape, i+1)
            elif label == "CA19-9":
                writer.add_scalar("Test/CA19_9_MAPE", mape, i+1)

            np.save(folder_path+f'metrics_{label_num}_{label}_{i}.npy', np.array([mse]))
            np.save(folder_path+f'metrics_{label_num}_{label}_{i}.npy', np.array([mae]))
            np.save(folder_path+f'metrics_{label_num}_{label}_{i}.npy', np.array([mape]))
            np.save(folder_path+f'pred_{label_num}_{label}_{i}.npy', pred)
            np.save(folder_path+f'true_{label_num}_{label}_{i}.npy', true)
        
        for label in total_mse: 
            total_mse_num = np.average(total_mse[label]['MSE'])
            print('Test Total: T:{} -> MSE:{}'.format(label, total_mse_num))
        for label in total_mae: 
            total_mae_num = np.average(total_mae[label]['MAE'])
            print('Test Total: T:{} -> MAE:{}'.format(label, total_mae_num))
        for label in total_mape: 
            total_mape_num = np.average(total_mape[label]['MAPE'])
            print('Test Total: T:{} -> MAPE:{}'.format(label, total_mape_num))

writer.close()
