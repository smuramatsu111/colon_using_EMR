# LSTM_regre.py実行時に呼び出すクラス
from genericpath import exists
import logging
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from decimal import *
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanAbsolutePercentageError

#log管理
sample_num = 5000 #任意
d_model = "numL" #任意

log_pass_Train_Loss = "output/loss_cheack/regre/" + d_model + "/train/loss/" + str(sample_num)
log_pass_Valid_Loss = "output/loss_cheack/regre/" + d_model + "/valid/loss/" + str(sample_num)
log_pass_Valid_CEA = "output/loss_cheack/regre/" + d_model + "/valid/MSE_CEA/" + str(sample_num)
log_pass_Valid_CA199 = "output/loss_cheack/regre/" + d_model + "/valid/MSE_CA19_9/" + str(sample_num)
log_pass_Test_CEA = "output/loss_cheack/regre/" + d_model + "/test/MSE_CEA/" + str(sample_num)
log_pass_Test_CA199 = "output/loss_cheack/regre/" + d_model + "/test/MSE_CA19_9/" + str(sample_num)
writer_Train = SummaryWriter(log_dir = log_pass_Train_Loss)
writer_Valid_Loss = SummaryWriter(log_dir = log_pass_Valid_Loss)
writer_Valid_CEA = SummaryWriter(log_dir = log_pass_Valid_CEA)
writer_Valid_CA199 = SummaryWriter(log_dir = log_pass_Valid_CA199)
writer_Test_CEA = SummaryWriter(log_dir = log_pass_Test_CEA)
writer_Test_CA199 = SummaryWriter(log_dir = log_pass_Test_CA199)

pd.set_option('display.max_columns', None)
logging.basicConfig(level=logging.ERROR)

# 回帰の際に利用
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target_label, del_label, train_term, pred_term):
        del_label += ['date']
        data = dataframe.drop(del_label, axis=1) #del_labelに追記されているlabelのデータを除去
        self.inputs = data
        self.targets = data[target_label].copy() #CEA,CA19-9に対応する値を取得
        self.train_term = train_term #21
        self.test_term = pred_term #1

    def __len__(self):
        return len(self.targets) - self.train_term  - self.test_term + 1

    def __getitem__(self, index):
        af = index+self.train_term
        input_df = self.inputs.iloc[index: af, :]
        target_df = self.targets.iloc[af: af+self.test_term, :]
        
        input_np = input_df.values
        target_np = target_df.values

        return {
            'inputs': torch.tensor(input_np, dtype=torch.long),
            'targets': torch.tensor(target_np, dtype=torch.float)
        } 

# 増加/減少/変化なしの3クラス分類の際に利用
class SequenceDataset_class(Dataset):
    def __init__(self, dataframe, target_label, del_label, train_term, pred_term):
        del_label += ['date']
        data = dataframe.drop(del_label, axis=1) 
        self.inputs = data
        self.targets = data[target_label].copy()
        self.targets = self.targets.diff().fillna(0) #差分を取得してクラス情報を算出
        self.target_len = len(target_label)
        self.train_term = train_term
        self.test_term = pred_term

    def __len__(self):
        return len(self.targets) - self.train_term  - self.test_term + 1

    def __getitem__(self, index):
        af = index+self.train_term
        input_df = self.inputs.iloc[index: af, :]
        target_df = self.targets.iloc[af: af+self.test_term, :]
        
        input_np = input_df.values
        target_np = np.sign([[0,0]]).astype(np.int)
        target_np = np.eye(3)[target_np].reshape([1, self.target_len*3])


        return {
            'inputs': torch.tensor(input_np, dtype=torch.long),
            'targets': torch.tensor(target_np, dtype=torch.float)
        }

# LSTMのモデル構造定義
class LSTMClass(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, dropout, list_num):
        super().__init__()
        self.l1 = torch.nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.d = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(hidden_size, list_num)

    def forward(self, inputs):
        outputs, (h_n, c_n) = self.l1(inputs)
        output = self.d(outputs)
        output = self.l3(output)
        return output

'''class multiLSTMClass(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, dropout, list_num):
        super().__init__()
        self.l1 = torch.nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.l2 = torch.nn.LSTM(hidden_size, 256, batch_first=True)
        self.d = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(256, list_num)

    def forward(self, inputs):
        outputs, output_hc = self.l1(inputs)
        outputs, _ = self.l2(outputs)
        output = self.d(outputs)
        output = self.l3(output)
        return output'''
        
# 回帰の際の学習・評価・テスト処理
class MLregression():
    def __init__(self, training_loader, validate_loader, testing_loader, model, device, optimizer, target, test_term):
        self.training_loaders= training_loader
        self.validate_loaders = validate_loader
        self.testing_loaders = testing_loader
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.target_label = target
        self.pred_len = test_term

    def loss_fn(self, outputs, targets):
        return torch.nn.MSELoss()(outputs, targets)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def metrics(self, pred, true):
        return np.mean((true-pred)**2)

    def train(self, epoch, EPOCHS):
        self.model.train()
        with tqdm(total=len(self.training_loaders), unit='batch', position=0, leave=True) as pbar:
            pbar.set_description(
                f"Epoch[{'{0:2d}'.format(epoch + 1)}/{EPOCHS}](train)")

            for loader in self.training_loaders:
                for data in loader:
                    inputs = data['inputs'].to(self.device, dtype=torch.float)
                    targets = data['targets'].to(self.device, dtype=torch.float)
                    outputs = self.model(inputs)
                    outputs = outputs[:, :self.pred_len, :]

                    loss = self.loss_fn(outputs, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
        print(f"Epoch [{epoch+1}] train Loss:{loss.item()}")
        writer_Train.add_scalar("Train/Loss", loss.item(), epoch+1)
    
    writer_Train.close()
        

    def validation(self, epoch, EPOCHS, *args):
        self.model.eval()
        ids_list = []
        target_list = []
        output_list = []
        total = {}
        with torch.no_grad():
            with tqdm(total=len(self.validate_loaders), unit='batch') as pbar:
                pbar.set_description(f"Epoch[{'{0:2d}'.format(epoch + 1)}/{EPOCHS}](valid)")
                for loader in self.validate_loaders:
                    fin_targets = []
                    fin_outputs = []
                    for data in loader:
                        inputs = data['inputs'].to(self.device, dtype=torch.float)
                        targets = data['targets'].to(self.device, dtype=torch.float)
                        outputs = self.model(inputs)
                        outputs = outputs[:, :self.pred_len, :]

                        loss = self.loss_fn(outputs, targets)
                        pbar.set_postfix({"Loss": loss.item()})

                        fin_targets.extend(targets.cpu().detach().numpy().tolist())
                        fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                    
                    pbar.update(1)

                    preds = np.array(fin_outputs)
                    trues = np.array(fin_targets)

                    for label_num, label in enumerate(self.target_label):
                        pred = preds[:, :, label_num]
                        true = trues[:, :, label_num]
                        print(f"metrics.pred.shape:{pred.shape}")
                        print(f"metrics.true.shape:{true.shape}")
                        mse = self.metrics(pred, true)

                        if label in total:
                            total[label]['MSE'] += [mse]
                        else:
                            sub = {'MSE': [mse]}
                            total.update({label: sub})

            print(f"Epoch [{epoch+1}] valid Loss:{loss.item()}")
            writer_Valid_Loss.add_scalar("Valid/valid_Loss", loss.item(), epoch+1)

            for label in total:
                total_mse = np.average(total[label]['MSE'])
                print('Test Total: T:{} -> MSE:{}'.format(label, total_mse))
                if label == "CEA":
                    writer_Valid_CEA.add_scalar("Valid/CEA_MSE", total_mse, epoch+1)
                elif label == "CA19-9":
                    writer_Valid_CA199.add_scalar("Valid/CA19_9_MSE", total_mse, epoch+1)
    
    writer_Valid_Loss.close()
    writer_Valid_CEA.close()
    writer_Valid_CA199.close()

    def test(self):
        self.model.eval()
        total = {}
        for i, loader in enumerate(self.testing_loaders, 0):
            fin_targets = []
            fin_outputs = []
            for data in loader:
                inputs = data['inputs'].to(self.device, dtype=torch.float)
                targets = data['targets'].to(self.device, dtype=torch.float)
                outputs = self.model(inputs)
                outputs = outputs[:, :self.pred_len, :]

                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
            
            preds = np.array(fin_outputs)
            trues = np.array(fin_targets)
            
            folder_path = './LSTM/result/'
            os.makedirs(folder_path, exist_ok=True)

            for label_num, label in enumerate(self.target_label):
                pred = preds[:, :, label_num]; true = trues[:, :, label_num]
                mse = self.metrics(pred, true)

                if label in total:
                    total[label]['MSE'] += [mse]
                else:
                    sub = {'MSE': [mse]}
                    total.update({label: sub})

                print('{} T:{} -> MSE:{}'.format(label_num, label, mse), end="  ")
                # mse(平均2乗誤差): 0に近いほど良い # R^2(決定係数): 相関係数の2乗
                if label == "CEA":
                    writer_Test_CEA.add_scalar("Test/CEA_MSE", mse, i+1)
                elif label == "CA19-9":
                    writer_Test_CA199.add_scalar("Test/CA19_9_MSE", mse, i+1)

                np.save(folder_path+f'metrics_{label_num}_{label}_{i}.npy', np.array([mse]))
                np.save(folder_path+f'pred_{label_num}_{label}_{i}.npy', pred)
                np.save(folder_path+f'true_{label_num}_{label}_{i}.npy', true)
            
        for label in total:
            total_mse = np.average(total[label]['MSE'])
            print('Test Total: T:{} -> MSE:{}'.format(label, total_mse))
    
    writer_Test_CEA.close()
    writer_Test_CA199.close()
