import os
import numpy as np
import torch
import pickle
import json
import pandas as pd
import torch.nn as nn
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from itertools import cycle
import torch.autograd as autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.decomposition import PCA
import torch.optim as optim
from torch.nn import DataParallel

seed = 42
ratio_test = 0.2
np.random.seed(seed)
torch.manual_seed(seed)
exp_name = 'exp_12e3'
curdir = '/curdir/' ## '/curdir/' ## ''
device_ids = [1, 3] ## [0, 1, 2, 3]

class SiameseNetwork(torch.nn.Module):
    def __init__(self, len_embedding, abstract_len_embedding):
        super(SiameseNetwork, self).__init__()
        self.loss = nn.L1Loss(reduction="mean") 
        self.len_embedding = len_embedding
        self.abstract_len_embedding = abstract_len_embedding  
        self.nn_reg = nn.Sequential(
            ## 1024 to 2048
            nn.Linear(self.len_embedding, int(self.len_embedding*2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(self.len_embedding*2)), 
            ## 2048 to 1536
            nn.Linear(int(self.len_embedding*2), int(self.len_embedding*1.5)),
            nn.ReLU(),
            nn.BatchNorm1d(int(self.len_embedding*1.5)),
            ## 1526 to 1024
            nn.Linear(int(self.len_embedding*1.5), self.abstract_len_embedding),
        )
        self.nn_final_reg = nn.Sequential(
            ##  (1024+1024) to 1
            nn.Linear(self.abstract_len_embedding * 2, 1),
        )

    def forward_reg(self, x):
        output = self.nn_reg(x)
        return output

    def forward_final_reg(self, x):
        output = self.nn_final_reg(x)
        return output

    def forward(self, fp1, fp2):
        a = self.forward_reg(fp1)
        b = self.forward_reg(fp2)
        x = torch.cat([a, b], dim=1)  # hstack
        output = self.forward_final_reg(x)
        return output

def get_secondary_env(env1):
    x, y = env1[0], env1[1]
    print(x.shape, y.shape)
    list_secondary_feature, list_secondary_target = [], []
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i != j:
                sf = np.hstack((x[i], x[j]))
                st = y[i] - y[j]
                list_secondary_feature.append(sf)
                list_secondary_target.append(st)
    array_secondary_feature = np.array(list_secondary_feature, dtype='float32')
    array_secondary_target = np.array(list_secondary_target, dtype='float32').reshape((-1, 1))
    senv = torch.from_numpy(array_secondary_feature), torch.from_numpy(array_secondary_target)
    print(senv[0].shape, senv[1].shape)
    return senv

def get_model_siamese(senvironments, len_embedding, abstract_len_embedding, batch_size=512, num_gpus=4):
    print(f'len_embedding: {len_embedding}, abstract_len_embedding: {abstract_len_embedding}')

    _lr, num_iterations = 1e-3, 1000
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu") ## torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_siamese = SiameseNetwork(len_embedding, abstract_len_embedding)
    if torch.cuda.is_available():    
        model_siamese = DataParallel(model_siamese, device_ids=device_ids)
    model_siamese.to(device)
    optimizer_siamese = torch.optim.Adam(model_siamese.parameters(), lr=_lr)    

    for epoch in range(num_iterations):
        total_loss = 0.0

        for x, y in senvironments:
            p = torch.randperm(len(x))
            x, y = x[p], y[p]
            
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size].to(device)
                batch_y = y[i:i + batch_size].to(device)
                fp1 = batch_x[:, list(range(0, len_embedding, 1))]
                fp2 = batch_x[:, list(range(len_embedding, 2 * len_embedding, 1))]
                y_pred_siamese = model_siamese(fp1, fp2)
                batch_loss = model_siamese.module.loss(y_pred_siamese, batch_y) if isinstance(model_siamese, DataParallel) else model_siamese.loss(y_pred_siamese, batch_y)
                
                optimizer_siamese.zero_grad()
                batch_loss.backward()
                optimizer_siamese.step()
                
                total_loss += batch_loss.item()
                

        if epoch % 1 == 0:
            with open(f'{curdir}v2_script_10/logger_' + exp_name + '.log', 'a+') as file1:
                file1.writelines(f'epoch: {epoch}, total_loss: {total_loss:.6f}\n\n')

    return model_siamese

list_pid = [
    'AraComputational2022', ## PBE-D3
    # 'BajdichWO32018', ## PBE+U
    'BoesAdsorption2018', ## RPBE
    'ComerUnraveling2022', ## PBE+U
    'HossainInvestigation2022', ## PBE+U    
    # 'MamunHighT2019',    
]

senvironments = []
for pid in list_pid:
    with open(f'{curdir}v2_script_10/exp12/df_{pid}.pickle', 'rb') as f:
        df = pickle.load(f)  
        print(df.shape)
        X = df.iloc[:, :-1].values
        y = df['nre'].values
        env = (X, y)
        senv = get_secondary_env(env)
        print(env[0].shape, env[1].shape, senv[0].shape, senv[1].shape)
        print()
        senvironments.append(senv)
print()
print(len(senvironments))
len_embedding, abstract_len_embedding = 1024, 1024    
model_siamese = get_model_siamese(
    senvironments, 
    len_embedding, 
    abstract_len_embedding,    
)
## save the model
model_path = f'{curdir}v2_script_10/model_siamese_{exp_name}.pt'
torch.save(model_siamese.state_dict(), model_path)