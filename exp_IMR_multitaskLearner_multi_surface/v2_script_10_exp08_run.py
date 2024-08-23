## install
# !pip install xgboost

## import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.nn import DataParallel

## variables
seed = 42
ratio_test = 0.2
n_sample_imr, exp_name = 2500, 'exp8' ## imr (pca_components) length 1024, siamese model structure changed 
## use diverse set to create the pair
np.random.seed(seed)
torch.manual_seed(seed)

## class
class SiameseNetwork(torch.nn.Module):
    def __init__(self, len_embedding, abstract_len_embedding):
        super(SiameseNetwork, self).__init__()
        self.loss = nn.L1Loss(reduction="mean") 
        self.len_embedding = len_embedding
        self.abstract_len_embedding = abstract_len_embedding  
        self.nn_reg = nn.Sequential(
            ## 1024 to 2048
            nn.Linear(self.len_embedding, int(self.len_embedding*2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(self.len_embedding*2)), 
            ## 2048 to 1536
            nn.Linear(int(self.len_embedding*2), int(self.len_embedding*1.5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(self.len_embedding*1.5)),
            ## 1526 to 1048
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

## function

def get_secondary_env(env1, env2):
    x_ea, y_ea = env1[0], env1[1]
    print(x_ea.shape, y_ea.shape)
    x_eb, y_eb = env2[0], env2[1]
    print(x_eb.shape, y_eb.shape)
    list_secondary_feature, list_secondary_target = [], []
    for i in range(x_ea.shape[0]):
        for j in range(x_eb.shape[0]):
            sf = np.hstack((x_ea[i], x_eb[j]))
            st = y_ea[i] - y_eb[j]
            list_secondary_feature.append(sf)
            list_secondary_target.append(st)
    array_secondary_feature = np.array(list_secondary_feature, dtype='float32')
    array_secondary_target = np.array(list_secondary_target, dtype='float32').reshape((-1, 1))
    senv = torch.from_numpy(array_secondary_feature), torch.from_numpy(array_secondary_target)
    print(senv[0].shape, senv[1].shape)
    return senv

def get_model_siamese(senvironments, len_embedding, abstract_len_embedding, batch_size=30000, num_gpus=4):
    print(f'len_embedding: {len_embedding}, abstract_len_embedding: {abstract_len_embedding}')

    _lr, num_iterations = 5e-3, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_siamese = SiameseNetwork(len_embedding, abstract_len_embedding)
    if torch.cuda.is_available():
        model_siamese = DataParallel(model_siamese)
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
            with open('/curdir/v2_script_10/logger_' + exp_name + '.log', 'a+') as file1:
                file1.writelines(f'epoch: {epoch}, total_loss: {total_loss:.6f}\n\n')

    return model_siamese

## check for word ('cathub/ocp', 'nre/energy', 'original/pca')
## common (dimenet++, Two-Functional-Model, xgboost) 
## (cathub/ocp x org/pca/imr x MAE/r2) 

## 5. cathub, imr, (MAE, r2)
## 6. ocp, imr, (MAE, r2)

df1 = pd.read_pickle(f'/curdir/datasets/df_cathub_dpp_combined.pickle')
X1 = df1.iloc[:, :-1].values
y1 = df1['nre'].values
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=ratio_test, random_state=seed)
df2 = pd.read_pickle(f'/curdir/datasets/df_ocp_dpp_combined.pickle')
X2 = df2.iloc[:, :-1].values
y2 = df2['energy'].values
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=ratio_test, random_state=seed)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)

## choose 'n' samples from cathub
## turn to 'n(n-1)' secondary by '2d'
idx1 = np.random.choice(X_train1.shape[0], 2 * n_sample_imr, replace=False)
# Split the indices into two parts
idx1a = idx1[:n_sample_imr]
idx1b = idx1[n_sample_imr:]
# Create samples for env1a
X_train1_sampled_a = X_train1[idx1a]
y_train1_sampled_a = y_train1[idx1a]
env1a = (X_train1_sampled_a, y_train1_sampled_a)
# Create samples for env1b
X_train1_sampled_b = X_train1[idx1b]
y_train1_sampled_b = y_train1[idx1b]
env1b = (X_train1_sampled_b, y_train1_sampled_b)
# Assuming get_secondary_env is a function defined elsewhere
senv1 = get_secondary_env(env1a, env1b)
## choose 'n' samples from ocp
## turn to 'n(n-1)' secondary by '2d'
idx2 = np.random.choice(X_train2.shape[0], 2 * n_sample_imr, replace=False)
# Split the indices into two parts
idx2a = idx2[:n_sample_imr]
idx2b = idx2[n_sample_imr:]
# Create samples for env2a
X_train2_sampled_a = X_train2[idx2a]
y_train2_sampled_a = y_train2[idx2a]
env2a = (X_train2_sampled_a, y_train2_sampled_a)
# Create samples for env2b
X_train2_sampled_b = X_train2[idx2b]
y_train2_sampled_b = y_train2[idx2b]
env2b = (X_train2_sampled_b, y_train2_sampled_b)
# Assuming get_secondary_env is a function defined elsewhere
senv2 = get_secondary_env(env2a, env2b)
## build a SiameseIMR to transform 'd' to 'pca(d)',
## use that 'SiameseIMR transformation' instead of pca
senvironments = [senv1, senv2]
pca_components = int(int(X_train1.shape[1])//1)
len_embedding, abstract_len_embedding = int(senvironments[0][0].shape[1])//2, pca_components    
model_siamese = get_model_siamese(
    senvironments, 
    len_embedding, 
    abstract_len_embedding,    
)
## save the model
model_path = '/curdir/v2_script_10/model_siamese_' + exp_name + '.pt'
torch.save(model_siamese.state_dict(), model_path)

# load the model
model_siamese = SiameseNetwork(len_embedding, abstract_len_embedding)
model_siamese.load_state_dict(torch.load(model_path))
model_siamese.eval()

X_train_siamese1 = model_siamese.forward_reg(torch.from_numpy(X_train1).float()).detach().numpy() 
X_test_siamese1 = model_siamese.forward_reg(torch.from_numpy(X_test1).float()).detach().numpy()

lr = 0.2
depth = 8
n_est = 500
model1 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
model1.fit(X_train_siamese1, y_train1)

y_pred1 = model1.predict(X_test_siamese1)
mae1 = mean_absolute_error(y_test1, y_pred1)
r2score1 = r2_score(y_test1, y_pred1)

plt.figure(figsize=(4, 3))
plt.scatter(y_test1, y_pred1, alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title(f'(cathub, siamese)')
plt.plot([y_test1.min(), y_test1.max()], [y_test1.min(), y_test1.max()], 'k--', lw=2)
plt.show()
print(f'X_train shape: {X_train1.shape}, X_test shape: {X_test1.shape}')
print(f'y_train shape: {y_train1.shape}, y_test shape: {y_test1.shape}')
print(f'train to test ratio: {1-ratio_test}:{ratio_test}')
print(f'Mean Abs. Error: {mae1:.2f}')
print(f'R2-score: {r2score1:.2f}')

X_train_siamese2 = model_siamese.forward_reg(torch.from_numpy(X_train2).float()).detach().numpy() 
X_test_siamese2 = model_siamese.forward_reg(torch.from_numpy(X_test2).float()).detach().numpy()

lr = 0.2
depth = 8
n_est = 500
model2 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
model2.fit(X_train_siamese2, y_train2)

y_pred2 = model2.predict(X_test_siamese2)
mae2 = mean_absolute_error(y_test2, y_pred2)
r2score2 = r2_score(y_test2, y_pred2)

plt.figure(figsize=(4, 3))
plt.scatter(y_test2, y_pred2, alpha=0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title(f'(ocp, siamese)')
plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'k--', lw=2)
plt.show()
print(f'X_train shape: {X_train2.shape}, X_test shape: {X_test2.shape}')
print(f'y_train shape: {y_train2.shape}, y_test shape: {y_test2.shape}')
print(f'train to test ratio: {1-ratio_test}:{ratio_test}')
print(f'Mean Abs. Error: {mae2:.2f}')
print(f'R2-score: {r2score2:.2f}')