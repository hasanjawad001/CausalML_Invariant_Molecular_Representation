## comments

## 10 trials, 
## 6 envs,
## 3 models (regular, PCA, Siamese) ## no SiameseIRM since specific env is used only
## tasks:
##     1. seperate feature target for 6 envs (6 files in v10/datasets/)
##     2. use utils for 2a and 2b- 
##        2a. take the SMILES notation and convert it by "Chemical_SMILES_to_Flat_Fingerprint_RingStructs.ipynb -> convertNewSmilesToOldSmiles" 
##            there are two cells which has the "convertNewSmilesToOldSmiles" function, use the first one (do not use the later cell's function)
##        2b. after converting use "Chemical_SMILES_to_Fingerprint.ipynb -> readSmilesToFingerprints" function to get the fingerprints
##     3. use 10 trials, 6 envs, 3 models to evaluate the performance
##     4. ----------USE SIAMESE NN ALMOST SIMILAR TO PCA
## extra:
##     1. estimator PCA from version 1.0.2 (saved object) when using version 0.24.2 (here)z;
##        1a. import nltk, sklearn
##        1b. !pip install -U scikit-learn==1.0.2
##        1c. print('The nltk version is {}.'.format(nltk.__version__))
##        1d. print('The scikit-learn version is {}.'.format(sklearn.__version__))
## extra:
##     1. train with 768(24) pretrained model (abstract len emb depends on PCA(768(24)) on 90%(99%)?)
##     *2. use all validation model (with struct_code 0 (simple) and 1 (complex))
##     *3. use v8 (0- simple/PCA) struct
##     *4. use v7 (1- complex) struct
##     5. use v9 (simp+comp, solo env, 768)
##     6. use v10 (simp+comp, solo env, 24)
##     7. use v11 (simp+comp, beef env, 768)

## imports

import numpy as np
from numpy import genfromtxt
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForMaskedLM
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader    
import torchvision.utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import random
# from pytorch_lightning.core.lightning import LightningModule
# from pytorch_lightning import Trainer
# import pytorch_lightning as pl
# from ray_lightning import RayPlugin, RayShardedPlugin
import ray
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import os
import re
import requests
from tqdm.auto import tqdm
import pickle as pk
import time
import utils
from sklearn.decomposition import PCA
import math
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem

import nltk, sklearn
print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))

## classes

class SiameseNetwork(torch.nn.Module):
    def __init__(self, len_embedding, abstract_len_embedding, use_irm=False, n_hidden_node=32, struct_code=0):
        '''
            struct_code {0=simple, 1=comple} structure
        '''
        
        super(SiameseNetwork, self).__init__()
        self.loss = nn.L1Loss(reduction="mean") 
        self.use_irm = use_irm
        self.len_embedding = len_embedding
        self.abstract_len_embedding = abstract_len_embedding  
        self.n_hidden_node = n_hidden_node
        #-----------change_1
        if struct_code == 0:
            self.nn_reg = nn.Sequential(
                nn.Linear(self.len_embedding, self.abstract_len_embedding),
            )
        elif struct_code == 1:
            self.nn_reg = nn.Sequential(
                nn.Linear(self.len_embedding, self.n_hidden_node),nn.ReLU(inplace=True),nn.BatchNorm1d(self.n_hidden_node),            
                nn.Linear(self.n_hidden_node, int(self.n_hidden_node/4)),nn.ReLU(inplace=True),nn.BatchNorm1d(int(self.n_hidden_node/4)),nn.Dropout(p=0.2),
                nn.Linear(int(self.n_hidden_node/4), self.abstract_len_embedding),
            )
        else:
            self.nn_reg = nn.Sequential(
                nn.Linear(self.len_embedding, self.abstract_len_embedding),
            )            
        #-----------change_1
        self.nn_final_reg = nn.Sequential(
            nn.Linear(self.abstract_len_embedding * 2, self.n_hidden_node),nn.ReLU(inplace=True),nn.BatchNorm1d(self.n_hidden_node),
            nn.Linear(self.n_hidden_node, int(self.n_hidden_node/4)),nn.ReLU(inplace=True),nn.BatchNorm1d(int(self.n_hidden_node/4)),nn.Dropout(p=0.2),
            nn.Linear(int(self.n_hidden_node/4), 1),
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

    def compute_penalty(self, losses, dummy_w):
        g = grad(losses, dummy_w, create_graph=True)[0]
        r = g.pow(2)
        return r     

## functions

def get_secondary_env(env=None):
    x_e, y_e = env[0].numpy(), env[1].numpy()

    list_primary_feature, list_primary_target = [], []
    list_secondary_feature, list_secondary_target = [], []

    for i in range(x_e.shape[0]):
        list_primary_feature.append(x_e[i])
        list_primary_target.append(y_e[i])
        i += 1

    for i in range(len(list_primary_feature)):
        for j in range(len(list_primary_feature)):
            if i == j: ## n * (n-1) combinations
                pass
            else:
                a = list_primary_feature[i]
                b = list_primary_feature[j]
                sf = np.hstack((a, b))
                st = list_primary_target[i] - list_primary_target[j]
                list_secondary_feature.append(sf)
                list_secondary_target.append(st)
    array_secondary_feature = np.array(list_secondary_feature, dtype='float32')
    array_secondary_target = np.array(list_secondary_target, dtype='float32').reshape((-1, 1))
    senv = torch.from_numpy(array_secondary_feature), torch.from_numpy(array_secondary_target)
    return senv

#### 1. seperate feature target for 6 envs (6 files in v10/datasets/)

dir_datasets_prefix = '/curdir/'

dir_datasets = dir_datasets_prefix + 'datasets/' ## 'datasets/v2/'

df_1_energy = pd.read_excel(dir_datasets + 'file_1.xlsx', sheet_name=0) ## energy sheet
m_1_energy = df_1_energy.values[1:]
m_1_name = m_1_energy[:,[0]]
m_1_smiles = m_1_energy[:,[1]]
m_1_energy_envs = m_1_energy[:,[2, 3, 4, 5]]
for env_no in range(m_1_energy_envs.shape[1]):
    m_1_name_smiles = np.hstack((m_1_name, m_1_smiles))
    feature_target = np.hstack((m_1_name_smiles, m_1_energy_envs[:, [env_no]]))
    np.savetxt(dir_datasets + 'smiles_vs_energy_env_' + str(env_no) + '.csv', feature_target, delimiter = ",", fmt='%s')

## funtion to generate pretrained embedding from SMILE notations

## change_2 : change in v10(24) from v9(768) to accomodate manual 24 length embedding 

def generate_morgan_fingerprint(smiles_str, radius=2, nbits=24): ## how many nbits should be chosen? nbits=24, radius=2
    """Generate Morgan fingerprint from a SMILES string."""
    molecule = Chem.MolFromSmiles(smiles_str)
    if molecule is None:
        return None
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nbits)
    return list(morgan_fp)

def get_embed_from_smiles(list_species_smiles, emb_style='auto_768'):
    if emb_style=='manual_24':
        list_emb = []
        for i, s1 in enumerate(list_species_smiles):
            try:
                s2 = utils.convertNewSmilesToOldSmiles(s1)
                s3 = utils.readSmilesToFingerprints(s2)
            except Exception as e:
                print(env_no, i, e)
                s3 = ['-'] * 24
            list_emb.append(s3)    
        return np.array(list_emb)
    elif emb_style=='morgan_24':        
        list_emb = []
        for ismiles, smiles in enumerate(list_species_smiles):
            fp = generate_morgan_fingerprint(smiles)
            list_emb.append(fp)    
        return np.array(list_emb)        
    else:
        ## --------------- this is commented out in the server ----------------------- ##
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/chEMBL26_smiles_v2")
        model = AutoModel.from_pretrained("mrm8488/chEMBL26_smiles_v2")
        # Load the model into the GPU if avilabile and switch to inference mode
        fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)
        sequences_Example = list_species_smiles
        embedding_1 = fe(sequences_Example)
        n1, n2, n3 = len(embedding_1), len(embedding_1[0]), len(embedding_1[0][0])
        embedding_2 = np.zeros((n1, n3))
        list_embedding = []
        for i, e1 in enumerate(embedding_1):
            e2 = np.array(e1).mean(axis=0)        
            embedding_2[i, :] = e2
        embedding_2 = np.array(embedding_2)                    
        return embedding_2

dir_datasets = dir_datasets_prefix + 'datasets/'
for env_no in range(4): 
    file_name = dir_datasets + 'smiles_vs_energy_env_' + str(env_no) + '.csv'
    df = pd.read_csv(file_name, sep=",", header=None) 
    m = df.values
    smiles = [row[1] for row in m] ## 1 for smiles ==> (row[0],row[1],row[2]) == (name, smiles, energy)
    
    array_emb = get_embed_from_smiles(smiles, emb_style='morgan_24') ## change_2 : change in v10(24) from v9(768) to accomodate manual 24 length embedding     
    feature_target = np.hstack((array_emb, m[:,[2]])) ## index 2 used here for energy
    
    print(feature_target.shape)
    np.savetxt(dir_datasets_prefix + 'datasets/embedding_vs_energy_env_' + str(env_no) + '.csv', feature_target, delimiter = ",", fmt='%s')
#     print()

## change_2 : change in v10(24) from v9(768) to accomodate manual 24 length embedding 
## skip invalid samples => index in each env [41,41,41,41,'-',9] for manual_24
## skip invalid samples => index in each env ['-','-','-','-','-','-'] for auto_768

list_skip_index = ['-','-','-','-']
dir_datasets = dir_datasets_prefix + 'datasets/'
for env_no in range(4): 
    data = genfromtxt(dir_datasets + 'embedding_vs_energy_env_' + str(env_no) + '.csv', delimiter=',', dtype='float32')
    skip_index = list_skip_index[env_no]
    if skip_index == '-':
        feature = data[:, 0:-1]
        target = data[:, [-1]]
    elif type(skip_index)== type(0):
        feature = np.vstack((data[:skip_index, 0:-1], data[skip_index+1:, 0:-1]))
        target = np.vstack((data[:skip_index, [-1]], data[skip_index+1:, [-1]]))
    else:
        print('Error!!!')
    feature_target = np.hstack((feature, target))
    print(feature_target.shape)
    np.savetxt(dir_datasets_prefix + 'datasets/filtered_embedding_vs_energy_env_' + str(env_no) + '.csv', feature_target, delimiter = ",", fmt='%s')

ray.shutdown()
ray.init(ignore_reinit_error=True, num_cpus=12) ## detects automatically: num_cpus=64

## functions

def get_best_models(
    len_embedding=None, abstract_len_embedding=None, num_iterations=None, 
    X_train=None, y_train=None, trial_no=None, env_no=None
):
    
    env = torch.from_numpy(X_train), torch.from_numpy(y_train)
    senv = get_secondary_env(env=env)
    x_env, y_env = senv
#     n = int(x_env.shape[0])
#     if env_no == 5:
#         n_train = int(n * 0.67) ##
#     else:
#         n_train = int(n * 0.8) ## train: validation == 80:20
#     p = torch.randperm(n)
#     p_train, p_test = p[0:n_train], p[n_train:]
#     X_train, X_test, y_train, y_test = x_env[p_train], x_env[p_test], y_env[p_train], y_env[p_test]   
    
    
    ## model siamese
    list_model_info = []
    best_model_info, best_loss_siamese = (None, None, None, None), math.inf
    #-----------change_1    
    for _n_hidden_node in [128]: ## hint: select something that is divisible by 4  ## [512,256,128,64,32,16,8]
    #-----------change_1
        for _lr in [1e-4, 1e-3, 1e-2, 1e-1]: ## [1e-4,1e-3,1e-2,1e-1,1]
            for _struct_code in [0]: ## [0,1]
                model_siamese = SiameseNetwork(
                    len_embedding, abstract_len_embedding, use_irm=False, n_hidden_node=_n_hidden_node,
                    struct_code=_struct_code
                )
                optimizer_siamese = torch.optim.Adam(model_siamese.parameters(), lr=_lr)

                loss_siamese = None
                for epoch in range(num_iterations):
                    error_siamese = 0
#                     for x, y in [(X_train, y_train)]:
                    for x, y in [(x_env, y_env)]:                      
                        p = torch.randperm(len(x))
                        x_e = x[p]
                        y_e = y[p]
                        fp1 = x_e[:, list(range(0, len_embedding, 1))]
                        fp2 = x_e[:, list(range(len_embedding, 2 * len_embedding, 1))]
                        y_pred_siamese = model_siamese(fp1, fp2) 
                        error_e_siamese = model_siamese.loss(y_pred_siamese, y_e)
                        error_siamese += error_e_siamese

                    loss_siamese = 1 * error_siamese 

                    optimizer_siamese.zero_grad() ## clear buffer   
                    loss_siamese.backward() ## calculate gradient for all params
                    optimizer_siamese.step() ## update parameters using calculated gradients

                ## validation
#                 test_fp1 = X_test[:, list(range(0, len_embedding, 1))]
#                 test_fp2 = X_test[:, list(range(len_embedding, 2 * len_embedding, 1))]
#                 test_y_pred_siamese = model_siamese(test_fp1, test_fp2)
#                 test_loss_siamese = model_siamese.loss(test_y_pred_siamese, y_test)
#                 if test_loss_siamese < best_loss_siamese:
#                     best_model_info = (_n_hidden_node, _lr, _struct_code, model_siamese) 
#                     best_loss_siamese = test_loss_siamese
                if loss_siamese < best_loss_siamese:
                    best_model_info = (_n_hidden_node, _lr, _struct_code, model_siamese)
                    best_loss_siamese = loss_siamese
                file1 = open(dir_datasets_prefix + 'logger.log', 'a+')  
                file1.writelines(
                    'trial_no, env_no, n_hidden_node, lr, struct_code, loss_siamese, best_loss_siamese: ' \
                    + str(trial_no) + str(env_no) + str(_n_hidden_node) + str(_lr) + str(_struct_code) \
                    + str(loss_siamese) + str(best_loss_siamese) + '\n\n'
                )
                file1.close()
                model_info = (_n_hidden_node, _lr, _struct_code, model_siamese)
                list_model_info.append(model_info)
             
    return best_model_info, list_model_info        

@ray.remote(num_returns=1)
def get_result(
    st, trial_no, list_env, dir_datasets, should_standardize, 
    len_embedding, num_iterations,  pca_n_percent
):
    for env_no in list_env:
        d_result = {}
        data = genfromtxt(dir_datasets + 'filtered_embedding_vs_energy_env_' + str(env_no) + '.csv', delimiter=',', dtype='float32')
        feature = data[:, 0:-1]
        target = data[:, [-1]]
        if should_standardize:
            scaler = StandardScaler().fit(feature)
            feature = scaler.transform(feature)
            
        #####################################################################################################################################            
        ## model 1: regular
        model_name = 'regular' ## use the embeddings as it is
        feature_regular, target_regular = feature, target
        X_train, X_test, y_train, y_test = train_test_split(
            feature_regular, target_regular, test_size=0.33, random_state=trial_no
        )        
        ##
        trainX, testX, trainY, testY = X_train, X_test, y_train.ravel(), y_test.ravel()
        mlrun = utils.MLPredsByCV(cross_validation_split_no = 5)        
        for alg in ['ridge','lasso','elastic','krr','svr']: ## ['ridge','lasso','elastic','krr','svr','gp']:
            if alg == 'svr':
                errors = mlrun.SVR_CV(trainX, testX, trainY, testY)
            elif alg == 'krr':
                errors = mlrun.KRR_CV(trainX, testX, trainY, testY)
            elif alg == 'ridge':
                errors = mlrun.Ridge_CV(trainX, testX, trainY, testY)
            elif alg == 'lasso':
                errors = mlrun.Lasso_CV(trainX, testX, trainY, testY)
            elif alg == 'elastic':
                errors = mlrun.Elastic_CV(trainX, testX, trainY, testY)
            error = np.mean(errors)
            key = (str(trial_no), str(env_no), str(model_name), str(alg))
            d_result[key] = error
        file1 = open(dir_datasets_prefix + 'logger.log', 'a+')  
        file1.writelines(
            'regular: time, trial_no, env_no, model_name: ' \
            + str(time.time()-st) + '=>  ' + str(trial_no) + ', ' \
            + str(env_no) + ', ' + str(model_name) + '\n\n'
        )
        file1.close()            
        
        #####################################################################################################################################    
        ## model 2: PCA
        model_name = 'PCA' ## use the transformed embeddings by PCA
        pca_dump_name = dir_datasets + 'pca_' + str(env_no) + '.pkl'
        if should_standardize:
            pca_dump_name = dir_datasets + 'pca_std_' + str(env_no) + '.pkl'                    
        #-----------change_1
        pca = PCA(n_components=pca_n_percent) ## e.g. 24 => 6 or other number of components
        feature_train, feature_test, target_train, target_test = train_test_split(
            feature, target, test_size=0.33, random_state=trial_no
        )    
        pca.fit(feature_train)
        ## pickle dump
        pk.dump(pca, open(pca_dump_name,"wb"))
        ## later reload the pickle file
        pca = pk.load(open(pca_dump_name,"rb"))
        #-----------change_1
        pca_n_components = pca.n_components_
        file1 = open(dir_datasets_prefix + 'logger.log', 'a+')  
        file1.writelines('trial_no, env_no, pca_n_components: ' + str(trial_no) + ', ' + str(env_no) + ', ' + str(pca_n_components) + '\n\n')
        file1.close()            

        feature_pca, target_pca = pca.transform(feature), target
        X_train, X_test, y_train, y_test = train_test_split(
            feature_pca, target_pca, test_size=0.33, random_state=trial_no
        )    
        ##
        trainX, testX, trainY, testY = X_train, X_test, y_train.ravel(), y_test.ravel()
        mlrun = utils.MLPredsByCV(cross_validation_split_no = 5)                
        for alg in ['ridge','lasso','elastic','krr','svr']: ## ['ridge','lasso','elastic','krr','svr','gp']:
            if alg == 'svr':
                errors = mlrun.SVR_CV(trainX, testX, trainY, testY)
            elif alg == 'krr':
                errors = mlrun.KRR_CV(trainX, testX, trainY, testY)
            elif alg == 'ridge':
                errors = mlrun.Ridge_CV(trainX, testX, trainY, testY)
            elif alg == 'lasso':
                errors = mlrun.Lasso_CV(trainX, testX, trainY, testY)
            elif alg == 'elastic':
                errors = mlrun.Elastic_CV(trainX, testX, trainY, testY)
            error = np.mean(errors)
            key = (str(trial_no), str(env_no), str(model_name), str(alg))
            d_result[key] = error
        file1 = open(dir_datasets_prefix + 'logger.log', 'a+')  
        file1.writelines(
            'pca: time, trial_no, env_no, model_name: ' \
            + str(time.time()-st) + '=>  ' + str(trial_no) + ', ' \
            + str(env_no) + ', ' + str(model_name) + '\n\n'
        )
        file1.close()            
            
        #####################################################################################################################################    
        ## model 3: Siamese
        pca_dump_name = dir_datasets + 'pca_' + str(env_no) + '.pkl'
        if should_standardize:
            pca_dump_name = dir_datasets + 'pca_std_' + str(env_no) + '.pkl'                    
        pca = pk.load(open(pca_dump_name,"rb"))
        pca_n_components = pca.n_components_
        #-----------------------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            feature, target, test_size=0.33, random_state=trial_no
        )           
        best_model_info, list_model_info = get_best_models(
            len_embedding=len_embedding, abstract_len_embedding=pca_n_components,
            num_iterations=num_iterations, X_train=X_train, y_train=y_train,
            trial_no=trial_no, env_no=env_no
        )
        # for model_info in list_model_info:  ## change_3
        for model_info in [best_model_info]:          
            _nhn, _lr, _sc, model_siamese = model_info[0], model_info[1], model_info[2], model_info[3]  
            
            # model_name = 'Siamese_' + str(_nhn) + '_' + str(_lr) + '_' + str(_sc) ## use the transformed embeddings by Siamese
            model_name = 'Siamese' ## change_3
            feature_siamese, target_siamese = model_siamese.forward_reg(torch.from_numpy(feature)).detach().numpy(), target 
            X_train, X_test, y_train, y_test = train_test_split(
                feature_siamese, target_siamese, test_size=0.33, random_state=trial_no
            )   
            
        #-----------------------------------------------------------------------------------        
        #
            trainX, testX, trainY, testY = X_train, X_test, y_train.ravel(), y_test.ravel()
            mlrun = utils.MLPredsByCV(cross_validation_split_no = 5)                
            for alg in ['ridge','lasso','elastic','krr','svr']: ## ['ridge','lasso','elastic','krr','svr','gp']:
                if alg == 'svr':
                    errors = mlrun.SVR_CV(trainX, testX, trainY, testY)
                elif alg == 'krr':
                    errors = mlrun.KRR_CV(trainX, testX, trainY, testY)
                elif alg == 'ridge':
                    errors = mlrun.Ridge_CV(trainX, testX, trainY, testY)
                elif alg == 'lasso':
                    errors = mlrun.Lasso_CV(trainX, testX, trainY, testY)
                elif alg == 'elastic':
                    errors = mlrun.Elastic_CV(trainX, testX, trainY, testY)
                error = np.mean(errors)
                key = (str(trial_no), str(env_no), str(model_name), str(alg))
                d_result[key] = error
            file1 = open(dir_datasets_prefix + 'logger.log', 'a+')  
            file1.writelines('siamese: time, trial_no, env_no, model_name: ' + str(time.time()-st) + '=>  ' + str(trial_no) + ', ' + str(env_no) + ', ' + str(model_name) + '\n\n')
            file1.close()            

        #####################################################################################################################################     
        ##
        
        with open(dir_datasets_prefix + 'datasets/d_result_' + str(trial_no) + '_' + str(env_no) + '.pickle', 'wb') as handle: 
            pk.dump(d_result, handle, protocol=pk.HIGHEST_PROTOCOL)
    return 0

if __name__=='__main__':
    #-----------change_1
    num_trials = 10 ## should be 10
    list_env = [0,1,2,3] ## should be [0,1,2,3,4,5]
    num_iterations = 5000 ## should be (500 or 1000) for train on BEEF (extrapolation) or (5000/500 or 10000/1000) for train on individual env (intrapolation)
    #-----------change_1
    #
    dir_datasets = dir_datasets_prefix + 'datasets/'
    should_standardize = False
    len_embedding=24 ## change_2
    pca_n_percent=0.99 ## change_2

    ## main_1: get_result

    st = time.time()
    list_result_id = []
    for trial_no in range(num_trials):
        result_id = get_result.remote(
            st, trial_no, list_env, dir_datasets, should_standardize, 
            len_embedding, num_iterations, pca_n_percent
        )
        list_result_id.append(result_id)
    list_result = ray.get(list_result_id)





