if __name__=='__main__':
    ## 1. extract all slabs from ocp and cathub(mamun)
    ## 2. generate slab descriptor with dimenet++
    ## 3. extract all product from ocp and cathub(mamun)
    ## 4. generate product descriptor with chEMBL
    ## 5. run experiments-
    ##    - multitask learner -> (cathub, ocp) x (xgboost) x (original 1024+1024, pca ncomponents, imr ncomponents)
    ##    - solves n^2 to n, solves descriptor generation for slab/surface

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
    from torch.utils.data import DataLoader, TensorDataset
    from itertools import cycle
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA

    ## variables
    seed = 45
    ratio_test = 0.25
    exp_name = 'v4s6_model1D'
    np.random.seed(seed)
    torch.manual_seed(seed)
    ntrial = 2 ## change this to 10
    ncomp_perc = 0.9949

    ## class
    class InvariantModel(nn.Module):
        def __init__(self, len_embedding, abstract_len_embedding):
            super(InvariantModel, self).__init__()
            self.loss = nn.L1Loss(reduction="mean") 
            self.len_embedding = len_embedding
            self.abstract_len_embedding = abstract_len_embedding  
            self.encoder = nn.Sequential(
                ## 1536 to ~1125
                nn.Linear(self.len_embedding, int(self.abstract_len_embedding*3)),
                nn.ReLU(),
                nn.BatchNorm1d(int(self.abstract_len_embedding*3)), 
                ## ~1125 to ~750
                nn.Linear(int(self.abstract_len_embedding*3), int(self.abstract_len_embedding*2)),
                nn.ReLU(),
                nn.BatchNorm1d(int(self.abstract_len_embedding*2)),
                ## ~750 to ~375
                nn.Linear(int(self.abstract_len_embedding*2), self.abstract_len_embedding),
            )        
            self.head1 = nn.Linear(self.abstract_len_embedding, 1)  # For dataset 1
            self.head2 = nn.Linear(self.abstract_len_embedding, 1)  # For dataset 2

        def forward(self, x, dataset_id):
            x = self.encoder(x)
            if dataset_id == 1:
                return self.head1(x)
            else:
                return self.head2(x)

    ## function

    def get_model_invariant(train_loader1, train_loader2, len_embedding, abstract_len_embedding, trial_no):    
        print(f'len_embedding: {len_embedding}, abstract_len_embedding: {abstract_len_embedding}')

        _lr, num_iterations = 1e-5, 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_invariant = InvariantModel(len_embedding, abstract_len_embedding)
        if torch.cuda.is_available():
            model_invariant = DataParallel(model_invariant)
        model_invariant.to(device)
        optimizer_invariant = torch.optim.Adam(model_invariant.parameters(), lr=_lr)

        model_invariant.train()
        ######################################################################################
        for epoch in range(num_iterations):
            total_loss = 0.0
            
            # Loop through train_loader1 and calculate total loss1
            for data1 in train_loader1:
                inputs1, labels1 = data1[0].to(device), data1[1].to(device)
                outputs1 = model_invariant(inputs1, dataset_id=1).squeeze()
                loss1 = model_invariant.module.loss(outputs1, labels1) if isinstance(model_invariant, DataParallel) else model_invariant.loss(outputs1, labels1)
                
                optimizer_invariant.zero_grad()
                loss1.backward()
                optimizer_invariant.step()                            
                total_loss += loss1.item()

            # Loop through train_loader2 and calculate total loss2
            for data2 in train_loader2:
                inputs2, labels2 = data2[0].to(device), data2[1].to(device)
                outputs2 = model_invariant(inputs2, dataset_id=2).squeeze()
                loss2 = model_invariant.module.loss(outputs2, labels2) if isinstance(model_invariant, DataParallel) else model_invariant.loss(outputs2, labels2)
                optimizer_invariant.zero_grad()
                loss2.backward()
                optimizer_invariant.step()                                            
                total_loss += loss2.item()            
        ######################################################################################


            if epoch % 1 == 0:
                with open('/curdir/v4/logger_' + exp_name + '_' + str(trial_no) + '.log', 'a+') as file1:
                    file1.writelines(f'epoch: {epoch}, total_loss: {total_loss:.6f}\n\n')

        return model_invariant

    df1 = pd.read_pickle(f'/curdir/v4/cathub_df.pickle')
    X1 = df1.iloc[:, :-1].values
    y1 = df1['energy'].values
    df2 = pd.read_pickle(f'/curdir/v4/ocp_df.pickle')
    X2 = df2.iloc[:, :-1].values
    y2 = df2['energy'].values

    for trial_no in range(ntrial):
        print()
        print(f'trial no: {trial_no}')

        #####################################################################################################
        ## original
        #####################################################################################################   
        print('-- original --')
        ## cathub     
        ## data
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=ratio_test, random_state=seed+(trial_no*10))
        # print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
        ## model train
        lr, depth, n_est = 0.2, 8, 500
        # model1 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
        model1 = LinearRegression()
        model1.fit(X_train1, y_train1)
        ## model test
        y_pred1 = model1.predict(X_test1)
        mae1 = mean_absolute_error(y_test1, y_pred1)
        r2score1 = r2_score(y_test1, y_pred1)
        ## evaluation
        print(f'Mean Abs. Error: {mae1:.6f}')
        print(f'R2-score: {r2score1:.6f}')
        ## ocp 
        ## data
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=ratio_test, random_state=seed+(trial_no*10))
        # print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)    
        ## model train
        lr, depth, n_est = 0.2, 8, 500
        # model2 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
        model2 = LinearRegression()
        model2.fit(X_train2, y_train2)
        ## model test
        y_pred2 = model2.predict(X_test2)
        mae2 = mean_absolute_error(y_test2, y_pred2)
        r2score2 = r2_score(y_test2, y_pred2)
        ## evaluation
        print(f'Mean Abs. Error: {mae2:.6f}')
        print(f'R2-score: {r2score2:.6f}')

        #####################################################################################################
        ## pca
        #####################################################################################################    
        print('-- pca --')    
        ## cathub     
        ## data
        pca1 = PCA(n_components=ncomp_perc)  # Keep 95% of variance
        X_train1_pca = pca1.fit_transform(X_train1)
        X_test1_pca = pca1.transform(X_test1)    
        ncomp1 = pca1.n_components_
        ## model train
        lr, depth, n_est = 0.2, 8, 500
        # model1 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
        model1 = LinearRegression()
        model1.fit(X_train1_pca, y_train1)
        ## model test
        y_pred1 = model1.predict(X_test1_pca)
        mae1 = mean_absolute_error(y_test1, y_pred1)
        r2score1 = r2_score(y_test1, y_pred1)
        ## evaluation
        print(f'ncomp1: {ncomp1}')
        print(f'Mean Abs. Error: {mae1:.6f}')
        print(f'R2-score: {r2score1:.6f}')
        ## ocp 
        ## data
        pca2 = PCA(n_components=ncomp_perc)  # Keep 95% of variance
        X_train2_pca = pca2.fit_transform(X_train2)
        X_test2_pca = pca2.transform(X_test2)    
        ncomp2 = pca2.n_components_    
        ## model train
        lr, depth, n_est = 0.2, 8, 500
        # model2 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
        model2 = LinearRegression()
        model2.fit(X_train2_pca, y_train2)
        ## model test
        y_pred2 = model2.predict(X_test2_pca)
        mae2 = mean_absolute_error(y_test2, y_pred2)
        r2score2 = r2_score(y_test2, y_pred2)
        ## evaluation
        print(f'ncomp2: {ncomp2}')
        print(f'Mean Abs. Error: {mae2:.6f}')
        print(f'R2-score: {r2score2:.6f}')

        #####################################################################################################
        ## invariant
        #####################################################################################################    
        print('-- invariant --')    

        #############################################################
        # Convert to PyTorch Tensors
        train_data1 = TensorDataset(torch.tensor(X_train1, dtype=torch.float32), torch.tensor(y_train1, dtype=torch.float32))
        train_data2 = TensorDataset(torch.tensor(X_train2, dtype=torch.float32), torch.tensor(y_train2, dtype=torch.float32))
        # Create Data Loaders
        batch_size = 10000  # Set your batch size
        train_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True)
        train_loader2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True)    
        ## build a InvariantModel to transform 'd' to 'pca(d)',
        ## use that 'InvariantModel transformation' instead of pca
        len_embedding, abstract_len_embedding = 1536, int((ncomp1+ncomp2)/2)    
        model_invariant = get_model_invariant(
            train_loader1, train_loader2, len_embedding, abstract_len_embedding, trial_no    
        )
        ## save the model
        model_path = '/curdir/v4/model_invariant_' + str(exp_name) + '_' + str(trial_no) + '.pt'
        torch.save(model_invariant.state_dict(), model_path)    
        # load the model
        model_invariant = DataParallel(InvariantModel(len_embedding, abstract_len_embedding))
        model_invariant.load_state_dict(torch.load(model_path))
        model_invariant = model_invariant.module
        # model_invariant.eval()    
        #############################################################
        ## cathub     
        ## data
        X_train1_inv = model_invariant.encoder(torch.from_numpy(X_train1).float()).detach().numpy()
        X_test1_inv = model_invariant.encoder(torch.from_numpy(X_test1).float()).detach().numpy()
        ## model train
        lr, depth, n_est = 0.2, 8, 500
        # model1 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
        model1 = LinearRegression()
        model1.fit(X_train1_inv, y_train1)
        ## model test
        y_pred1 = model1.predict(X_test1_inv)
        mae1 = mean_absolute_error(y_test1, y_pred1)
        r2score1 = r2_score(y_test1, y_pred1)
        ## evaluation
        print(f'Mean Abs. Error: {mae1:.6f}')
        print(f'R2-score: {r2score1:.6f}')
        ## ocp 
        ## data
        X_train2_inv = model_invariant.encoder(torch.from_numpy(X_train2).float()).detach().numpy()
        X_test2_inv = model_invariant.encoder(torch.from_numpy(X_test2).float()).detach().numpy()
        ## model train
        lr, depth, n_est = 0.2, 8, 500
        # model2 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
        model2 = LinearRegression()
        model2.fit(X_train2_inv, y_train2)
        ## model test
        y_pred2 = model2.predict(X_test2_inv)
        mae2 = mean_absolute_error(y_test2, y_pred2)
        r2score2 = r2_score(y_test2, y_pred2)
        ## evaluation
        print(f'Mean Abs. Error: {mae2:.6f}')
        print(f'R2-score: {r2score2:.6f}')