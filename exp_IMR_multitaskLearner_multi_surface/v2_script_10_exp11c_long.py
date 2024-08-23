if __name__=='__main__':
    
    ## 1. if name == main
    ## 2. curdir
    ## 3. # iterations

    ## created from (exp9e => exp11a => exp11b)
    ## exp_name = 'exp11c'
    ## 1e-4 x 10000
    ## 1024 => 512 => 256

    ## install
    # !pip install xgboost

    ## import
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.nn import DataParallel
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from torch.utils.data import TensorDataset, DataLoader
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    from itertools import cycle
    import torch.autograd as autograd
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    ## variables
    seed = 42
    ratio_test = 0.2
    exp_name = 'exp_11c' ## change
    np.random.seed(seed)
    torch.manual_seed(seed)
    curdir = '/curdir/' ## ''

    ## class
    class InvariantModel(nn.Module):
        def __init__(self, len_embedding, abstract_len_embedding):
            super(InvariantModel, self).__init__()
            self.loss = nn.L1Loss(reduction='mean')
            self.len_embedding = len_embedding
            self.abstract_len_embedding = abstract_len_embedding
            self.encoder = nn.Sequential(
                nn.Linear(self.len_embedding, int(self.len_embedding*1)),
                nn.ReLU(),
                nn.BatchNorm1d(int(self.len_embedding*1)),

                nn.Linear(int(self.len_embedding*1), self.abstract_len_embedding),
            )
            self.head1 = nn.Linear(self.abstract_len_embedding, 1)
            self.head2 = nn.Linear(self.abstract_len_embedding, 1)

        def forward(self, x, dataset_id):
            x = self.encoder(x)
            if dataset_id == 1:
                return self.head1(x)
            else:
                return self.head2(x)

    ## function

    def penalty(logits, y, device):
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss = nn.L1Loss(reduction='mean')(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def pc_grad_update(gradients):
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                gi = gradients[i]
                gj = gradients[j]
                conflict = torch.dot(gi, gj) < 0
                if conflict:
                    # Project gi onto the normal plane of gj and vice versa
                    gj_norm_sq = torch.dot(gj, gj)
                    gi_projected = gi - torch.dot(gi, gj) / gj_norm_sq * gj
                    gradients[i] = gi_projected

        return torch.mean(torch.stack(gradients), dim=0)

    def get_model_invariant(train_loader1, train_loader2, len_embedding, abstract_len_embedding):
        print(f'len_embedding: {len_embedding}, abstract_len_embedding: {abstract_len_embedding}')
        _lr, num_iterations = 1e-4, 10000
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        model_invariant = InvariantModel(len_embedding, abstract_len_embedding)
        if torch.cuda.is_available():
            model_invariant = DataParallel(model_invariant)
        model_invariant.to(device)
        optimizer_invariant = torch.optim.Adam(model_invariant.parameters(), lr=_lr)

        model_invariant.train()
        for epoch in range(num_iterations):
            total_loss = 0.0
            train_loader1_cycle = cycle(train_loader1)
            train_loader2_cycle = cycle(train_loader2)
            max_batches = max(len(train_loader1), len(train_loader2))
            for i in range(max_batches):                        

                gradients = []
                
                optimizer_invariant.zero_grad()
                data1 = next(train_loader1_cycle)
                inputs1, labels1 = data1[0].to(device), data1[1].to(device)
                outputs1 = model_invariant(inputs1, dataset_id=1).squeeze()            
                loss1 = model_invariant.module.loss(outputs1, labels1) if isinstance(model_invariant, DataParallel) else model_invariant.loss(outputs1, labels1)
                loss1.backward()
                grads = [param.grad.clone() for param in model_invariant.parameters() if param.grad is not None]
                gradients.append(parameters_to_vector(grads))
                total_loss += loss1.item()

                optimizer_invariant.zero_grad()                
                data2 = next(train_loader2_cycle)
                inputs2, labels2 = data2[0].to(device), data2[1].to(device)
                outputs2 = model_invariant(inputs2, dataset_id=2).squeeze()            
                loss2 = model_invariant.module.loss(outputs2, labels2) if isinstance(model_invariant, DataParallel) else model_invariant.loss(outputs2, labels2)
                loss2.backward()
                grads = [param.grad.clone() for param in model_invariant.parameters() if param.grad is not None]
                gradients.append(parameters_to_vector(grads))
                total_loss += loss2.item()

                optimizer_invariant.zero_grad()            
                # combined_grad1 = torch.mean(torch.stack(gradients), dim=0)
                combined_grad2 = pc_grad_update(gradients)
                combined_grad = combined_grad2
                start_index = 0
                for name, param in model_invariant.named_parameters():
                    param_numel = param.numel()
                    param.grad = combined_grad[start_index : start_index + param_numel].view_as(param)
                    start_index += param_numel                
                optimizer_invariant.step()

            if epoch%1==0:
                with open(curdir + 'v2_script_10/logger_' + exp_name + '.log', 'a+') as file1:
                    file1.writelines(f'epoch: {epoch}, total_loss: {total_loss:.6f}\n\n')


        return model_invariant.module

    ## cathub, imr, (MAE, r2)
    ## ocp, imr, (MAE, r2)
    df1 = pd.read_pickle(f'{curdir}datasets/df_cathub_dpp_combined.pickle')
    X1 = df1.iloc[:, :-1].values
    y1 = df1['nre'].values
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=ratio_test, random_state=seed)
    df2 = pd.read_pickle(f'{curdir}datasets/df_ocp_dpp_combined.pickle')
    X2 = df2.iloc[:, :-1].values
    y2 = df2['energy'].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=ratio_test, random_state=seed)
    print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
    print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)

    train_data1 = TensorDataset(torch.tensor(X_train1, dtype=torch.float32), torch.tensor(y_train1, dtype=torch.float32))
    train_data2 = TensorDataset(torch.tensor(X_train2, dtype=torch.float32), torch.tensor(y_train2, dtype=torch.float32))

    batch_size = 10000
    train_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True)

    len(train_loader1), len(train_loader2)

    len_embedding = 1024
    abstract_len_embedding = int(len_embedding*1)
    model_path = curdir + 'v2_script_10/model_invariant_' + exp_name + '.pt'
    model_invariant = get_model_invariant(
        train_loader1, train_loader2, 
        len_embedding, abstract_len_embedding
    )
    torch.save(model_invariant.state_dict(), model_path)

    ## evaluation

    len_embedding = 1024
    abstract_len_embedding = int(len_embedding*1)
    model_path = curdir + 'v2_script_10/model_invariant_' + exp_name + '.pt'
    model_invariant = InvariantModel(len_embedding, abstract_len_embedding)
    model_invariant.load_state_dict(torch.load(model_path))
    model_invariant.eval()

    X_train_invariant1 = model_invariant.encoder(torch.tensor(X_train1, dtype=torch.float32)).detach().numpy()
    X_test_invariant1 = model_invariant.encoder(torch.tensor(X_test1, dtype=torch.float32)).detach().numpy()

    lr = 0.2
    depth = 8
    n_est = 500
    model1 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
    model1.fit(X_train_invariant1, y_train1)

    y_pred1 = model1.predict(X_test_invariant1)
    mae1 = mean_absolute_error(y_test1, y_pred1)
    r2score1 = r2_score(y_test1, y_pred1)

    plt.figure(figsize=(4, 3))
    plt.scatter(y_test1, y_pred1, alpha=0.5)
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title(f'(cathub, invariant)')
    plt.plot([y_test1.min(), y_test1.max()], [y_test1.min(), y_test1.max()], 'k--', lw=2)
    plt.show()
    print()
    print(f'X_train shape: {X_train1.shape}, X_test shape: {X_test1.shape}')
    print(f'y_train shape: {y_train1.shape}, y_test shape: {y_test1.shape}')
    print(f'train to test ratio: {1-ratio_test}:{ratio_test}')
    print(f'Mean Abs. Error: {mae1:.2f}')
    print(f'R2-score: {r2score1:.2f}')

    X_train_invariant2 = model_invariant.encoder(torch.tensor(X_train2, dtype=torch.float32)).detach().numpy()
    X_test_invariant2 = model_invariant.encoder(torch.tensor(X_test2, dtype=torch.float32)).detach().numpy()

    lr = 0.2
    depth = 8
    n_est = 500
    model2 = xgb.XGBRegressor(learning_rate=lr, max_depth=depth, n_estimators=n_est)
    model2.fit(X_train_invariant2, y_train2)

    y_pred2 = model2.predict(X_test_invariant2)
    mae2 = mean_absolute_error(y_test2, y_pred2)
    r2score2 = r2_score(y_test2, y_pred2)

    plt.figure(figsize=(4, 3))
    plt.scatter(y_test2, y_pred2, alpha=0.5)
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title(f'(cathub, invariant)')
    plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'k--', lw=2)
    plt.show()
    print()
    print(f'X_train shape: {X_train2.shape}, X_test shape: {X_test2.shape}')
    print(f'y_train shape: {y_train2.shape}, y_test shape: {y_test2.shape}')
    print(f'train to test ratio: {1-ratio_test}:{ratio_test}')
    print(f'Mean Abs. Error: {mae2:.2f}')
    print(f'R2-score: {r2score2:.2f}')

