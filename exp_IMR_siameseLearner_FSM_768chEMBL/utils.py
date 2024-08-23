# !pip install gpy
import GPy
import numpy as np
# from sklearn.gaussian_process import GaussianProcess
# from sklearn.gaussian_process import GaussianProcessRegressor as GP
# from sklearn.gaussian_process import GaussianProcessClassifier as GP
from sklearn.linear_model import Ridge, Lasso
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from pandas import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
import os
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler

## classes ----------------------------------------------------------------------------------------------------------------------

#This class contains methods for different machine learning algorithms
#which takes train and test data and return the prediction errors (absolute error) for test set.
#Each method makes a grid search on the probable hyperparameters settings for the corresponding algorithm.
#Using cross-validation, it determines an optimum hyperparameter setting and then report prediction error
#on the test set using that setting of the hyperparameters.
class MLPredsByCV():
    def __init__(self, cross_validation_split_no = 5):
        self.cv_split_no = cross_validation_split_no
        self.tol=0.01 ## default 0.001 from extrapolation code
        self.max_iter=100000 ## default 1000 for most ML algo library
        
    def SVR_CV(self, trainX, testX, trainY, testY):
        C_vals = [1.0, 10.0, 100.0, 500.0, 1000.0]
        inverse_gamma_vals = [1.0, 10.0, 20.0, 40.0, 80.0, 200.0]
        epsilon_vals = [0.001, 0.01, 0.1]
        cv_errors = np.empty([len(C_vals)*len(inverse_gamma_vals)*len(epsilon_vals), 4])
        i = 0
        for c in C_vals:
            for g in inverse_gamma_vals:
                for e in epsilon_vals:
                    errors = np.empty([self.cv_split_no, 1])
                    kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
                    j = 0
                    for train_indices, validation_indices in kf.split(trainX):
                        training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
                        training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
                        regr = SVR(C=c, gamma=1.0/g, kernel='rbf', epsilon=e, tol=self.tol)
                        regr.fit(training_set_X, training_set_Y)
                        predY = regr.predict(validation_set_X)
                        errorY = np.absolute(predY - validation_set_Y)
                        errors[j] = np.mean(errorY)
                        j = j + 1
                    cv_errors[i,:] = c, g, e, np.mean(errors)
                    i = i + 1
        C_opt, g_opt, eps_opt, _ = cv_errors[np.argmin(cv_errors[:, 3]), :]
        regr = SVR(C=C_opt, gamma=1.0/g_opt, kernel='rbf', epsilon=eps_opt)
        regr.fit(trainX, trainY)
        predY = regr.predict(testX)
        err_on_opt_params = np.absolute(predY - testY)                
        return err_on_opt_params


    def KRR_CV(self, trainX, testX, trainY, testY):
        kernel_vals = ['rbf', 'laplacian']
        kernel_indices = [0,1]
        inverse_gamma_vals = [1.0, 10.0, 20.0, 40.0, 80.0]
        alpha_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
        cv_errors = np.empty([len(kernel_vals)*len(inverse_gamma_vals)*len(alpha_vals), 4])
        i = 0
        for kern in kernel_vals:
            for g in inverse_gamma_vals:
                for a in alpha_vals:
                    errors = np.empty([self.cv_split_no, 1])
                    kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
                    j = 0
                    for train_indices, validation_indices in kf.split(trainX):
                        training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
                        training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
                        regr = KernelRidge(alpha=a, gamma=1.0/g, kernel=kern)
                        regr.fit(training_set_X, training_set_Y)
                        predY = regr.predict(validation_set_X)
                        errorY = np.absolute(predY - validation_set_Y)
                        errors[j] = np.mean(errorY)
                        j = j + 1
                    cv_errors[i,:] = kernel_indices[kernel_vals.index(kern)], g, a, np.mean(errors)
                    i = i + 1
        k_opt, g_opt, a_opt, _ = cv_errors[np.argmin(cv_errors[:, 3]), :]
        k_opt = kernel_vals[kernel_indices.index(k_opt)]
        regr = KernelRidge(alpha=a_opt, gamma=1.0/g_opt, kernel=k_opt)
        regr.fit(trainX, trainY)
        predY = regr.predict(testX)
        err_on_opt_params = np.absolute(predY - testY)                 
        return err_on_opt_params



    def Ridge_CV(self, trainX, testX, trainY, testY):
        alpha_vals = [0.001, 0.01, 0.1, 1.0]
        cv_errors = np.empty([len(alpha_vals), 2])
        i = 0
        for a in alpha_vals:        
            errors = np.empty([self.cv_split_no, 1])
            kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
            j = 0
            for train_indices, validation_indices in kf.split(trainX):
                training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
                training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
                regr = Ridge(alpha=a, tol=self.tol, max_iter=self.max_iter)
                regr.fit(training_set_X, training_set_Y)
                predY = regr.predict(validation_set_X)
                errorY = np.absolute(predY - validation_set_Y)
                errors[j] = np.mean(errorY)
                j = j + 1
            cv_errors[i,:] = a, np.mean(errors)
            i = i + 1
        a_opt, _ = cv_errors[np.argmin(cv_errors[:, 1]), :]
        regr = Ridge(alpha=a_opt)
        regr.fit(trainX, trainY)
        predY = regr.predict(testX)
        err_on_opt_params = np.absolute(predY - testY)                
        return err_on_opt_params


    def Lasso_CV(self, trainX, testX, trainY, testY):
        alpha_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
        cv_errors = np.empty([len(alpha_vals), 2])
        i = 0
        for a in alpha_vals:        
            errors = np.empty([self.cv_split_no, 1])
            kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
            j = 0
            for train_indices, validation_indices in kf.split(trainX):
                training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
                training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
                regr = Lasso(alpha=a, tol=self.tol, max_iter=self.max_iter)
                regr.fit(training_set_X, training_set_Y)
                predY = regr.predict(validation_set_X)
                errorY = np.absolute(predY - validation_set_Y)
                errors[j] = np.mean(errorY)
                j = j + 1
            cv_errors[i,:] = a, np.mean(errors)
            i = i + 1
        a_opt, _ = cv_errors[np.argmin(cv_errors[:, 1]), :]
        regr = Lasso(alpha=a_opt)
        regr.fit(trainX, trainY)
        predY = regr.predict(testX)
        err_on_opt_params = np.absolute(predY - testY)                 
        return err_on_opt_params


    def Elastic_CV(self, trainX, testX, trainY, testY):
        alpha_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
        l1ratio_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cv_errors = np.empty([len(alpha_vals)*len(l1ratio_vals), 3])
        i = 0
        for a in alpha_vals:    
            for l in l1ratio_vals:
                errors = np.empty([self.cv_split_no, 1])
                kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
                j = 0
                for train_indices, validation_indices in kf.split(trainX):
                    training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
                    training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
                    regr = linear_model.ElasticNet(alpha=a, l1_ratio=l, tol=self.tol, max_iter=self.max_iter)
                    regr.fit(training_set_X, training_set_Y)
                    predY = regr.predict(validation_set_X)
                    errorY = np.absolute(predY - validation_set_Y)
                    errors[j] = np.mean(errorY)
                    j = j + 1
                cv_errors[i,:] = a, l, np.mean(errors)
                i = i + 1
        a_opt, l_opt, _ = cv_errors[np.argmin(cv_errors[:, 2]), :]
        regr = linear_model.ElasticNet(alpha=a_opt, l1_ratio=l_opt)
        regr.fit(trainX, trainY)
        predY = regr.predict(testX)
        err_on_opt_params = np.absolute(predY - testY)                 
        return err_on_opt_params


    def Run_GPy(self, trainX, testX, trainY, testY, kern, ard, vr, ls, nv):
        inputdim = trainX.shape[1]
        kernel = None
        m = None
        if kern == 'rbf':
            kernel = GPy.kern.RBF(input_dim=inputdim, variance=vr, lengthscale=ls, ARD=ard)
        elif kern == 'laplacian':
            kernel = GPy.kern.Exponential(input_dim=inputdim, variance=vr, lengthscale=ls, ARD=ard)
        X = trainX
        Y = np.transpose(np.array([trainY]))
        m = GPy.models.GPRegression(X, Y , kernel, noise_var=nv)
        #m.optimize()
        predY,mseY = m.predict(testX, full_cov=False)
        predY = np.transpose(predY)[0,:]
        mseY = np.transpose(mseY)[0,:]
        errorY = np.absolute(predY - testY)
        return errorY, np.sqrt(mseY)
        

    def GP_CV(self, trainX, testX, trainY, testY):
        kernel_vals = ['rbf', 'laplacian']
        kernel_indices = [0,1]
        ard_vals = [True, False]
        ard_indices = [0,1]
        var_vals = [1.0, 10.0, 50.0, 100.0, 250.0]
        lscale_vals = [1.0, 10.0, 40.0, 80.0, 200.0]
        noise_vals = [0.0001, 0.001, 0.01, 0.1]
        cv_errors = np.empty([len(kernel_vals)*len(ard_vals)*len(var_vals)*len(lscale_vals)*len(noise_vals), 6])
        i = 0
        for kern in kernel_vals:
            for ard in ard_vals: 
                for vr in var_vals:
                    for ls in lscale_vals:
                        for nv in noise_vals:
                            #print(i)
                            try:
                                errors = np.empty([self.cv_split_no, 1])
                                kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
                                j = 0
                                for train_indices, validation_indices in kf.split(trainX):
                                    training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
                                    training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
                                    errvals, _ = self.Run_GPy(training_set_X, validation_set_X, training_set_Y, validation_set_Y, 
                                                        kern, ard, vr, ls, nv)
                                    errors[j] = np.mean(errvals)
                                    j = j + 1
                                cv_errors[i,:] = kernel_indices[kernel_vals.index(kern)], ard_indices[ard_vals.index(ard)]\
                                                    , vr, ls, nv ,np.mean(errors)
                                i = i + 1
                            except Exception as e:
                                print(str(e))
                                print('error occured for ', kern, ard, vr, ls, nv)
                                raise
        k_opt, a_opt, v_opt, l_opt, n_opt, _ = cv_errors[np.argmin(cv_errors[:, 5]), :]
        k_opt = kernel_vals[kernel_indices.index(k_opt)]
        a_opt = ard_vals[ard_indices.index(a_opt)]
        err_on_opt_params, std_mean_on_opt_params = self.Run_GPy(trainX, testX, trainY, testY, k_opt, a_opt, v_opt, l_opt, n_opt)
        return err_on_opt_params, std_mean_on_opt_params

    
    
## functions ---------------------------------------------------------------    

def readSmilesToFingerprints(smstr):
    l = len(smstr)
    i = 0
    prevatom = None
    prevnegval = None
    atom = None
    negval = 0
    nxt = None
    doubleBond = False
    tripleBond = False
    vec = np.zeros(24)
    bondmatrix = np.zeros((6,6)) # matrix for pairwise bonds among O0,O1,C0,C1,C2,C3
    o_zero_pos_at_vec = 3
    c_zero_pos_at_vec = 5
    while i < l:
        prevatom = atom
        prevnegval = negval
        negval = 0
        atom = None
        leftBranch = False
        rightBranch = False
        elemToRight = False
        if i >= 4 and smstr[i-4:i] == '(O=)':
            leftBranch = True
            vec[o_zero_pos_at_vec] += 1
            vec[2] += 1 #number of O
        if smstr[i] == '[':
            atom = smstr[i+1]
            negval = int(smstr[i+3])
            i += 4
        elif smstr[i] == 'C' or smstr[i] == 'O':
            atom = smstr[i]
            negval = 0
        if i < (l-4) and smstr[i+1:i+5] == '(=O)':
            rightBranch = True
            vec[o_zero_pos_at_vec] += 1
            vec[2] += 1 #number of O
            i += 4
        if i < (l-1) and (smstr[i+1] == '[' or smstr[i+1] == 'C' or smstr[i+1] == 'O'):
            elemToRight = True
            
        rem_positions = 0
        if atom == 'C':
            rem_positions = 4
            vec[1] += 1 #number of C
            vec[c_zero_pos_at_vec + negval] += 1 #number of C0 or C1 or C2 or C3; starting at vector index 5
        elif atom == 'O':
            rem_positions = 2
            vec[2] += 1 #number of O
            vec[o_zero_pos_at_vec + negval] += 1 #number of O0 or O1; starting at vector index 3
        rem_positions -= negval
        if leftBranch:
            rem_positions -= 2
        if rightBranch:
            rem_positions -= 2        
        if elemToRight:
            rem_positions -= 1
        if prevatom is not None:
            rem_positions -= 1
        
        curr = str(atom) + str(negval)
        vec[0] += rem_positions #number of H
        if atom == 'C':
            vec[9] += rem_positions #number of C-H
            if leftBranch:
                vec[12] += 1 #number of C=O
            if rightBranch:
                vec[12] += 1 #number of C=O
        if atom == 'O':
            vec[13] += rem_positions #number of O-H
        
        if prevatom is not None:
            rowstartidx = 0
            colstartidx = 0
            if prevatom == 'C':
                rowstartidx = 2
            if atom == 'C':
                colstartidx = 2
            bondmatrix[rowstartidx + prevnegval, colstartidx + negval] += 1
        i += 1
        
    for j in range(3,6):
        for i in range(2,j):
            bondmatrix[i,j] += bondmatrix[j,i]
    vec[14:] = (bondmatrix[2:,2:])[np.triu_indices(4)]
    vec[10] += np.sum(bondmatrix[2:,0]) + np.sum(bondmatrix[0,2:])
    vec[11] += np.sum(bondmatrix[2:,1]) + np.sum(bondmatrix[1,2:])
    return vec
        
    
def isElementNextToCurr(smstr, curridx, seekright = 1, norecurse = 0):
    if seekright == 1:
        if curridx == len(smstr)-1:
            return True # because it is a ring
        if smstr[curridx+1] == 'C' or smstr[curridx+1] == 'O':
            return True
        if smstr[curridx+1] == '[':
            return True
        if smstr[curridx+1] == '(':
            idx_plus = 3
            if smstr[curridx+2] == '=' or smstr[curridx+3] == '=':
                idx_plus += 1
            return isElementNextToCurr(smstr, curridx+idx_plus, seekright, 1)
    else: # seek left
        if curridx == 0:
            return True # because it is a ring
        if smstr[curridx-1] == 'C' or smstr[curridx-1] == 'O':
            return True
        if smstr[curridx-1] == ']':
            return True
        if smstr[curridx-1] == ')':
            idx_minus = 3
            if smstr[curridx-2] == '=' or smstr[curridx-3] == '=':
                idx_minus += 1
            return isElementNextToCurr(smstr, curridx-idx_minus, seekright, 1)
        
        
        
def BranchBondCountNextToCurr(smstr, curridx, seekright = 1):
    if seekright == 1:
        if curridx == len(smstr)-1:
            return 0
        if smstr[curridx+1] == '(':
            if (smstr[curridx+2] == 'C' or smstr[curridx+2] == 'O') and smstr[curridx+3] == ')': #(O)
                return 1
            if smstr[curridx+2] == '=' and (smstr[curridx+3] == 'C' or smstr[curridx+3] == 'O'): #(=O)
                return 2
            if smstr[curridx+2] == '[' and smstr[curridx+3]=='O' and smstr[curridx+4]==']' and smstr[curridx+5]==')': #([O])
                return 1
        return 0
    else: # seek left
        if curridx == 0:
            return 0
        if smstr[curridx-1] == ')':
            if (smstr[curridx-2] == 'C' or smstr[curridx-2] == 'O') and smstr[curridx-3] == '(':
                return 0
            if smstr[curridx-2] == '=' and (smstr[curridx-3] == 'C' or smstr[curridx-3] == 'O'):
                return 2
        return 0

    
def convertNewSmilesToOldSmiles(smstr):
    smstr = smstr.replace('1','')
    convStr = ''
    i = 0
    while i < len(smstr):
        if smstr[i] != '[':
            if smstr[i] == '(' and i < len(smstr)-2 and smstr[i:i+3] == '(H)':
                i += 3
            else:
                convStr += smstr[i]
                i += 1
        else:
            numH = 0
            idxPlus = 2
            if smstr[i+2] == 'H':
                if smstr[i+3] == ']':
                    numH = 1
                    idxPlus += 1
                else:
                    numH = int(smstr[i+3])
                    idxPlus += 2
            if i < len(smstr)-5 and smstr[i+3:i+6] == '(H)':
                numH = 1
                idxPlus += 3
                    
            usedValence = numH
            if isElementNextToCurr(smstr, i, 0):
                usedValence += 1
            usedValence += BranchBondCountNextToCurr(smstr, i, 0)            
            if isElementNextToCurr(smstr, i + idxPlus, 1):
                usedValence += 1
            usedValence += BranchBondCountNextToCurr(smstr, i + idxPlus, 1)
            
            freeval = 0
            if smstr[i+1] == 'C':
                freeval = 4 - usedValence
                convStr += '[C-' + str(freeval) + ']'
            else:
                #freeval = 2 - usedValence
                #convStr += '[O-' + str(freeval) + ']'
                convStr += '[O-1]'
            i = i + idxPlus + 1
    return convStr


