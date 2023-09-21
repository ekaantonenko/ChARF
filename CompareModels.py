### Basics
import pandas as pd
import numpy as np
import copy
import time
from sklearn.base import clone, BaseEstimator
import random

from scipy import stats

from sklearn import ensemble
    
### Metrics
from sklearn.metrics import *


#######################
### Local Packages: ###
#######################
import ImputationChains as IC

   

def compareModels (Data, Data_na, 
                   gridsearch = False,
                   save_imputations = False,
                   onehot=True):   
    
    ##################################################
    ############### Useful functions: ################
    ##################################################
    
    def reverse_dummy(Data_enc, nb_classes=3):
        '''
        Reverse One-Hot encoded data after imputation
        '''
        tmp = Data_enc.reshape(Data_enc.shape[0] * int(Data_enc.shape[1]/nb_classes),
                               nb_classes)
        Data_dec = np.argmax(tmp, axis=1).reshape(Data_enc.shape[0],
                            int(Data_enc.shape[1]/nb_classes))
        return Data_dec
    
    def accuracy_individual(A, B):
        '''
        Counts proportion of matching values between A and B
        '''
        numerator = A[(A==B)].size
        denominator = A.size
        acc_indiv = numerator / denominator
        return acc_indiv
        
    
    def OneHotEncoding(Data, nb_classes=3):
        '''
        One-Hot Encoding for SNP data
        '''
        targets = np.array(Data).reshape(-1).astype(int)
        return np.eye(nb_classes)[targets].reshape([Data.shape[0],nb_classes*Data.shape[1]])
    
    
    
    def OneHotEncoding_missing(Data, Data_na, nb_classes=3):
        '''
        One-Hot Encoding for SNP data with missing values
        (puts -1 for all columns corresponding to one feature)
        '''
        targets = np.array(Data).reshape(-1).astype(int)
        missing_encoded = np.eye(nb_classes)[targets].reshape([Data.shape[0],nb_classes*Data.shape[1]])
        
        idx = np.where(Data_na == -1)
        for i in range(nb_classes):
            missing_encoded[idx[0], idx[1]*nb_classes + i] = -1
        return missing_encoded
    
    
    ##################################################
    ##################################################
    ##################################################
    
    if onehot == True:
        Data_na_oh = OneHotEncoding_missing(Data.copy(), Data_na, nb_classes=3)
        Data_oh = OneHotEncoding(Data.copy(), nb_classes=3)
        print('one-hot encoded')
    else:
        Data_na_oh, Data_oh = Data_na, Data
    
    
    ##################################################                                                                      
    ################# Models: ########################   
    ##################################################
    
    
    ### number of chains in ensemble
    n_range = [5]

    ### imputing models ###
    allModels = [ensemble.RandomForestClassifier(n_estimators=10)
                ]
    
    
        
    ##################################################                                                                      
    ############### Prediction: ######################   
    ##################################################  
    
    logi = []

    ###### Naive approach: MODE ######
    def imputeMODE():
        '''
        imputation with 'mode' for comparison
        '''
        
        name = 'Mode'
        print(name)
    
        X_pred = Data_na.copy()
        for j in range(X_pred.shape[1]):
            m = stats.mode(Data_na[Data_na[:,j] != -1][:,j])[0]
            X_pred[:,j] = np.where((X_pred[:,j] == -1), m, X_pred[:,j])
        
        acc_indiv = accuracy_individual(Data, X_pred)
        correctly_imputed = accuracy_individual(Data[Data_na == -1], X_pred[Data_na == -1])
        changed_oop = 1 - accuracy_individual(Data[Data_na != -1], X_pred[Data_na != -1])
        
        if save_imputations == True:
            X_pred = pd.DataFrame(X_pred)
            X_pred.to_csv('./Results/X_mode.csv')
    
        logi.append([name,
                     acc_indiv,
                     correctly_imputed,
                     changed_oop
                     ])
        print()
    
    imputeMODE()
    
    
    
    #########################
    ###### Test models ######
    #########################
    
    if gridsearch == True:
        deltas = [1,3,5,8,10]
        nus = [0,1,3,5]
    else:
        deltas = [5]
        nus = [10]
        
        
    for mdl in allModels: 
        for delta in deltas:
            for nu in nus:
                name = mdl.__class__.__name__
                
                
                name += '_delta=' + str(delta) + '_past=' + str(nu)
                print(name)  
                
                
                model = IC.imputeEnsembleChains(mdl, delta=delta, past=nu)
                
                X_pred_oh = model.impute(Data_na_oh)
                    
                if onehot == True:
                    X_pred = reverse_dummy(X_pred_oh)
                else:
                    X_pred = X_pred_oh
                    
                
             
                acc_indiv = accuracy_individual(Data, X_pred)
                correctly_imputed = accuracy_individual(Data[Data_na == -1], X_pred[Data_na == -1])
                changed_oop = 1 - accuracy_individual(Data[Data_na != -1], X_pred[Data_na != -1])
                #print(correctly_imputed)
                #print(changed_oop)
                
                if save_imputations == True:
                    X_pred = pd.DataFrame(X_pred)
                    X_pred.to_csv('./Results/X_pred_' + str(delta) + '_' + str(nu) + '.csv')
                
                logi.append([name,
                             acc_indiv,
                             correctly_imputed,
                             changed_oop
                             ])
                print()
        
        
           
      
    ################################################## 
    ################################################## 
    
    logi = pd.DataFrame(logi)       
    logi.columns = ['Classifier', 
                    'acc_indiv',
                    'correctly_imputed',
                    'changed_oop'
                   ]
    
    
    logi = logi.set_index('Classifier')
    
    
    
    return logi#, ys






