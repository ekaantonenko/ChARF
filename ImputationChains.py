import numpy as np
from sklearn.base import clone, BaseEstimator
from scipy import stats

#################################################################################
#################################################################################
#################################################################################


class imputeEnsembleChains(BaseEstimator):
    
    def __init__(self, base_estimator, random_state=None,
                 delta=5, past=3, n=5):
        
        self.base_estimator = base_estimator
        self.random_state = random_state
        
        self.delta = 3*delta
        self.past = past
        
        self.n = n
    
    
    #====================================================================
    #====================================================================
    #====================================================================
    
    def _addNA(self, Data, frac, dupl=1):
        '''
        Manually corrupting complete data with missing values
        '''

        N = Data.shape[0]
        p = Data.shape[1]
        
        if dupl > 1:
            p = int(p/dupl)
            
        mask = np.zeros(N*p, dtype=bool)
        mask[:int(frac*N*p)] = True
        np.random.shuffle(mask)
        mask = mask.reshape(N, p)
        
        if dupl > 1:
            reps = np.ones(mask.shape[1],dtype=int)
            reps *= dupl
            mask = np.repeat(mask,reps,axis=1)

        Data[mask] = -1
        return Data
        
    
    
    def _oneChainWindow(self, mdl, X_sub_na, frac):
        '''
        Prediction in one window 'X_sub_na' of width 'delta' with model 'mdl'
        '''
    
                    
        ### Select rows without m.v. for training data
        X_sub_train = X_sub_na[~np.any(X_sub_na == -1, axis=1), :]  
        
        

        if X_sub_train.shape[0] == X_sub_na.shape[0]:
            return X_sub_na

        ### If training data is empty: replace m.v. with mode for each column
        elif X_sub_train.shape[0] == 0:
            print ('WARNING: empty array!')
            X_sub_pred = X_sub_na.copy()
            for j in range(X_sub_pred.shape[1]):
                m = stats.mode(X_sub_na[X_sub_na[:,j] != -1][:,j])[0][0]
                X_sub_pred[:,j] = np.where((X_sub_pred[:,j] == -1), m, X_sub_pred[:,j])
                
        ### Else: corrupt with m.v. and train "autoencoder"
        else:
            X_sub_na_train = self._addNA(X_sub_train.copy(), frac, dupl=3)            
            X_sub_na_test = X_sub_na[np.any(X_sub_na == -1, axis=1), :]
        

            ### fit & predict                
            mdl.fit(X_sub_na_train, X_sub_train)
            X_sub_pred_test = mdl.predict(X_sub_na_test)
            
            X_sub_pred = X_sub_na.copy()
            rows_with_na = np.any(X_sub_pred == -1, axis=1)
            rows_for_replacement = np.where(X_sub_na[np.any(X_sub_na == -1, axis=1), :] != -1, 
                                       X_sub_na[rows_with_na, :], 
                                       X_sub_pred_test)
            X_sub_pred[rows_with_na,:] = rows_for_replacement
            
        return X_sub_pred

    #====================================================================
    #====================================================================
    #====================================================================         
    
    def impute(self, X_na):
        '''
        Missing value imputation
        '''
        
        N, p = X_na.shape[0], X_na.shape[1]
        num_windows = int(p / self.delta)
        
        
        ### 1 forward chain, 1 backward chain, n-2 random chains
        orders = [list(range(num_windows)),
                  list(range(num_windows-1, -1, -1))]
        orders += [np.random.permutation(num_windows) for _ in range(self.n-2)]
        
        frac = list(X_na.reshape(-1)).count(-1) / X_na.size
        
        predictions = []
        
        for order in orders: 
            
            self.estimators = [clone(self.base_estimator)
                                 for _ in range(len(order))] 
            
            X_pred = (np.ones((N,p)) * (-1)).astype(int)
            
            for i in range(len(order)):
                ancs = order[max(0,i-self.past):i]                    
                w = order[i]
                
                X_ancs = np.ones((N,0)).astype(int)
                for v in ancs:
                    X_ancs = np.hstack((X_ancs, X_pred[:, v*self.delta : (v+1)*self.delta]))
                X_sub_na = X_na[:, w*self.delta : (w+1)*self.delta]                
                X_train_na = np.hstack((X_ancs, X_sub_na))
                
                ### !!! ###
                X_sub_pred = self._oneChainWindow(self.estimators[i], X_train_na, frac)

                X_pred[:,w*self.delta : (w+1)*self.delta] = X_sub_pred[:,-self.delta:]
                
            predictions.append(X_pred)

    
        ### Take average prediction of ensemble
        X_pred_final = np.mean(predictions, axis=0).round(0).astype(int)    
                        
        return X_pred_final



