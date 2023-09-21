# ChARF
Chains of Autoreplicative Random Forests for SNP imputation
(High-dimensional and low-sampled categorical data, features are ordered)



### ImputationChains.py: 

class **imputeEnsembleChains** for running an ensemble of imputation chains

### CompareModels.py: 

gridsearch for **imputeEnsembleChains.impute** function + imputation with mode

### _Experiments_Eucalyptus_.py ###

Example of imputation on Eucalyptus SNP dataset 


### Example of imputation: ###

```python

### upload dataset containing missing values
X_na = ... 

### model
mdl = ensemble.RandomForestClassifier(n_estimators=10)
delta=10 ## window size
nu=5 ## number of previous windows
model = ImputationChains.imputeEnsembleChains(mdl, delta=delta, past=nu)

### imputation
X_pred = model.impute(X_na)
```

### Example of model comparison: ###


```python

### upload complete ground-truth dataset
X = ...

### upload dataset containing missing values
X_na = ... 

### model
logi = CompareModels.compareModels(X, X_na, gridsearch=True, save_imputations=False, onehot=True)

### output imputation accuracy
print(logi.correctly_imputed)
```

