import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from scipy.io import arff

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.patches as ptch

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


#import sys
# insert at position 1 in the path, as 0 is the path of this file.
#sys.path.insert(1, './')
import CompareModels

###################################
############ DATA #################
###################################
    

### EUCALYPTUS dataset ###
Data_na_fracs = {}
Data = pd.read_csv('./Data/Eucalyptus/Eucalyptus.txt', sep='\t')
Data = Data.set_index('Unnamed: 0')
Data = Data.transpose()
Data = Data.iloc[:,:1000] ### truncate features for  quick tests ###
Data = Data.to_numpy()
#for frac in [0.01, 0.05, 0.1, 0.2, 0.3]:
for frac in [0.1]:
    name = 'Eucalyptus_' + str(frac)
    Data_na = pd.read_csv('./Data/Eucalyptus/Eucalyptus_na_' + str(frac) + '.csv')
    Data_na = Data_na.set_index('Unnamed: 0')
    Data_na_fracs[name] = Data_na.iloc[:,:1000].to_numpy()
print('Data loaded')




###########################################
############## IMPUTATION #################
###########################################


#path_res = './Results/Eucalyptus/'

print('###########################################')

for name in list(Data_na_fracs.keys()):      
    
    print(name)
    
    logi = CompareModels.compareModels(Data, Data_na_fracs[name], gridsearch=True, save_imputations=False, onehot=True)
    
    print('Correctly imputed:')
    print(logi.correctly_imputed)
    print('Changed out of purpose:')
    print(logi.changed_oop)
    print()
           
    

    
