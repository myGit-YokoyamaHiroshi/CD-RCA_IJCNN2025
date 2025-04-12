# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 08:58:40 2025

@author: H.Yokoyama
"""

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)
from IPython import get_ipython
#get_ipython().magic('reset -sf')
#get_ipython().magic('clear')

#%%   
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
#%%
import sys
sys.path.append(current_path)
import numpy as np
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm
from dowhy.gcm.anomaly_scorers import *
from dowhy.gcm._noise import *
from dowhy.gcm.shapley import ShapleyConfig, ShapleyApproximationMethods

# from tigramite import data_processing as pp
from tigramite import plotting as tp
from sklearn.linear_model import LassoCV
from joblib import Parallel, delayed

from copy import deepcopy
from my_modules.various_func import *

#%%
save_path     = current_path + '/save_data/'
save_name     = 'intervened_graph' 
fullpath_save = save_path + save_name 

save_dict     = np.load(fullpath_save + '.npy', 
                        encoding='ASCII', 
                        allow_pickle='True').item()

graphs        = save_dict['graphs_true']
coeffs        = save_dict['coeffs_true']
amat          = save_dict['amat_true']
tau_max       = save_dict['tau_max']
Nnode         = save_dict['Nnode']

graphs_add    = save_dict['graphs_add']
coeffs_add    = save_dict['coeffs_add']
amat_add      = save_dict['amat_add']

graphs_change = save_dict['graphs_change']
coeffs_change = save_dict['coeffs_change']
amat_change   = save_dict['amat_change']

Ncond = len(amat_add.keys())
Nrand = len(amat_add[list(amat_add.keys())[0]])


#%%
causal_order = [0,2,1,3]
z_i     = 0
z_scale = .7

anomaly_sample = 500
Nt = 10000
Ntri = 50

labels = ['$X_1$', '$X_2$', '$X_3$', '$X_4$']

ATE_add    = np.zeros((Nrand, Ncond))
ATE_change = np.zeros((Nrand, Ncond))

from_id    = 0
to_id      = 3

ATEmat_true   = calc_total_effect_numerical(amat[0])
ATE_true      = ATEmat_true[to_id,from_id]
for i, key in enumerate(amat_add.keys()):
    #%%
    ##### Root cause analysis with graph interventions (random adding edge)
    for j, adj_add in enumerate(amat_add[key]):
        ATEmat_a     = calc_total_effect_numerical(adj_add[0])
        ATE_add[j,i] = ATEmat_a[to_id,from_id]
    
    for j, adj_change in enumerate(amat_change[key]):
        ATEmat_c        = calc_total_effect_numerical(adj_change[0])
        ATE_change[j,i] = ATEmat_c[to_id,from_id]
#%%
save_path = current_path + '/save_data/'
if os.path.exists(save_path)==False:  # Make the directory for figures
    os.makedirs(save_path)
save_name   = 'ATE' 

fullpath_save = save_path + save_name 

save_dict = {}

save_dict['ATE_true']   = ATE_true
save_dict['ATE_add']    = ATE_add
save_dict['ATE_change'] = ATE_change


np.save(fullpath_save, save_dict)