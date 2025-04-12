# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:06:36 2025

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
save_name     = 'ATE' 
fullpath_save = save_path + save_name 

save_dict     = np.load(fullpath_save + '.npy', 
                        encoding='ASCII', 
                        allow_pickle='True').item()

ATE_true   = save_dict['ATE_true']
ATE_add    = save_dict['ATE_add']
ATE_change = save_dict['ATE_change']
#%%
causal_order = [0,2,1,3]
z_i     = 0
z_scale = .5

anomaly_sample = 500
Nt = 10000
Ntri = 50

labels = ['$X_1$', '$X_2$', '$X_3$', '$X_4$']

acc_add    = np.zeros((Nrand, Ncond))
acc_change = np.zeros((Nrand, Ncond))

shd_add    = np.zeros((Nrand, Ncond))
shd_change = np.zeros((Nrand, Ncond))

for i, key in enumerate(amat_add.keys()):
    #%%
    ##### Root cause analysis with graph interventions (random adding edge)
    for j, amat_rca in enumerate(amat_add[key]):
        shd_add[j, i] = SHD(amat[0,:,:], amat_rca[0,:,:], double_for_anticausal=False)
        
        
        data_anomaly_list = Parallel(n_jobs=Ntri, verbose=3)(
                                delayed(generate_noise_distorted_data)
                                (Nnode, Nt, amat[0,:,:], causal_order, z_i, z_scale, anomaly_sample)
                                for tri in range(Ntri))
        
        
        # do_root_cause_analysis(data_anomaly_list[0], z_i, anomaly_sample, amat_rca[0,:,:], 
        #                        labels, Nnode, target_node = 3, 
        #                        num_distribution_samples=10000)
        
        out = Parallel(n_jobs=Ntri, verbose=3)(
                        delayed(do_root_cause_analysis)
                            (data_anomaly, z_i, anomaly_sample, amat_rca[0,:,:], 
                             labels, Nnode, target_node = 3, 
                             num_distribution_samples=10000)
                             for data_anomaly in data_anomaly_list)
        
        correct = np.array([result[0] for result in out])
        Np      = np.sum(correct==1)
        acc_add[j,i] = Np/len(correct)
    #%%
    ##### Root cause analysis with graph interventions (random changes in edge weight)
    for j, amat_rca in enumerate(amat_change[key]):
        shd_change[j, i] = SHD(amat[0,:,:], amat_rca[0,:,:], double_for_anticausal=False)
        
        data_anomaly_list = Parallel(n_jobs=Ntri, verbose=3)(
                                delayed(generate_noise_distorted_data)
                                (Nnode, Nt, amat[0,:,:], causal_order, z_i, z_scale, anomaly_sample)
                                for tri in range(Ntri))
        
        out = Parallel(n_jobs=Ntri, verbose=3)(
                        delayed(do_root_cause_analysis)
                            (data_anomaly, z_i, anomaly_sample, amat_rca[0,:,:], 
                             labels, Nnode, target_node = 3, 
                             num_distribution_samples=10000)
                             for data_anomaly in data_anomaly_list)
        
        correct = np.array([result[0] for result in out])
        Np      = np.sum(correct==1)
        acc_change[j,i] = Np/len(correct)
#%%
     
save_csv = current_path + '/save_data/csv/add/RCA_results_z_%03.1f.csv'%z_scale

colmun   = np.hstack(('z_scale', '$k=1$', '$k=2$', '$k=3$'))
mu       = np.mean(acc_add, axis=0)
sd       = np.std(acc_add, axis=0)
csv_mat  = np.array(['%.2f$\pm$%.2f'%(val1, val2) for val1, val2 in zip(mu, sd)])
csv_mat  = np.hstack(('%.2f'%z_scale, csv_mat))

df = pd.DataFrame(csv_mat[np.newaxis,:], 
                  columns=colmun)
df.to_csv(save_csv, index = False)

save_csv = current_path + '/save_data/csv/change/RCA_results_z_%03.1f.csv'%z_scale

colmun   = np.hstack(('z_scale', '$k=1$', '$k=2$', '$k=3$'))
mu       = np.mean(acc_change, axis=0)
sd       = np.std(acc_change, axis=0)
csv_mat  = np.array(['%.2f$\pm$%.2f'%(val1, val2) for val1, val2 in zip(mu, sd)])
csv_mat  = np.hstack(('%.2f'%z_scale, csv_mat))

df = pd.DataFrame(csv_mat[np.newaxis,:], 
                  columns=colmun)
df.to_csv(save_csv, index = False)
    
