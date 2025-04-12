# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:11:45 2025

@author: H.Yokoyama
"""
import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)

from IPython import get_ipython
get_ipython().magic('reset -sf')
get_ipython().magic('clear')

if os.name == 'posix': # for linux
    import matplotlib
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/arial.ttf')
    matplotlib.rc('font', family="Arial")
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
import dowhy.gcm as gcm
from dowhy.gcm.anomaly_scorers import *
from dowhy.gcm._noise import *
from dowhy.gcm.shapley import ShapleyConfig, ShapleyApproximationMethods

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import  Prediction
from joblib import Parallel, delayed

from my_modules.various_func import *

coeffs = {
            0: {(0, 0): 0.0,
                },
            1: {(0, 0): 3.8, 
                (2, 0): 1.0,  
                },
            2: {(2, 0): 0.0,
                },
            3: {
                (1, 0): 3.8
                },
        }

Nx      = 4
tau_max = 0
Ntri    = 5

graphs  = get_link_from_tigramite_coeffs(coeffs, Nx, tau_max)
amat    = get_amat_from_tigramite_coeffs(coeffs, Nx, tau_max)


# amat_add, graphs_add, coeffs_add, amat_change, graphs_change, coeffs_change = generate_intervened_graph(coeffs, graphs, amat, Ntri, 2)

out = Parallel(n_jobs=3, verbose=3)(
                delayed(generate_intervened_graph)
                (coeffs, graphs, amat, Ntri, k+1)
                for k in range(3))
#%%
amat_add      = {}
graphs_add    = {}
coeffs_add    = {}
amat_change   = {}
graphs_change = {}
coeffs_change = {}

for k, result in enumerate(out):
    amat_add[k+1]      = result[0]
    graphs_add[k+1]    = result[1]
    coeffs_add[k+1]    = result[2]
    amat_change[k+1]   = result[3]
    graphs_change[k+1] = result[4]
    coeffs_change[k+1] = result[5]
    
#%%
save_path = current_path + '/save_data/'
if os.path.exists(save_path)==False:  # Make the directory for figures
    os.makedirs(save_path)
save_name   = 'intervened_graph' 

fullpath_save = save_path + save_name 

save_dict = {}

save_dict['graphs_true']   = graphs
save_dict['coeffs_true']   = coeffs
save_dict['amat_true']     = amat
save_dict['tau_max']       = tau_max
save_dict['Nnode']         = Nx

save_dict['graphs_add']    = graphs_add
save_dict['coeffs_add']    = coeffs_add
save_dict['amat_add']      = amat_add 

save_dict['graphs_change'] = graphs_change
save_dict['coeffs_change'] = coeffs_change
save_dict['amat_change']   = amat_change

np.save(fullpath_save, save_dict)