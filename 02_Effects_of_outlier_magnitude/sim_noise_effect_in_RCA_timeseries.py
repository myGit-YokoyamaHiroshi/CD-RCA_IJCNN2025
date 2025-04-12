# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:28:54 2023
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
from tigramite.models import LinearMediation, Prediction
import sklearn
from joblib import Parallel, delayed
from scipy.linalg import block_diag
# from statsmodels.tsa.api import VAR
## カスタムスクリプトのパス
# sys.path.append(os.getcwd() + "/custom_lib/")
from custom_lib.convert_to_dowhy import * #カスタムスクリプトを読み込む
from copy import deepcopy

#%%
def make_lagged_time_label(amat, var_name, tau_min, tau_max):
    lag_val         = np.arange(tau_min-1, tau_max+1)
    var_names_lag   = []
    Nnode = amat.shape[0]
    Nvar  = Nnode/(tau_max + 1)
    cnt = 0
    lag = 0
    for i in range(Nnode):
        if lag_val[lag] == 0:
            str_name = var_name[cnt] + " at t" 
        else:
            str_name = var_name[cnt] + " at t-" + str(lag_val[lag]) 
        var_names_lag.append(str_name)
        cnt += 1
        if np.mod(cnt, Nvar)==0:
            cnt = 0
            lag += 1
    
    return var_names_lag


def generate_VAR_data(amat, T, exog, T_spurious = 20, confounder=False, Z=None):
     # amat: size (tau_max,p,p): amat[t] is the coefficient matrix at time t
    # T_spurious: the data from T_spurious:T will be returned
    # exog: a ndarray with shape (T + T_spurious,p)
    # containing the residuals of each node at each time-step
    noise   = exog
    
    if confounder==True:
        noise += Z
    
    tau_max = amat.shape[0] - 1
    p       = amat.shape[2]
    data_sim = np.zeros((T + T_spurious, p))
    initial = np.random.randn(tau_max, p)
    data_sim[:tau_max, :] = initial
    start = tau_max
    mixing_matrix = np.identity(p) - amat[0]


    for t in range(start, T + T_spurious):
        data_temp = np.copy(noise[t,:])
        for jj in range(1, tau_max + 1):
            data_temp += np.dot(amat[jj], data_sim[t - jj, :])
        # data_sim[t, :] = np.linalg.solve(mixing_matrix, data_temp)
        data_sim[t, :] = np.linalg.lstsq(mixing_matrix, data_temp)[0]
        
    return data_sim[T_spurious:,:]

def draw_data(data, Gnx_lag, Nsample):
    from scipy.stats import norm, uniform
    
    causal_model = gcm.StructuralCausalModel(Gnx_lag)
    
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    gcm.fit(causal_model, data)
    result = gcm.draw_samples(causal_model, num_samples=Nsample)
    
    return result, causal_model

def put_graph_into_dowhy(data, Gnx):
    causal_model = gcm.StructuralCausalModel(Gnx)    
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    
    return causal_model

def root_cause_analysis(dataframe, causal_model, target_node, 
                        target_sample_index, num_distribution_samples):

    # Automatically assigns additive noise models to non-root nodes
    gcm.auto.assign_causal_mechanisms(causal_model, dataframe)
    gcm.fit(causal_model, dataframe)
    
    node_samples, _  = noise_samples_of_ancestors(causal_model, target_node, num_distribution_samples)
    anomaly_score    = []
    attribution_list = []

    #Shapley値の分解の近似を設定する
    config = ShapleyConfig(approximation_method = ShapleyApproximationMethods.SUBSET_SAMPLING) # defaultは5000。
    
    if target_sample_index == None:
        st  = tau_max
        end = len(dataframe)
    else:
        st  = target_sample_index
        end = target_sample_index + 1
    
    for target_sample in range(st, end):
        anomaly_samples = dataframe.iloc[target_sample:(target_sample+1),:]

        tau_score = MeanDeviationScorer() # 異常メトリック
        IT_score = ITAnomalyScorer(tau_score)

        #異常度合いの計算：
        IT_score.fit(node_samples[target_node].to_numpy())
        anomaly_score.append(IT_score.score(anomaly_samples.to_numpy())[target_node])
        
        # その分解の計算：
        attribution = gcm.attribute_anomalies(causal_model,target_node = target_node,
                                              anomaly_samples=anomaly_samples,
                                              num_distribution_samples = num_distribution_samples,
                                              anomaly_scorer = tau_score,
                                              shapley_config = config)
        attribution_list.append(attribution)

    return anomaly_score, attribution_list


def do_root_cause_analysis(Nnode, Ntrain, amat, graph, link, z_i, z_scale, 
                           label, lagged_label, target, tau_max,
                           num_distribution_samples= 10000):
    #%%
    T_spurious      = 20
    anomaly_samples = 500 #+ T_spurious 
    #### make unobserved value Z
    Z             = np.zeros((Ntrain+T_spurious, Nnode))
    Z[anomaly_samples + T_spurious, z_i] += z_scale#np.sqrt(z_SD) * np.random.randn(100)
    
    #### draw distorted data by Z 
    # Nx           = np.random.uniform(low=0, high=.5, size=(Ntrain+T_spurious,Nnode))
    Nx           = np.random.uniform(low=0, high=1, size=(Ntrain+T_spurious,Nnode))
    
    data_err      = generate_VAR_data(amat, Ntrain, Nx, 
                                      T_spurious = T_spurious, 
                                      confounder=True, Z=Z)
    
    ### convert timeseries matrix to lagged data matrix
    lagged_data = create_lagged_data(data_err, tau_max = tau_max)
    
    p = data_err.shape[1]
    Gnx_lag, amat_lag = convert_tigramite_to_dowhy(graph, link)
    ### convert from array to dataframe
    dataframe_err  = pd.DataFrame(lagged_data, columns = np.array(range(p * (tau_max + 1))))
    #### put the causal graph into dowhy.gcm
    causal_model = put_graph_into_dowhy(dataframe_err, Gnx_lag)
    
    #### do root cause analysis
    target_node    = np.where(np.array(lagged_label)==target)[0]
    
    idx_tmp = np.arange(anomaly_samples, anomaly_samples + tau_max + 1)
    anomaly_samples = idx_tmp[abs(lagged_data[idx_tmp, target_node[0]]).argmax()]
    #%%
    anomaly_score, attribution_list = root_cause_analysis(dataframe_err, 
                                                          causal_model, 
                                                          target_node[0], 
                                                          anomaly_samples,
                                                          num_distribution_samples)

    attribution = attribution_list[0]
    keys        = attribution.keys()
    key_val     = [int(key) for key in keys]
    values      = np.array([attribution[key][0] for key in keys])

    attrib_err  = abs(anomaly_score[0] - values.sum())
    attrib_max  = values.max()
    
    result      = [Xnames[z_i] in var for var in Xnames_lag]
    idx         = np.where(np.array(result)==True)[0]
    
    if (idx == key_val[values.argmax()]).any():
        correct = 1
    else:
        correct = 0
    
    return correct, attrib_err, attrib_max


def calc_total_effect(dataframe, causal_model, var_names_lag, varname_interven, varname_target):

    ##### fit the generative model
    gcm.auto.assign_causal_mechanisms(causal_model, dataframe)
    gcm.fit(causal_model, dataframe)

    ### get interventional samples
    result  = [varname_interven in var for var in var_names_lag]
    idx     = np.where(np.array(result)==True)[0]
    target  = np.where(np.array(var_names_lag)==varname_target)[0]
    
    do1    = gcm.interventional_samples(causal_model,
                                        {num: lambda x: 1 for num in idx},
                                        num_samples_to_draw=1000)

    do0    = gcm.interventional_samples(causal_model,
                                        {num: lambda x: 0 for num in idx},
                                        num_samples_to_draw=1000)
    
    #### assess the averaged total effect
    effect = do1[target].values.mean() - do0[target].values.mean()
    return effect

def convert_tigramite_to_dowhy(tigramite_dag, tigramite_coeff):
    import networkx as nx
    # convert the DAG returned by pcmci to the format of networkx used in dowhy
    p       = tigramite_dag.shape[0]
    tau_max = tigramite_dag.shape[2] - 1
    amat    = np.full((p * (tau_max + 1), p * (tau_max + 1)),-1.0)
    G       = nx.DiGraph()
    for i in range(p * (tau_max + 1)):
        G.add_node(i)
    for i in range(p):
        for j in range(p):
            tau = 0
            if tigramite_dag[i, j, 0] == '-->' and tigramite_dag[j, i, 0] == '<--':
                # i --> j
                amat[i,j] = tigramite_coeff[i,j,tau]
                G.add_edge(i, j)
            if tigramite_dag[i, j, 0] == '<--' and tigramite_dag[j, i, 0] == '-->':
                # j  --> i
                amat[j,i] = tigramite_coeff[j,i,tau]
                G.add_edge(j, i)
            if tigramite_dag[i, j, 0] == '' and tigramite_dag[j, i, 0] == '':
                amat[i,j] = 0

            # time-invariance
            for delta_tau in range(1,tau_max + 1):
                amat[i + delta_tau*p, j + delta_tau * p] = amat[i, j]
                amat[j + delta_tau * p, i + delta_tau * p] = amat[j, i]
                if amat[i,j] > 0:
                    G.add_edge(i + delta_tau * p, j + delta_tau * p)
                if amat[j,i] > 0:
                    G.add_edge(j + delta_tau * p, j + delta_tau * p)

            for tau in range(1,tau_max + 1):
                amat[j, i + tau*p] = 0 # arrow of time
                if tigramite_dag[i,j,tau] == '-->':
                    # i-tau --> j
                    amat[i + tau * p,j] = tigramite_coeff[i,j,tau]
                    G.add_edge(i + tau*p, j)
                if tigramite_dag[i,j,tau] == '':
                    amat[i + tau * p,j] = 0

                # time-invariance
                for delta_tau in range(1,tau_max - tau + 1):
                    if delta_tau > 0:
                        amat[i + (tau + delta_tau) * p, j + delta_tau * p] = amat[i + tau * p, j]
                        amat[j + delta_tau * p, i + (tau + delta_tau) * p] = amat[j, i + tau * p]
                        if amat[i + tau*p,j] > 0:
                            coeff = amat[i + tau*p,j]
                            G.add_edge(i + (tau + delta_tau) * p, j + delta_tau * p, weight=coeff)
                        if amat[j, i + tau * p] > 0:
                            coeff = amat[j, i + tau * p]
                            G.add_edge(j + delta_tau * p, i + (tau + delta_tau) * p, weight=coeff)

    return G,amat
#%% [1] Generate synthetic data
np.random.seed(1000)
###########################
dataset = 1
N       = 4
tau_max = 3
num_distribution_samples = 10000 # 

fig_path = current_path + '/figures/RCA_noise_effect_timeseries/N%05d/dataset%02d/'%(num_distribution_samples, dataset)
if os.path.exists(fig_path)==False:  # Make the directory for figures
    os.makedirs(fig_path)

if dataset == 1:
    links_coeffs = {
                    0: {(0, -1): 0.8,
                        },
                    1: {(0, -1): 0.8, #(0, -2): 0.2, (0, -3): 0.2, 
                        (2, -1): 0.8, #(2, -2): 0.2, (2, -3): 0.2, 
                        },
                    2: {(2, -1): 0.8,
                        },
                    3: {
                        #(0, -1): 0.2, 
                        (1, -1): 0.8#, (1, -2): 0.1, (1, -3): 0.1, 
                        },
                    }
elif dataset == 2:
    links_coeffs = {
                    0: {(0, -1): 0.2,
                        },
                    1: {(0, -1): 0.2, #(0, -2): 0.2, (0, -3): 0.2, 
                        (2, -1): 0.4, #(2, -2): 0.2, (2, -3): 0.2, 
                        },
                    2: {(2, -1): 0.2,
                        },
                    3: {
                        #(0, -1): 0.2, 
                        (1, -1): 0.1, #(1, -2): 0.1, (1, -3): 0.1, 
                        },
                    }
#%%
### make variables to plot graph structures
graph_true   = np.zeros((N, N, tau_max+1), dtype='object')
link_true    = np.zeros((N, N, tau_max+1))
amat_true    = np.zeros((tau_max+1, N, N))  
for j in range(N):
    for p in links_coeffs[j].keys():
        graph_true[p[0], j, abs(p[1])] = '-->'
        link_true[p[0], j, abs(p[1])]  = links_coeffs[j][p]
        amat_true[abs(p[1]), j, p[0]]  = links_coeffs[j][p]
        
graph_true[graph_true==0] = ''
#%%
T_spurious   = 20
Nnode        = link_true.shape[0]
Ntrain       = 10000

# Nx           = np.random.uniform(low=0, high=0.5, size=(Ntrain+T_spurious,Nnode))
Nx           = np.random.uniform(low=0, high=1, size=(Ntrain+T_spurious,Nnode))

data_sim  = generate_VAR_data(amat_true, Ntrain, Nx,
                              T_spurious = T_spurious)

seed      = 201

tp.plot_time_series_graph(
    graph=graph_true,
    val_matrix=link_true,
    var_names= [r'$X_0$', r'$X_1$', r'$X_2$', r'$Y$'],
    link_colorbar_label='coefficient',
    label_fontsize=16,
    )
plt.title('lagged regression model\n', fontsize=18)
plt.savefig(fig_path + 'synthetic_data_time_graph.png', bbox_inches="tight")
plt.savefig(fig_path + 'synthetic_data_time_graph.svg', bbox_inches="tight")
plt.show()
#%% Plot training data set

x_train    = data_sim[:, :3]
y_train    = data_sim[:, 3]

base = np.flipud(np.arange(0, 4*3, 3))

for n in range(4):
    timestamp = np.arange(0, len(y_train))
    if n < 3:
        x = x_train[:,n] - x_train[:,n].mean() 
        plt.plot(timestamp, x + base[n], c='k')
    elif n==3:
        y = y_train - y_train[0]
        plt.plot(timestamp, y + base[n], c='r')

plt.yticks(np.flipud(base),
            [r'$Y$', r'$X_2$', r'$X_1$', r'$X_0$'])
plt.xlabel('# sample')
plt.xlim(0,len(y_train))
plt.savefig(fig_path + 'synthetic_data_time_series.png', bbox_inches="tight")
plt.savefig(fig_path + 'synthetic_data_time_series.svg', bbox_inches="tight")
plt.show()
#%% Make lagged dataset to apply dowhy package
lagged_data = create_lagged_data(data_sim, tau_max = tau_max)
p = data_sim.shape[1]
### convert from array to dataframe
dataframe   = pd.DataFrame(lagged_data, columns = np.array(range(p * (tau_max + 1))))

Gnx_lag, amat_lag = convert_tigramite_to_dowhy(graph_true, link_true)
_, causal_model   = draw_data(dataframe.iloc[tau_max:,:], Gnx_lag, Ntrain)
#%% Calculate average causal effect (ACE)

Xnames     = ['X0', 'X1', 'X2', 'Y']
Xnames_lag = make_lagged_time_label(amat_lag, Xnames, 1, tau_max)

target_names     = ['X0', 'X1', 'X2']
effect     = {}
idx_target = Nnode-1
for Xname in target_names:  
    
    out = Parallel(n_jobs=-1, verbose=6)(
                     delayed(calc_total_effect)
                          (dataframe.iloc[tau_max:,:], causal_model, Xnames_lag, Xname, 'Y at t') 
                              for i in range(1000))
    effect[Xname] = np.array(out)

##### plot result in ACE analysis
fig = plt.figure(figsize=(15, 3))
gs  = fig.add_gridspec(1,len(Xnames))
plt.subplots_adjust(wspace=0.6, hspace=0.3)

for i, Xname in enumerate(target_names):
    plt.subplot(gs[0, i]) 
    plt.hist(effect[Xname], bins=30, range=(-0.1, 1), edgecolor='black', linewidth=1.2)
    
    if i == 0:
        plt.ylabel('frequency (a.u.)')
    
    xlabel = 'ACE (a.u.)'
    plt.xlabel(xlabel)
    plt.xlim(-0.1, 1)
    plt.xticks(ticks=np.arange(0,1.5,0.5))
    
    plt.title('$\mathbb{E}[Y|~do(X_{%d}=1)]-\mathbb{E}[Y|~do(X_{%d}=0)]$'%(i,i), fontsize=15)

plt.savefig(fig_path + 'ACE_root_cause_graph_%02d.png'%dataset, bbox_inches="tight")
plt.savefig(fig_path + 'ACE_root_cause_graph_%02d.svg'%dataset, bbox_inches="tight")
plt.show()
#%% Do Root cause analysis with unobservational cause Z
np.random.seed(10)

# sd_list    = np.arange(0.5, 10.5, 0.5)
z_list     = np.arange(0.0, 10.5, 0.5)
z_list[0]  = 0.1
coeff_list = np.arange(0.5, 3.5, 0.5)

##### Root cause X0, coeff beta10
accuracy1       = np.zeros((len(z_list), len(coeff_list)))
attribute_error = np.zeros((len(z_list), len(coeff_list)))

z_i      = 0
idx_amat = (1, 1, 0) # (tau, i, j), coefficient from 0 to 1  at t-tau
idx_link = (1, 0, 1) # (i, j, tau), coefficient from 0 to 1  at t-tau
target   = 'Y at t'
for j, coeff in enumerate(coeff_list):
    amat  = deepcopy(amat_true)
    graph = deepcopy(graph_true)
    link  = deepcopy(link_true)

    amat[idx_amat] = coeff
    link[idx_link] = coeff
    
    print(amat[idx_amat])
    print(link[idx_link])
    
    # do_root_cause_analysis(Nnode, Ntrain, amat, graph, link, 1, 10, 
    #   Xnames, Xnames_lag, target, tau_max, num_distribution_samples)
    
    for i, z_scale in enumerate(z_list):        
        out = Parallel(n_jobs=-1, verbose=6)(
                         delayed(do_root_cause_analysis)
                             (Nnode, Ntrain, amat, graph, link, z_i, z_scale, 
                              Xnames, Xnames_lag, target, tau_max, 
                              num_distribution_samples)
                                 for k in range(50))
        
        correct = np.array([res[0] for res in out])    
        Np      = np.sum(correct==1)
        accuracy1[i,j] = Np/len(correct)
        
        err = np.array([res[1] for res in out])
        attribute_error[i,j] = err.mean()
        
        print('(Z = %.1f, beta = %.1f) acc = %.4f'%(z_scale, coeff, Np/len(correct)))
########### plot result
fig, ax = plt.subplots(figsize=(10,10))
im = ax.contourf(coeff_list, z_list, 
                  accuracy1, 
                  cmap='Reds',
                  levels = np.arange(0, 1.1,.1))
im.set_clim(vmin=0, vmax=1)
ax.set_xlabel('coefficient', fontsize=24)
ax.set_ylabel('Z', fontsize=24)
ax.set_title('root cause: $X_{%d}$'%(z_i))
ax.set_xticks(coeff_list)

if dataset==1:
    ax.plot([0.5, 3.0], [0.5, 0.5], c='b', linewidth=4)

cbar = fig.colorbar(im, ticks=np.arange(0.0, 1.2, .2))
cbar.ax.set_ylim(0.0, 1)  
cbar.set_label('accuracy (a.u.)')



fig_save_dir = current_path + '/figures/'
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)
    
plt.savefig(fig_path + 'RCA_acc_noise_effect_X%d.png'%z_i, bbox_inches="tight")
plt.savefig(fig_path + 'RCA_acc_noise_effect_X%d.svg'%z_i, bbox_inches="tight")
    
plt.show()
#%%
##### Root cause X1, coeff beta31
accuracy2       = np.zeros((len(z_list), len(coeff_list)))
attribute_error = np.zeros((len(z_list), len(coeff_list)))

z_i      = 1
idx_amat = (1, 3, 1) # (tau, i, j), coefficient from 1 to 3  at t-1
idx_link = (3, 1, 1) # (i, j, tau), coefficient from 1 to 3  at t-1
target   = 'Y at t'
for j, coeff in enumerate(coeff_list):
    amat  = deepcopy(amat_true)
    graph = deepcopy(graph_true)
    link  = deepcopy(link_true)
    
    amat[idx_amat] = coeff
    link[idx_link] = coeff
    
    print(amat[idx_amat])
    print(link[idx_link])
    
    for i, z_scale in enumerate(z_list):
        out = Parallel(n_jobs=-1, verbose=6)(
                         delayed(do_root_cause_analysis)
                             (Nnode, Ntrain, amat, graph, link, z_i, z_scale, 
                              Xnames, Xnames_lag, target, tau_max, 
                              num_distribution_samples)
                                 for k in range(50))
        
        correct = np.array([res[0] for res in out])    
        Np      = np.sum(correct==1)
        accuracy2[i,j] = Np/len(correct)
        
        err = np.array([res[1] for res in out])
        attribute_error[i,j] = err.mean()
        
        print('(Z = %.1f, beta = %.1f) acc = %.4f'%(z_scale, coeff, Np/len(correct)))
########### plot result
fig, ax = plt.subplots(figsize=(10,10))
im = ax.contourf(coeff_list, z_list, 
                  accuracy2, 
                  cmap='Reds',
                  levels = np.arange(0, 1.1,.1))
im.set_clim(vmin=0, vmax=1)
ax.set_xlabel('coefficient', fontsize=24)
ax.set_ylabel('Z', fontsize=24)
ax.set_title('root cause: $X_{%d}$'%(z_i))
ax.set_xticks(coeff_list)

if dataset==1:
    ax.plot([0.5, 3.0], [0.5, 0.5], c='b', linewidth=4)

cbar = fig.colorbar(im, ticks=np.arange(0.0, 1.2, .2))
cbar.ax.set_ylim(0.0, 1)  
cbar.set_label('accuracy (a.u.)')

fig_save_dir = current_path + '/figures/'
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)
    
plt.savefig(fig_path + 'RCA_acc_noise_effect_X%d.png'%z_i, bbox_inches="tight")
plt.savefig(fig_path + 'RCA_acc_noise_effect_X%d.svg'%z_i, bbox_inches="tight")
    
plt.show()
#%%
##### Root cause X1, coeff beta10
accuracy3       = np.zeros((len(z_list), len(coeff_list)))
attribute_error = np.zeros((len(z_list), len(coeff_list)))

z_i      = 1
idx_amat = (1, 1, 0) # (tau, i, j), coefficient from 0 to 1  at t-tau
idx_link = (1, 0, 1) # (i, j, tau), coefficient from 0 to 1  at t-tau

target   = 'Y at t'
for j, coeff in enumerate(coeff_list):
    amat  = deepcopy(amat_true)
    graph = deepcopy(graph_true)
    link  = deepcopy(link_true)

    amat[idx_amat] = coeff
    link[idx_link] = coeff
    
    print(amat[idx_amat])
    print(link[idx_link])
    
    for i, z_scale in enumerate(z_list):
        
        out = Parallel(n_jobs=-1, verbose=6)(
                         delayed(do_root_cause_analysis)
                             (Nnode, Ntrain, amat, graph, link, z_i, z_scale, 
                              Xnames, Xnames_lag, target, tau_max, 
                              num_distribution_samples)
                                 for k in range(50))
        
        correct = np.array([res[0] for res in out])    
        Np      = np.sum(correct==1)
        accuracy3[i,j] = Np/len(correct)
        
        err = np.array([res[1] for res in out])
        attribute_error[i,j] = err.mean()
        
        print('(Z = %.1f, beta = %.1f) acc = %.4f'%(z_scale, coeff, Np/len(correct)))
########### plot result
#%%
fig, ax = plt.subplots(figsize=(10,10))
im = ax.contourf(coeff_list, z_list, 
                  accuracy3, 
                  cmap='Reds',
                  levels = np.arange(0, 1.1,.1))
im.set_clim(vmin=0, vmax=1)
ax.set_xlabel('coefficient', fontsize=24)
ax.set_ylabel('Z', fontsize=24)
ax.set_title('root cause: $X_{%d}$'%(z_i))
ax.set_xticks(coeff_list)

if dataset==1:
    ax.plot(coeff_list, coeff_list, c='b', linewidth=4)

cbar = fig.colorbar(im, ticks=np.arange(0.0, 1.2, .2))
cbar.ax.set_ylim(0.0, 1)  
cbar.set_label('accuracy (a.u.)')

fig_save_dir = current_path + '/figures/'
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)
    
plt.savefig(fig_path + 'RCA_acc_noise_effect_X%d_beta10.png'%z_i, bbox_inches="tight")
plt.savefig(fig_path + 'RCA_acc_noise_effect_X%d_beta10.svg'%z_i, bbox_inches="tight")
    
plt.show()