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

# if os.name == 'posix': # for linux
#     import matplotlib
#     from matplotlib import font_manager
#     font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/arial.ttf')
#     matplotlib.rc('font', family="Arial")
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
from joblib import Parallel, delayed

# from statsmodels.tsa.api import VAR
from copy import deepcopy

#%%
def generate_data(causal_order, Nnode, noise, Nt, amat, confounder=False, Z=None):
    data_sim = np.zeros((Nt, Nnode))
    
    for n in causal_order:
        data_sim[:,n] += noise[:,n]
        
        if amat[n,:].sum() != 0:
            data_sim[:,n] += amat[n,:] @ data_sim.T
            
        if confounder == True:
            data_sim[:,n] += Z[:,n]    
    
    return data_sim

def plot_graph_non_timeseries(graph, link_coeffs, seed, vmin=0, vmax=1):
    import networkx as nx
    import matplotlib as mpl
    plt.figure(figsize=(5, 5))
    im_ratio = 5 / 5
    
    links = deepcopy(link_coeffs)
    
    # link_coeffs[(graph=='') | (graph=='<--')] = 0
    links[(graph=='') | (graph=='<--')] = 0
    links = links[:,:,0]
    links = links - np.diag(np.diag(links))
    Gnx   = nx.from_numpy_array(links, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(Gnx, seed=seed)
    
    labels = {i : i for i in Gnx.nodes()}          
    
    node_sizes  = [1000  for i in range(len(Gnx))]
    M           = Gnx.number_of_edges()
    edge_colors = np.ones(M, dtype = int)
    
    
    weight = deepcopy(links).reshape(-1)
    weight = weight[weight != 0]
    edge_alphas = weight/vmax
    edge_alphas[edge_alphas>1] = 1
    
    nodes       = nx.draw_networkx_nodes(Gnx, pos, node_size=node_sizes, node_color='blue')
    edges       = nx.draw_networkx_edges(Gnx, pos, node_size=node_sizes, arrowstyle='->',
                                         connectionstyle='arc3, rad = 0.09',
                                         arrowsize=10, edge_color=edge_colors,
                                         width=4,
                                         edge_vmin=vmin, edge_vmax=vmax)
    
    ##### plot graph
    nx.draw_networkx_labels(Gnx, pos, labels, font_size=15, font_color = 'w')
    plt.axis('equal')
    # set alpha value for each edge
    if vmin < 0:       
        from matplotlib.colors import LinearSegmentedColormap
        
        cm_b = plt.get_cmap('Blues', 128)
        cm_r = plt.get_cmap('Reds', 128)
        
        color_list_b = []
        color_list_r = []
        for i in range(128):
            color_list_b.append(cm_b(i))
            color_list_r.append(cm_r(i))
        
        color_list_r = np.array(color_list_r)
        color_list_b = np.flipud(np.array(color_list_b))
        
        color_list   = list(np.concatenate((color_list_b, color_list_r), axis=0))
        
        cm = LinearSegmentedColormap.from_list('custom_cmap', color_list)
            
    elif vmin>=0:
        cm = plt.get_cmap('Reds', 256)
        
    for i in range(M):
        if vmin < 0:
            c_idx = int((weight[i]/vmax + 1)/2 * cm.N)
        elif vmin>=0:
            c_idx = int((edge_alphas[i] * cm.N))
            
        rgb = np.array(cm(c_idx))[0:3]
        # edges[i].set_alpha(edge_alphas[i])
        edges[i].set_color(rgb)
    
    pc = mpl.collections.PatchCollection(edges, cmap=cm)
    pc.set_array(edge_colors)
    pc.set_clim(vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.set_axis_off()
    
    plt.colorbar(pc, ax=ax, label='coupling strength (a.u.)', 
                 fraction=0.05*im_ratio, pad=0.035)


def draw_data(data, link_coeffs, Nsample):
    import networkx as nx
    from scipy.stats import uniform, norm
    
    links = link_coeffs[:,:,0]
    links = links - np.diag(np.diag(links))
    amat  = links.T
    
    Gnx   = nx.from_numpy_array(links, create_using=nx.DiGraph())    
    causal_model = gcm.StructuralCausalModel(Gnx)
    for i, n in enumerate(Gnx.nodes):
        if amat[i,:].sum()==0:
            causal_model.set_causal_mechanism(n, 
                                              gcm.EmpiricalDistribution())
        else:
            causal_model.set_causal_mechanism(n, 
                                              gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor(), 
                                                                      noise_model=gcm.ScipyDistribution(uniform)))
    gcm.fit(causal_model, data)
    result = gcm.draw_samples(causal_model, num_samples=Nsample)
    
    return result, causal_model

def put_graph_into_dowhy(amat):
    import networkx as nx
    from scipy.stats import uniform, norm
    
    links = amat.T 
    Gnx   = nx.from_numpy_array(links, create_using=nx.DiGraph())    
    causal_model = gcm.StructuralCausalModel(Gnx)
    
    # gcm.auto.assign_causal_mechanisms(causal_model, data)
    for i, n in enumerate(Gnx.nodes):
        if amat[i,:].sum()==0:
            causal_model.set_causal_mechanism(n, 
                                              gcm.EmpiricalDistribution())
        else:
            causal_model.set_causal_mechanism(n, 
                                              gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor(), 
                                                                      noise_model=gcm.ScipyDistribution(uniform)))
    return causal_model

def root_cause_analysis(dataframe, causal_model, target_node, target_sample_index, num_distribution_samples):

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

def do_root_cause_analysis(Nnode, Ntrain, amat, causal_order, z_i, z_scale, num_distribution_samples=10000):
    #### put the causal graph into dowhy.gcm
    causal_model = put_graph_into_dowhy(amat)
    #### make unobserved value Z
    Z             = np.zeros((Ntrain,Nnode))
    Z[500:501, z_i] += z_scale#np.sqrt(z_SD) * np.random.randn(100)
    
    #### draw distorted data by Z 
    # Nx             = np.random.uniform(low=0, high=0.5, size=(Ntrain,Nnode))
    Nx             = np.random.uniform(low=0, high=1, size=(Ntrain,Nnode))
    data_err       = generate_data(causal_order, Nnode, Nx, Ntrain, amat, 
                                   confounder=True, Z=Z)
    target_node    = 3
    anomaly_sample = 500#abs(data_err[:,target_node]).argmax()
    dataframe_err  = pd.DataFrame(data_err, columns = np.array(range(4)))
    anomaly_score, attribution_list = root_cause_analysis(dataframe_err, 
                                                          causal_model, 
                                                          target_node, 
                                                          anomaly_sample,
                                                          num_distribution_samples)
    attribution = attribution_list[0]
    keys        = attribution.keys()
    key_val     = [int(key) for key in keys]
    values      = np.array([attribution[key][0] for key in keys])
    attrib_err  = abs(anomaly_score[0] - values.sum())
    attrib_max  = values.max()
    
    if key_val[values.argmax()] == z_i:
        correct = 1
    else:
        correct = 0
    
    return correct, attrib_err, attrib_max

def calc_total_effect(dataframe, causal_model, idx_interven, idx_target):

    ##### fit the generative model
    gcm.auto.assign_causal_mechanisms(causal_model, dataframe)
    gcm.fit(causal_model, dataframe)

    ### get interventional samples
    idx    = idx_interven

    do1    = gcm.interventional_samples(causal_model,
                                        {idx: lambda x: 1},
                                        num_samples_to_draw=1000)
    do0    = gcm.interventional_samples(causal_model,
                                        {idx: lambda x: 0},
                                        num_samples_to_draw=1000)
    
    #### assess the averaged total effect
    effect = do1[idx_target].values.mean() - do0[idx_target].values.mean()
    return effect
#%% [1] Generate synthetic data
np.random.seed(1000)
###########################
dataset = 1
N       = 4
tau_max = 0

# number of samples to estimate anomaly and attritude value in Shapley-RCA
num_distribution_samples = 10000 #  500 # 


fig_path = current_path + '/figures/RCA_noise_effect/N%05d/dataset%02d/'%(num_distribution_samples, dataset)
if os.path.exists(fig_path)==False:  # Make the directory for figures
    os.makedirs(fig_path)

if dataset == 1:
    links_coeffs = {
                    0: {(0, 0): 0.0,
                        },
                    1: {(0, 0): 0.8, 
                        (2, 0): 0.8,  
                        },
                    2: {(2, 0): 0.0,
                        },
                    3: {
                        (1, 0): 0.8
                        },
                    }
elif dataset == 2:
    links_coeffs = {
                    0: {(0, 0): 0.0,
                        },
                    1: {(0, 0): 0.2, 
                        (2, 0): 0.4, 
                        },
                    2: {(2, 0): 0.0,
                        },
                    3: { 
                        (1, 0): 0.1, 
                        },
                    }
#%%
### make variables to plot graph structures
graph_true   = np.zeros((N, N, tau_max+1), dtype='object')
graph_true[graph_true==0] = ''

link_true    = np.zeros((N, N, tau_max+1))
amat_true    = np.zeros((tau_max+1, N, N))  
for j in range(N):
    for p in links_coeffs[j].keys():
        if links_coeffs[j][p] != 0:
            graph_true[p[0], j, abs(p[1])] = '-->' 
        
        link_true[p[0], j, abs(p[1])]  = links_coeffs[j][p]
        amat_true[abs(p[1]), j, p[0]]  = links_coeffs[j][p]
#%%
Nnode        = link_true.shape[0]
Ntrain       = 10000

# Nx           = np.random.uniform(low=0, high=0.5, size=(Ntrain,Nnode))
Nx           = np.random.uniform(low=0, high=1, size=(Ntrain,Nnode))
causal_order = [0,2,1,3]

data_sim     = generate_data(causal_order, Nnode, Nx, Ntrain, amat_true[0,:,:])

seed      = 201

plot_graph_non_timeseries(graph_true, link_true, seed)
plt.savefig(fig_path + 'synthetic_data_graph.png', bbox_inches="tight")
plt.savefig(fig_path + 'synthetic_data_graph.svg', bbox_inches="tight")
plt.show()


dataframe = pd.DataFrame(data_sim, columns = np.array(range(4)))

result, causal_model = draw_data(dataframe, link_true, Ntrain)

#%% Plot training data set
data_sim_ = np.zeros(data_sim.shape)

for n in result.keys():
    data_sim_[:,n] = result[n].to_numpy()

x_train    = data_sim_[:, :3]
y_train    = data_sim_[:, 3]

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
#%% Calculate average causal effect (ACE)
Xnames     = ['X0', 'X1', 'X2']
effect     = {}
idx_target = Nnode-1
for i, Xname in enumerate(Xnames):
    idx_interven = i    
    out = Parallel(n_jobs=-1, verbose=6)(
                     delayed(calc_total_effect)
                          (dataframe, causal_model, idx_interven, idx_target) 
                              for i in range(1000))
    effect[Xname] = np.array(out)
#%%
##### plot result in ACE analysis
fig = plt.figure(figsize=(15, 3))
gs  = fig.add_gridspec(1,len(Xnames))
plt.subplots_adjust(wspace=0.6, hspace=0.3)

for i, Xname in enumerate(Xnames):
    plt.subplot(gs[0, i]) 
    plt.hist(effect[Xname], bins=30, range=(-0.1, 1), edgecolor='black', linewidth=1.2)
    
    if i == 0:
        plt.ylabel('frequency (a.u.)')
    
    xlabel = 'ATE (a.u.)'
    plt.xlabel(xlabel)
    plt.xlim(-0.1, 1)
    plt.xticks(ticks=np.arange(0,1.5,0.5))
    
    plt.title('$\mathbb{E}[Y|~do(X_{%d}=1)]-\mathbb{E}[Y|~do(X_{%d}=0)]$'%(i,i), fontsize=15)

plt.savefig(fig_path + 'ATE_root_cause_graph_%02d.png'%dataset, bbox_inches="tight")
plt.savefig(fig_path + 'ATE_root_cause_graph_%02d.svg'%dataset, bbox_inches="tight")
plt.show()
#%% Do root cause analysis
##### Root cause X1, coeff beta10
np.random.seed(1000)

z_list     = np.arange(0.0, 10.5, 0.5)
z_list[0]  = 0.1
coeff_list = np.arange(0.5, 3.5, 0.5)

accuracy1       = np.zeros((len(z_list), len(coeff_list)))
attribute_error = np.zeros((len(z_list), len(coeff_list)))

z_i       = 1
coeff_idx = (1, 0)
for j, coeff in enumerate(coeff_list):
    amat = deepcopy(amat_true[0,:,:])
    amat[coeff_idx] = coeff
    print(amat[coeff_idx])
    
    for i, z_scale in enumerate(z_list):
        out = Parallel(n_jobs=-1, verbose=6)(
                         delayed(do_root_cause_analysis)
                             (Nnode, Ntrain, amat, causal_order, z_i, z_scale, num_distribution_samples)
                                 for k in range(50))
        
        correct = np.array([res[0] for res in out])    
        Np      = np.sum(correct==1)
        accuracy1[i,j] = Np/len(correct)
        
        err = np.array([res[1] for res in out])
        attribute_error[i,j] = err.mean()
        
        print('(Z = %.1f, beta = %.1f) acc = %.4f'%(z_scale, coeff, Np/len(correct)))
########### plot result
#%%
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
#%%
##### Root cause X0, coeff beta10
np.random.seed(1000)
accuracy2       = np.zeros((len(z_list), len(coeff_list)))
attribute_error = np.zeros((len(z_list), len(coeff_list)))

z_i       = 0
coeff_idx = (1, 0)
for j, coeff in enumerate(coeff_list):
    amat = deepcopy(amat_true[0,:,:])
    amat[coeff_idx] = coeff
    print(amat[coeff_idx])
    
    for i, z_scale in enumerate(z_list):
        out = Parallel(n_jobs=-1, verbose=6)(
                          delayed(do_root_cause_analysis)
                              (Nnode, Ntrain, amat, causal_order, z_i, z_scale, num_distribution_samples)
                                  for k in range(50))
        
        correct = np.array([res[0] for res in out])    
        Np      = np.sum(correct==1)
        accuracy2[i,j] = Np/len(correct)
        
        err = np.array([res[1] for res in out])
        attribute_error[i,j] = err.mean()
        
        print('(Z = %.1f, beta = %.1f) acc = %.4f'%(z_scale, coeff, Np/len(correct)))
########### plot result
#%%
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
##### Root cause X1, coeff beta31
accuracy3       = np.zeros((len(z_list), len(coeff_list)))
attribute_error = np.zeros((len(z_list), len(coeff_list)))

z_i       = 1
coeff_idx = (3, 1)
for j, coeff in enumerate(coeff_list):
    amat = deepcopy(amat_true[0,:,:])
    amat[coeff_idx] = coeff
    print(amat[coeff_idx])
    
    for i, z_scale in enumerate(z_list):
        out = Parallel(n_jobs=-1, verbose=6)(
                          delayed(do_root_cause_analysis)
                              (Nnode, Ntrain, amat, causal_order, z_i, z_scale, num_distribution_samples)
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