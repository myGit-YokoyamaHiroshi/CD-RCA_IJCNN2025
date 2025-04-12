# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:29:31 2024
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
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.models import  Prediction
import sklearn
from joblib import Parallel, delayed
# from statsmodels.tsa.api import VAR
## カスタムスクリプトのパス
# sys.path.append(os.getcwd() + "/custom_lib/")
# from convert_to_dowhy import * #カスタムスクリプトを読み込む
from my_modules.myfunctions import make_lagged_time_label
from my_modules.myfunctions import generate_nonlinear_VAR_data
from my_modules.myfunctions import do_root_cause_analysis
from my_modules.myfunctions import convert_tigramite_to_dowhy
from my_modules.myfunctions import run_pcmci_boot

from copy import deepcopy
from my_modules.baseline_method import LIME, z_score, EIG_vec, gpa_map_gaussian, gpa_map
#%%

#%% [1] Set parameter settings
np.random.seed(1000)
###########################
N       = 4
tau_max = 1
tau_min = 1
z_mean  = 30
num_distribution_samples = 50000

fig_path = current_path + '/figures/comparison_nonlinear/z_%03.1f/'%z_mean
if os.path.exists(fig_path)==False:  # Make the directory for figures
    os.makedirs(fig_path)
    
save_path = current_path + '/save_data/nonlinear/z_%03.1f/'%z_mean
if os.path.exists(save_path)==False:  # Make the directory for figures
    os.makedirs(save_path)
save_name   = 'RCA_result' 

fullpath_save = save_path + save_name 

if os.path.exists(fullpath_save + '.npy') == False:
    links_coeffs = {
                    0: {(0, -1): 0.8,
                        },
                    1: {(0, -1): 3.8,
                        (2, -1): 0.8,
                        },
                    2: {(2, -1): 0.8,
                        },
                    3: {
                        (1, -1): 3.8
                        },
                    }
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
    
    Gnx_lag, amat_lag    = convert_tigramite_to_dowhy(graph_true, link_true)
    
    #%% [2] Generate synthetic data
    T_spurious   = 20
    Nnode        = link_true.shape[0]
    Npredictor   = 5000 
    Npcmci       = 5000 + tau_max
    Ntrain       = Npredictor + Npcmci 
    # Nx           = np.random.normal(loc=0, scale=np.sqrt(0.05), size=(Ntrain+T_spurious,Nnode))
    Nx           = np.random.uniform(low=0, high=1, size=(Ntrain+T_spurious,Nnode))
    
    data_sim   = generate_nonlinear_VAR_data(amat_true, Ntrain, Nx,
                                             T_spurious = T_spurious)
    #%% [3] Learn predictor (linear regression model)
    #### make dataframe
    dataframe = pp.DataFrame(data_sim[:Npredictor,:], var_names = [r'$X_0$', r'$X_1$', r'$X_2$', r'$Y$'])
    #### Define class module function of prediction model
    pred = Prediction(dataframe=dataframe,
                      cond_ind_test=ParCorr(),
                      prediction_model = sklearn.linear_model.LinearRegression(),
                      train_indices    = range(0, int(0.8*Npredictor)),
                      test_indices     = range(int(0.8*Npredictor), Npredictor),
                      verbosity=0)
    
    target  = N - 1
    ##### Run PC algorithm for feature selection
    predictors = pred.get_predictors(
                      selected_targets=[target],
                      steps_ahead=1,
                      tau_max=tau_max,
                      pc_alpha=None
                      )
    ##### fit the model with link assumption
    pred.fit(target_predictors=predictors, 
             selected_targets=[target],
             tau_max=tau_max)
    
    predicted = pred.predict(target)
    true_data = pred.get_test_array(target)[0]
    
    ##############################################################################
    #%% [4] Learn causal structure with pcmci + bootstrap
    data_graph_learn = data_sim[Npredictor:,:]
    newframe      = pp.DataFrame(data_graph_learn, var_names = [r'$X_0$', r'$X_1$', r'$X_2$', r'$Y$'])
    predicted     = pred.predict(target, new_data=newframe)
    
    error         = data_graph_learn[tau_max:, target] - predicted 
    # error         = predicted - data_graph_learn[tau_max:, target] 
    ######## store prediction error to data matrix
    data_with_err = np.concatenate((error[:,np.newaxis], data_graph_learn[tau_max:,:]), axis=1)
    var_names     = [r'$Err.$', r'$X_0$', r'$X_1$', r'$X_2$', r'$Y$']
    dataframe_e   = pp.DataFrame(data_with_err, var_names = var_names)
    
    ###### define link assumption
    p = data_with_err.shape[1]
    error_number = 0
    _vars = list(range(p))
    link_assumptions = {}
    for j in _vars:
        link_assumptions[j] = {}
        for i in _vars:
            for lag in range(tau_min, tau_max + 1):
                if not (i == j):
                    if (j==0) or (j >= 1 and i !=0):
                        link_assumptions[j][(i, -lag)] = '-?>'
                        
    print(link_assumptions)
    
    ################################################################################
    ###### run pcmci with bootstrap ###########################
    ################################################################################
    # Nlist = [5000]
    Nlen  = Npcmci - tau_max
    out   = []
    
    pc_alpha_bootstrap = 0.5
    Ntri         = 50
    boot_samples = 500
    block_length = 1
    ## Create PCMCI object to call run_bootstrap_of
    method_args={
                 'tau_min':0, 
                 'tau_max':tau_max, 
                 'pc_alpha':pc_alpha_bootstrap,
                 'link_assumptions':link_assumptions
                }
    
    pcmci = PCMCI(
            dataframe=pp.DataFrame(data_with_err[:Nlen,:], var_names = var_names),
            cond_ind_test=RobustParCorr(),#ParCorr(),
            verbosity=0
            )
    
    out = Parallel(n_jobs=-1, verbose=3)(
                    delayed(run_pcmci_boot)
                    (pcmci, method_args, boot_samples, block_length)
                    for tri in range(Ntri))
    
    ### Store estimation results
    graph_est = np.array([res['summary_results']['most_frequent_links'] for res in out])
    coeff_est = np.array([res['summary_results']['val_matrix_mean'] for res in out])
    ###############################################################################
    #%% [5] Generate anomaly data (noise distorted time-series) for RCA
    ###############################################################################
    att_val_all = []
    Xnames      = ['Err', 'X0', 'X1', 'X2', 'Y']
    Xnames_lag  = make_lagged_time_label(Xnames, 1, tau_max)
    
    target_node = 'Err at t'
    z_i         = 0
    
    anomaly     = 500
    
    amat        = deepcopy(amat_true)
        
    data_err_list  = []
    anomaly_sample = []
    for tri in range(Ntri):
        #### Make unobserved value Z
        Z          = np.zeros((Npcmci+T_spurious, Nnode))
        Z[anomaly+T_spurious, z_i] += z_mean 
        # Z[520:620, z_i] += z_mean 
        #### Draw new observation distorted by Z 
        
        Nx         = np.random.uniform(low=0, high=1, size=(Npcmci+T_spurious,Nnode))
        data_new   = generate_nonlinear_VAR_data(amat, Npcmci, Nx, 
                                                 T_spurious = T_spurious, 
                                                 confounder=True, Z=Z)
        ##### Apply prediction model to distorted observation
        #### evaluate prediction error
        pred_target = Nnode - 1 
        newframe    = pp.DataFrame(data_new, var_names = Xnames[1:])
        predicted   = pred.predict(pred_target, new_data=newframe)
        ##### Calculate prediction error
        error       = predicted - data_new[tau_max:, pred_target] 
        data_err    = np.concatenate((error[:,np.newaxis], data_new[tau_max:,:]), axis=1)
        
        data_err_list.append(data_err)
        # anomaly_sample = abs(error).argmax()  
        idx_tmp     = np.arange(anomaly, anomaly + tau_max + 1)
        err_sample  = idx_tmp[abs(error[idx_tmp]).argmax()]
        
        anomaly_sample.append(err_sample)
    ################################################################################
    #%% [6] Conduct root-cause analysis (our proposed method) ######################
    ################################################################################
    att_val_all = np.zeros((Nnode+1, Ntri))
    
    
    results = Parallel(n_jobs=-1, verbose=3)(
                      delayed(do_root_cause_analysis)
                      (Nnode, Nlen, data_err_list[tri], 
                       graph_est[tri,:,:,:], 
                       coeff_est[tri,:,:,:], 
                       Xnames, Xnames_lag, 
                       target_node, 
                       anomaly_sample[tri], 
                       tau_max,
                       num_distribution_samples= num_distribution_samples)
                      for tri in range(Ntri))
    
    for tri in range(Ntri):
        attribution   = results[tri][0]
        keys          = results[tri][1]
        anomaly_score = results[tri][2]
        
        for n, name in enumerate(Xnames):
            check   = np.array([name in key for key in keys])
            idx_att = np.where(check==True)[0]
            
            att_val_all[n, tri] = attribution[idx_att].sum()
    ###############################################################################
    #%% [7] Conduct root-cause analysis with baseline method ######################
    ###############################################################################
    #### LIME
    for tri in range(Ntri):
        att_val_lime = np.zeros((Nnode+1, Ntri))
        n_test       = [anomaly_sample[tri]]
        
        data_new     = data_err_list[tri][:,1:]
        
        x_test       = data_new[[n_test[0]-tau_max],:]
        y_test       = np.array(data_new[n_test,3])
        
        lime_score,_ = LIME(x_test, y_test, pred, pred_target, var_names=Xnames[1:], 
                            N_grad=10, eta=1, l1=0.2, seed=None)
        att_val_lime[1:, tri] = lime_score
    
    #### EIG
    att_val_eig = np.zeros((Nnode+1, Ntri))
    for tri in range(Ntri):
        n_test       = [anomaly_sample[tri]]
        data_new     = data_err_list[tri][:,1:]
        x_test       = data_new[[n_test[0]-tau_max],:]
        
        eig_score  = EIG_vec(x_test, pred, data_graph_learn, pred_target, 
                            Xnames[1:], tau_max, seed=None)
        att_val_eig[1:, tri] = eig_score
    
    #### z_score
    att_val_z = np.zeros((Nnode, Ntri))
    for tri in range(Ntri):
        n_test       = [anomaly_sample[tri]]
        data_new     = data_err_list[tri][:,1:]
        x_test       = data_new[n_test,:]
        
        zvalue    = z_score(x_test,
                            data_graph_learn).reshape(-1)
        
        att_val_z[:,tri] = zvalue
    
    ###### LC method: GPA with Gaussian observation and elastic-net prior
    att_val_LC = np.zeros((Nnode+1, Ntri))
    var_names  = Xnames[1:]
    for tri in range(Ntri):
        n_test     = [anomaly_sample[tri]]
        data_new   = data_err_list[tri][:,1:]
        error      = data_err_list[tri][:,0]
        
        stddev_yf  = np.std(error) # estimate the std of the predictor
        x_test     = data_new[(n_test[0]-tau_max-1):(n_test[0]),:]
        y_test     = np.array(data_new[n_test,3])
        
        LC_score,_ = gpa_map_gaussian(x_test, y_test, pred,
                                      stddev_yf, var_names, pred_target,
                                      seed_initialize=None,
                                      seed_grad = None)
        #print(LC_score)
        att_val_LC[1:, tri] = LC_score
        
    ##### GPA method
    att_val_GPA = np.zeros((Nnode+1, Ntri))
    var_names   = Xnames[1:]
    
    for tri in range(Ntri):
        data_new   = data_err_list[tri][:,1:]
        error      = data_err_list[tri][:,0]
        n_test     = [anomaly_sample[tri]]
        
        stddev_yf  = np.std(error) # estimate the std of the predictor
        x_test     = data_new[(n_test[0]-tau_max-1):(n_test[0]),:]
        y_test     = np.array(data_new[n_test,3])
        a          = 1
        b          = stddev_yf * a
        
        GPA_score,_ =  gpa_map(x_test, y_test, pred, a, b,
                               var_names, pred_target,
                               seed_initialize=None,
                               seed_grad = None)
        att_val_GPA[1:, tri] = GPA_score
    
    ###############################################################################
    #%% [8] Save all results ######################################################
    ###############################################################################
    save_dict                 = {} 
    ########## training data
    save_dict['graph_true']     = graph_true
    save_dict['link_true']      = link_true
    save_dict['data_sim']       = data_sim
    save_dict['tau_max']        = tau_max
    save_dict['tau_min']        = tau_min
    save_dict['Nnode']          = Nnode
    save_dict['Npredictor']     = Npredictor
    save_dict['Npcmci']         = Npcmci
    save_dict['Ntrain']         = Ntrain
    save_dict['pred_target']    = target
    ######## predictor
    save_dict['pred']           = pred # class object (tigramite.models.Prediction )
    save_dict['predictors']     = predictors # learned predictor (model coefficients)
    ######## learned causal graph (pcmci)
    save_dict['dataframe_e']    = dataframe_e # learning data for pcmci boot
    save_dict['pcmci']          = pcmci # class object of PCMCI
    save_dict['graph_est']      = graph_est
    save_dict['coeff_est']      = coeff_est
    save_dict['Ntri']           = Ntri
    save_dict['Nlen']           = Nlen
    save_dict['pc_alpha_boot']  = pc_alpha_bootstrap # alpha value for pcmci boot
    save_dict['boot_samples']   = boot_samples # num. of sample for bootstrap
    save_dict['block_length']   = block_length # block length for bootstrap sampling
    ######## anomaly data
    save_dict['Xnames']         = Xnames
    save_dict['Xnames_lag']     = Xnames_lag
    save_dict['target_rca']     = target_node
    save_dict['z_i']            = z_i # location of unobserved cause Z
    save_dict['z_mean']         = z_mean # amplitude of unobserved cause Z
    save_dict['Z']              = Z # unobserved cause Z
    save_dict['data_new']       = data_new # anomaly observation 
    save_dict['error']          = data_new # prediction error 
    save_dict['data_new_err']   = data_err # anomaly observation with prediction error
    save_dict['anomaly_sample'] = anomaly_sample
    save_dict['amat']           = amat # VAR coefficients to generate anomaly samples

    ######### attribution value (all RCA result)
    save_dict['results']        = results # all results of our proposed method
    save_dict['att_val_all']    = att_val_all ## attribution value of our method
    save_dict['att_val_lime']   = att_val_lime  ## attribution value of LIME
    save_dict['att_val_eig']    = att_val_eig   ## attribution value of EIG
    save_dict['att_val_LC']     = att_val_LC    ## attribution value of LC
    save_dict['att_val_GPA']    = att_val_GPA    ## attribution value of GPA
    save_dict['att_val_z']      = att_val_z  ## attribution value of z-score
    
    np.save(fullpath_save, save_dict)
    #%%
else:
    #%%
    save_dict  = np.load(fullpath_save + '.npy', 
                         encoding='ASCII', 
                         allow_pickle='True').item()
    
    graph_true = save_dict['graph_true'] 
    link_true  = save_dict['link_true'] 
    Nnode      = save_dict['Nnode']
    N          = Nnode
    Npredictor = save_dict['Npredictor']
    Npcmci     = save_dict['Npcmci']
    Ntrain     = save_dict['Ntrain']
    tau_max    = save_dict['tau_max']
    tau_min    = save_dict['tau_min']
    data_sim   = save_dict['data_sim'] 
    target     = save_dict['pred_target']
    pred       = save_dict['pred'] # class object (tigramite.models.Prediction )
    predictors = save_dict['predictors'] # learned predictor (model coefficients)
    ######### pcmci
    dataframe_e = save_dict['dataframe_e'] # learning data for pcmci boot
    pcmci       = save_dict['pcmci'] # class object of PCMCI
    graph_est   = save_dict['graph_est']
    ceoff_est   = save_dict['coeff_est']
    
    Ntri               = save_dict['Ntri']
    Nlen               = save_dict['Nlen']
    pc_alpha_bootstrap = save_dict['pc_alpha_boot']  # alpha value for pcmci boot
    boot_samples       = save_dict['boot_samples'] # num. of sample for bootstrap
    block_length       = save_dict['block_length'] # block length for bootstrap sampling
    ######## anomaly data
    Xnames             = save_dict['Xnames']
    Xnames_lag         = save_dict['Xnames_lag']
    target_node        = save_dict['target_rca']
    z_i                = save_dict['z_i'] # location of unobserved cause Z
    z_mean             = save_dict['z_mean'] # amplitude of unobserved cause Z
    Z                  = save_dict['Z'] # unobserved cause Z
    data_new           = save_dict['data_new'] # anomaly observation 
    error              = save_dict['error'] # prediction error 
    data_err           = save_dict['data_new_err']  # anomaly observation with prediction error
    anomaly_sample     = save_dict['anomaly_sample']
    ######### attribution value (all RCA result)
    results            = save_dict['results'] # all results of our proposed method
    att_val_all        = save_dict['att_val_all']  ## attribution value of our method
    att_val_lime       = save_dict['att_val_lime'] ## attribution value of LIME
    att_val_eig        = save_dict['att_val_eig']  ## attribution value of EIG
    att_val_LC         = save_dict['att_val_LC']   ## attribution value of LC
    att_val_GPA        = save_dict['att_val_GPA']  ## attribution value of GPA
    att_val_z          = save_dict['att_val_z'] ## attribution value of z-score
###############################################################################
#%% [9] Plot all results ######################################################
###############################################################################
#%% Plot training data set 
# visualize exact graph (training data)
tp.plot_time_series_graph(
    graph=graph_true,
    val_matrix=link_true,
    var_names= [r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$'],
    link_colorbar_label='coefficient',
    label_fontsize=16,
    )
plt.savefig(fig_path + 'synthetic_data_time_graph.png', bbox_inches="tight")
plt.savefig(fig_path + 'synthetic_data_time_graph.svg', bbox_inches="tight")
plt.show()
###############################################################################
#%% Plot training data set 

x_train    = data_sim[:Npredictor, :3]
y_train    = data_sim[:Npredictor, 3]

base = np.flipud(np.arange(0, 50*4, 50))

for n in range(4):
    timestamp = np.arange(0, len(y_train))
    if n < 3:
        x = x_train[:,n] - x_train[:,n].mean() 
        plt.plot(timestamp, x + base[n], c='k')
    elif n==3:
        y = y_train - y_train.mean()
        plt.plot(timestamp, y + base[n], c='r')

plt.yticks(np.flipud(base),
            [r'$X_4$', r'$X_3$', r'$X_2$', r'$X_1$'])
plt.xlabel('# sample')
plt.xlim(0,len(y_train))
plt.ylim(-60, 260)
plt.savefig(fig_path + 'synthetic_data_time_series.png', bbox_inches="tight")
plt.savefig(fig_path + 'synthetic_data_time_series.svg', bbox_inches="tight")
plt.show()
###############################################################################
#%% Plot result of learning for time-series predictor

###### visualize estimated regressor
link_matrix    = np.zeros((N, N, tau_max+1))
link_skeleton  = np.zeros((N, N, tau_max+1))
skeleton_graph = np.zeros((N, N, tau_max+1), dtype='object')
for j in [target]:
    for p in predictors[j]:
        link_matrix[p[0], j, abs(p[1])]    = pred.get_coefs()[target][p]
        link_skeleton[p[0], j, abs(p[1])]  = 1
        skeleton_graph[p[0], j, abs(p[1])] = '-->'

tp.plot_time_series_graph(
    graph=skeleton_graph,
    val_matrix=link_skeleton,
    var_names= [r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$'],
    link_colorbar_label='coefficient',
    arrowhead_size=0.1,
    label_fontsize=16,
    cmap_edges="bone_r",
    )
plt.title('lagged regression model\n', fontsize=18)
plt.savefig(fig_path + 'estimated_skeleton.png', bbox_inches="tight")
plt.savefig(fig_path + 'estimated_skeleton.svg', bbox_inches="tight")
plt.show()

tp.plot_time_series_graph(
    graph=skeleton_graph,
    val_matrix=link_matrix,
    var_names= [r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$'],
    link_colorbar_label='coefficient',
    label_fontsize=16,
    vmin_edges=-0.4,
    vmax_edges=+0.4,
    )
plt.title('lagged regression model\n', fontsize=18)
plt.savefig(fig_path + 'estimated_prediction_model.png', bbox_inches="tight")
plt.savefig(fig_path + 'estimated_prediction_model.svg', bbox_inches="tight")
plt.show()
###############################################################################
#%% plot anomaly data (noise distorted data)

x_anomaly = data_err[:, 1:4]
y_anomaly = data_err[:, 4]
base      = np.flipud(np.arange(0, 50*4, 50))

for n in range(4):
    timestamp = np.arange(0, len(y_train))
    if n < 3:
        x = x_anomaly[:,n] - x_anomaly[:,n].mean() 
        plt.plot(timestamp, x + base[n], c='k')
    elif n==3:
        y = y_anomaly - y_anomaly.mean()
        plt.plot(timestamp, y + base[n], c='r')

plt.yticks(np.flipud(base),
            [r'$X_4$', r'$X_3$', r'$X_2$', r'$X_1$'])
plt.xlabel('# sample')
plt.xlim(0,len(y_train))
plt.ylim(-60, 260)
fname = 'synthetic_anomaly_time_series'
plt.savefig(fig_path + fname + '.png', bbox_inches="tight")
plt.savefig(fig_path + fname + '.svg', bbox_inches="tight")
plt.show()
###############################################################################
#%% [10] Plot RCA result
###############################################################################
fig = plt.figure(figsize=(6, 4))
labels = []

norm_att  = att_val_all[1:,:]/np.max(abs(att_val_all[1:,:]),axis=0)
norm_lime = att_val_lime[1:,:]/abs(att_val_lime[np.isnan(att_val_lime)==False].reshape(att_val_lime.shape)).max(axis=0)
norm_eig  = att_val_eig[1:,:] /abs(att_val_eig[np.isnan(att_val_eig)==False].reshape(att_val_eig.shape)).max(axis=0)
norm_lc   = att_val_LC[1:,:]/abs(att_val_LC[np.isnan(att_val_LC)==False].reshape(att_val_LC.shape)).max(axis=0)
norm_gpa  = att_val_GPA[1:,:] /abs(att_val_GPA[np.isnan(att_val_GPA)==False].reshape(att_val_GPA.shape)).max(axis=0)
norm_z    = att_val_z/abs(att_val_z[np.isnan(att_val_z)==False].reshape(att_val_z.shape)).max(axis=0)
#### plot result (our proposed method) ########################################
violin   = plt.violinplot(norm_att.T, #att_val_all[i,:,:].T, 
                          positions=np.arange(len(Xnames)-1)*2 - 0.6, 
                          showextrema=True, showmedians=True
                          ) 
[plt.scatter((i*2)*np.ones(Ntri)-0.6 + 0.04*np.random.randn(Ntri), 
             val, 
             color='tab:blue', edgecolors='blue', 
             alpha=0.4, s=15) 
             for i, val in enumerate(norm_att)]
color  = violin["bodies"][0].get_facecolor().flatten()
label  = 'proposed method'
labels.append((mpatches.Patch(color=color), label))
#### plot result (LIME) #######################################################
norm_lime[np.isnan(norm_lime)]=0
violin   = plt.violinplot(norm_lime.T, #att_val_all[i,:,:].T, 
                          positions=np.arange(len(Xnames)-1)*2  - 0.4, 
                          showextrema=True, showmedians=True
                          ) 
[plt.scatter((i*2)*np.ones(Ntri)-0.4 + 0.04*np.random.randn(Ntri), 
             val, 
             color='tab:orange', edgecolors='orange', 
             alpha=0.4, s=15) 
             for i, val in enumerate(norm_lime)]
color  = violin["bodies"][0].get_facecolor().flatten()
label  = 'LIME'
labels.append((mpatches.Patch(color=color), label))

#### plot result (EIG) #######################################################
norm_eig[np.isnan(norm_eig)]=0
violin   = plt.violinplot(norm_eig.T, #att_val_all[i,:,:].T, 
                          positions=np.arange(len(Xnames)-1)*2  - 0.2, 
                          showextrema=True, showmedians=True
                          ) 
[plt.scatter((i*2)*np.ones(Ntri) - 0.2 + 0.04*np.random.randn(Ntri), 
             val, 
             color='tab:green', edgecolors='green', 
             alpha=0.4, s=15) 
             for i, val in enumerate(norm_eig)]
color  = violin["bodies"][0].get_facecolor().flatten()
label  = 'EIG'
labels.append((mpatches.Patch(color=color), label))

#### plot result (LC) #######################################################
norm_lc[np.isnan(norm_lc)]=0
violin   = plt.violinplot(norm_lc.T, #att_val_all[i,:,:].T,
                          positions=np.arange(len(Xnames)-1)*2 ,
                          showextrema=True, showmedians=True
                          )
[plt.scatter((i*2)*np.ones(Ntri) + 0.04*np.random.randn(Ntri),
             val,
             color='tab:red', edgecolors='red',
             alpha=0.4, s=15)
             for i, val in enumerate(norm_lc)]
color  = violin["bodies"][0].get_facecolor().flatten()
label  = 'LC'
labels.append((mpatches.Patch(color=color), label))


#### plot result (GPA) #######################################################
norm_gpa[np.isnan(norm_gpa)]=0
violin   = plt.violinplot(norm_gpa.T, #att_val_all[i,:,:].T,
                          positions=np.arange(len(Xnames)-1)*2 + 0.2,
                          showextrema=True, showmedians=True
                          )
[plt.scatter((i*2)*np.ones(Ntri) + 0.2 + 0.04*np.random.randn(Ntri),
             val,
             color='tab:purple', edgecolors='purple',
             alpha=0.4, s=15)
             for i, val in enumerate(norm_gpa)]
color  = violin["bodies"][0].get_facecolor().flatten()
label  = 'GPA'
labels.append((mpatches.Patch(color=color), label))

##### plot result (z-score) ###################################################
violin   = plt.violinplot(norm_z.T, #att_val_all[i,:,:].T, 
                          positions=np.arange(len(Xnames)-1)*2  + 0.4, 
                          showextrema=True, showmedians=True
                          ) 
[plt.scatter((i*2)*np.ones(Ntri)+ 0.4 + 0.04*np.random.randn(Ntri), 
             val, 
             color='tab:brown', edgecolors='brown', 
             alpha=0.4, s=15) 
             for i, val in enumerate(norm_z)]
color  = violin["bodies"][0].get_facecolor().flatten()
label  = 'z-score'
labels.append((mpatches.Patch(color=color), label))

plt.xticks(ticks=np.arange(0.0, 8.0, 2), 
           labels=[r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$']);
plt.xlabel('variables')
plt.ylabel('attribution')
plt.grid();
plt.legend(*zip(*labels), 
           bbox_to_anchor=(1.05, 1), loc='upper left', 
           borderaxespad=0, fontsize=18)

plt.ylim(-1.1, 1.1)
fname = 'Attribution_sample_effect'
plt.savefig(fig_path + fname + '.png', bbox_inches="tight")
plt.savefig(fig_path + fname + '.svg', bbox_inches="tight")
plt.show();
#%%
acc_att  = [(abs(val).argmax()==0) & ((val==0).sum() != Nnode) for val in norm_att.T]
acc_att  = np.array(acc_att).sum()/len(acc_att)

acc_lime = [(abs(val).argmax()==0) & ((val==0).sum() != Nnode) for val in norm_lime.T]
acc_lime = np.array(acc_lime).sum()/len(acc_lime)

acc_eig = [(abs(val).argmax()==0) & ((val==0).sum() != Nnode) for val in norm_eig.T]
acc_eig = np.array(acc_eig).sum()/len(acc_eig)

acc_lc = [(abs(val).argmax()==0) & ((val==0).sum() != Nnode) for val in norm_lc.T]
acc_lc = np.array(acc_lc).sum()/len(acc_lc)

acc_gpa = [(abs(val).argmax()==0) & ((val==0).sum() != Nnode) for val in norm_gpa.T]
acc_gpa = np.array(acc_gpa).sum()/len(acc_gpa)

acc_z    = [(abs(val).argmax()==0) & ((val==0).sum() != Nnode) for val in norm_z.T]
acc_z    = np.array(acc_z).sum()/len(acc_z)

plt.bar(np.arange(6), np.array([acc_att, acc_lime, acc_eig,acc_lc,acc_gpa, acc_z]), width=.5)
plt.scatter(np.arange(6), np.array([acc_att, acc_lime, acc_eig,acc_lc,acc_gpa, acc_z]), c='k')
plt.ylim(-.1, 1.2)
plt.grid()
plt.ylabel('True positive rate')
plt.xticks(ticks=np.arange(6), labels=['proposed', 'LIME', 'EIG','LC','GPA','z-score'], rotation=45)
fname = 'RCA_performance'
plt.savefig(fig_path + fname + '.png', bbox_inches="tight")
plt.savefig(fig_path + fname + '.svg', bbox_inches="tight")
plt.show()
#%%
save_csv  = current_path + '/save_data/nonlinear/z_%03.1f/RCA_results.csv'%z_mean
col_lab   = ['$\phi_%d$'%(n+1) for n in range(Nnode)]
colmun    = np.hstack(('', col_lab, 'acc'))
line_att  = np.hstack(('CD-RCA(our)', np.median(norm_att, axis=1),  np.array(acc_att)))
line_lime = np.hstack(('LIME',        np.median(norm_lime, axis=1), np.array(acc_lime)))
line_eig  = np.hstack(('EIG',         np.median(norm_eig, axis=1),  np.array(acc_eig)))
line_lc   = np.hstack(('LC',          np.median(norm_lc, axis=1),   np.array(acc_lc)))
line_gpa  = np.hstack(('GPA',         np.median(norm_gpa, axis=1),  np.array(acc_gpa)))
line_z    = np.hstack(('z-score',     np.median(norm_z, axis=1),    np.array(acc_z)))

csv_mat   = np.vstack((line_att, 
                       line_lime,
                       line_eig,
                       line_lc,
                       line_gpa,
                       line_z,))

df = pd.DataFrame(csv_mat, 
                  columns=colmun)
df.to_csv(save_csv, index = False)