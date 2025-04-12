# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:21:54 2023
"""
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

###############################################################################
def make_lagged_time_label(var_name, tau_min, tau_max):
    lag_val         = np.arange(tau_min-1, tau_max+1)
    var_names_lag   = []
    Nnode = int(len(var_name) * (tau_max+1))
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
        data_sim[t, :] = np.linalg.solve(mixing_matrix, data_temp)
        # data_sim[t, :] = np.linalg.lstsq(mixing_matrix, data_temp)[0]
        
    return data_sim[T_spurious:,:]

def generate_nonlinear_VAR_data(amat, T, exog, T_spurious = 20, confounder=False, Z=None):
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
            
            data_temp += np.dot(amat[jj], 
                                0.7*np.sin(data_sim[t - jj, :])*np.exp(0.1*np.cos(data_sim[t - jj, :])))
        data_sim[t, :] = np.linalg.solve(mixing_matrix, data_temp)
        # data_sim[t, :] = np.linalg.lstsq(mixing_matrix, data_temp)[0]
        
    return data_sim[T_spurious:,:]

def draw_data(data, Gnx_lag, Nsample):
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
    

    st  = target_sample_index
    end = target_sample_index + 1
    
    for target_sample in range(st, end):
        anomaly_samples = dataframe.iloc[target_sample:(target_sample+1),:]

        tau_score = MeanDeviationScorer() # 異常メトリック
        IT_score  = ITAnomalyScorer(tau_score)

        #異常度合いの計算：
        IT_score.fit(node_samples[target_node].to_numpy())
        anomaly_score.append(IT_score.score(anomaly_samples.to_numpy())[target_node])
        
        # その分解の計算：
        attribution = gcm.attribute_anomalies(causal_model,
                                              target_node = target_node,
                                              anomaly_samples=anomaly_samples,
                                              num_distribution_samples = num_distribution_samples,
                                              anomaly_scorer = tau_score,
                                              shapley_config = config)
        attribution_list.append(attribution)

    return anomaly_score, attribution_list


def do_root_cause_analysis(Nnode, Ntrain, data_err, graph, link, 
                           label, lagged_label, target, anomaly_sample, tau_max,
                           num_distribution_samples= 10000):
    
    
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
    
    anomaly_score, attribution_list = root_cause_analysis(dataframe_err, 
                                                          causal_model, 
                                                          target_node[0], 
                                                          anomaly_sample,
                                                          num_distribution_samples)

    attribution = attribution_list[0]
    keys        = attribution.keys()
    att_val     = np.array([attribution[key][0] for key in keys])
    att_key     = [lagged_label[key]  for key in keys]
    
    
    return att_val, att_key, anomaly_score


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

def run_pcmci_boot(pcmci, method_args, boot_samples, boot_blocklength):
    boot_results = pcmci.run_bootstrap_of(
        method='run_pcmci', 
        method_args=method_args, 
        boot_samples=boot_samples,
        boot_blocklength=boot_blocklength)

    return boot_results#, criteria

def create_lagged_data(data,tau_max,**kwargs):
    T = data.shape[0]
    p = data.shape[1]
    if "mask_vector" in kwargs.keys():
        mask_vector = kwargs["mask_vector"]
    else:
        mask_vector = np.full(T, False)

    num_samples = np.sum(mask_vector == False)

    lagged_data = np.full((num_samples,p * (tau_max + 1)),np.nan)

    # each column is 1,...,p, 1 at lag - 1, 2 at lag - 1,..., p at lag -1,...,1 at lag -tau_max,...,p at lag -tau_max
    for i in range(p):
        for tau in range(tau_max + 1):
            data_temp = np.full((num_samples),np.nan)
            index_vec = np.full((num_samples), False)
            index_vec[range(tau, num_samples)] = True
            # only choose the positions that are NOT masked, i.e., mask_vector = FALSE
            index_vec[mask_vector] = False
            data_temp[index_vec] = data[range(T - tau),i]

            lagged_data[:,i +  tau * p] = data_temp
    return lagged_data
