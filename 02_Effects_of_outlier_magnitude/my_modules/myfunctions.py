# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:21:54 2023
"""
import numpy as np
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
###############################################################################
def root_cause_analysis(data, G, tau_max, error_index, target_index):
    ######## data: time-series data with size of (Nt, Nvar)
    ########    G: causal graph (networkx format)
    #### make lagged data
    data_np = np.array(data)
    lagged_data = create_lagged_data(data_np,tau_max = tau_max)
    p = data_np.shape[1]

    lagged_data_pd = pd.DataFrame(lagged_data,columns = np.array(range(p * (tau_max + 1))))
    
    #### put the lagged data and the causal graph into dowhy.gcm and learn the generative model
    causal_model = gcm.StructuralCausalModel(G)
    # Automatically assigns additive noise models to non-root nodes
    gcm.auto.assign_causal_mechanisms(causal_model, lagged_data_pd.iloc[tau_max:,:])
    gcm.fit(causal_model, lagged_data_pd.iloc[tau_max:,:])
    
    target_node      = error_index # 予測誤差の変数
    node_samples, _ = noise_samples_of_ancestors(causal_model, target_node,5000)
    # node_samples, _ = noise_samples_of_ancestors(causal_model, target_node, 1000)
    anomaly_score = []
    attribution_list = []
    # start = time.perf_counter()

    #Shapley値の分解の近似を設定する
    config = ShapleyConfig(approximation_method = ShapleyApproximationMethods.SUBSET_SAMPLING,
                           num_samples = 5000) # defaultは5000。
    
    if target_index == None:
        st  = tau_max
        end = len(data)
    else:
        st  = target_index
        end = target_index + 1
    
    for target_sample in range(st, end):
        anomaly_samples = pd.DataFrame(lagged_data[target_sample:(target_sample+1),:])


        tau_score = MeanDeviationScorer() # 異常メトリック
        IT_score = ITAnomalyScorer(tau_score)

        #異常度合いの計算：
        IT_score.fit(node_samples[target_node].to_numpy())
        anomaly_score.append(IT_score.score(anomaly_samples.to_numpy())[target_node])
        

        # その分解の計算：
        attribution = gcm.attribute_anomalies(causal_model,target_node = target_node,
                                              anomaly_samples=anomaly_samples,
                                              num_distribution_samples = 5000,
                                              anomaly_scorer = tau_score,
                                              shapley_config = config)
        attribution_list.append(attribution)

    return anomaly_score, attribution_list

def make_lagged_time_label(amat_nx, var_name_e, tau_min, tau_max):
    lag_val         = np.arange(tau_min-1, tau_max+1)
    var_names_lag   = []
    Nnode = amat_nx.shape[0]
    Nvar  = Nnode/(tau_max + 1)
    cnt = 0
    lag = 0
    for i in range(Nnode):
        if lag_val[lag] == 0:
            str_name = var_name_e[cnt] + " at t" 
        else:
            str_name = var_name_e[cnt] + " at t-" + str(lag_val[lag]) 
        var_names_lag.append(str_name)
        cnt += 1
        if np.mod(cnt, Nvar)==0:
            cnt = 0
            lag += 1
    
    return var_names_lag

def make_lagged_attribution_graph(Nvar, tau_max, attribution):
    Nmax       = len(attribution)
    graph_root = np.zeros((Nvar, Nvar, tau_max+1), dtype='object')
    link_root  = np.zeros((Nvar, Nvar, tau_max+1))
    
    cnt = 0
    lag = 0
    for n in range(Nmax):
        if attribution[n] != 0:
            if (cnt == 0) & (lag==0):
                graph_root[0, 0, lag] = '-->' 
                link_root[0, 0, lag]  = attribution[n]
            else:
                
                graph_root[cnt, 0, lag] = '-->' 
                link_root[cnt, 0, lag]  = attribution[n]
        
        cnt += 1
        
        if np.mod(cnt, Nvar)==0:
            cnt = 0
            lag += 1
    
    return graph_root, link_root

def get_anomaly_attribution(data_root, graph_info, Nvar, tau_max, tau_min, error_index, t_index):
    G    = graph_info[0]
    amat = graph_info[1]
    
    try:
        anomaly_score, attribution_list = root_cause_analysis(data_root, G, tau_max, error_index, t_index)                                                 
        # shapley_val   = np.array([attribution_list[0][key][0] for key in attribution_list[0].keys()])
        var_names_lag = make_lagged_time_label(amat, var_names, tau_min, tau_max)
        attribution     = np.zeros(len(var_names_lag))
    
        attribution_key = list(attribution_list[0].keys())
        for key in attribution_key:
            attribution[key] = attribution_list[0][key][0]
        
        graph_root, link_root = make_lagged_attribution_graph(Nvar, tau_max, attribution)
        
        result_dict = {}
        result_dict['anomaly_score']    = anomaly_score
        result_dict['var_names_lag']    = var_names_lag
        result_dict['attribution']      = attribution
        result_dict['error_root_graph'] = graph_root
        result_dict['error_root_link']  = link_root
    except ValueError as e:
        print("ValueError occured... Skip this trial")
        result_dict = {}
        result_dict['anomaly_score']    = np.nan
        result_dict['var_names_lag']    = np.nan
        result_dict['attribution']      = np.nan
        result_dict['error_root_graph'] = np.nan
        result_dict['error_root_link']  = np.nan
    
    return result_dict