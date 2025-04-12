# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:59:54 2025
"""
import networkx as nx
import numpy as np
from copy import deepcopy
import matplotlib.pylab as plt

import dowhy.gcm as gcm
from dowhy.gcm.anomaly_scorers import *
from dowhy.gcm._noise import *
from dowhy.gcm.shapley import ShapleyConfig, ShapleyApproximationMethods

#%% metric
def retrieve_adjacency_matrix(graph, order_nodes=None, weight=False):
    """Retrieve the adjacency matrix from the nx.DiGraph or numpy array."""
    if isinstance(graph, np.ndarray):
        return graph
    elif isinstance(graph, nx.DiGraph):
        if order_nodes is None:
            order_nodes = graph.nodes()
        if not weight:
            return np.array(nx.adjacency_matrix(graph, order_nodes, weight=None).todense())
        else:
            return np.array(nx.adjacency_matrix(graph, order_nodes).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")


def SHD(target, pred, double_for_anticausal=True):
    r"""Compute the Structural Hamming Distance.

    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either 
    missing or not in the target graph is counted as a mistake. Note that 
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the 
    `double_for_anticausal` argument accounts for this remark. Setting it to 
    `False` will count this as a single mistake.

    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of 
            ones and zeros.
        prediction (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate.
        double_for_anticausal (bool): Count the badly oriented edges as two 
            mistakes. Default: True
 
    Returns:
        int: Structural Hamming Distance (int).

            The value tends to zero as the graphs tend to be identical.

    Examples:
        >>> from cdt.metrics import SHD
        >>> from numpy.random import randint
        >>> tar, pred = randint(2, size=(10, 10)), randint(2, size=(10, 10))
        >>> SHD(tar, pred, double_for_anticausal=False) 
    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() 
                                            if isinstance(target, nx.DiGraph) else None)

    diff = np.abs(true_labels - predictions)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff)/2

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

def calc_total_effect_numerical(B_est):
    ndim   = B_est.shape[0]
    TE     = deepcopy(B_est)
    Bpower = deepcopy(B_est) 
    
    for i in range(1,ndim+1):
        Bpower = Bpower.dot(B_est) 
        TE    += Bpower
        
    return TE
#%%    
def is_DAG(Adj):
    G = nx.from_numpy_array(Adj, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)

def get_amat_from_tigramite_coeffs(tigramite_coeffs, Nx, tau_max):
    coeffs = tigramite_coeffs
    
    amat   = np.zeros((tau_max+1, Nx,Nx))
    
    for j in list(coeffs):
        for par in list(coeffs[j]):
            i, tau = par
            amat[abs(tau),j,i] = coeffs[j][par]
            
    return amat

def get_link_from_tigramite_coeffs(tigramite_coeffs, Nx, tau_max):
    graph_link = np.zeros((Nx, Nx, tau_max+1), dtype='object')
    graph_link[graph_link==0] = ''
    
    for j in range(Nx):
        for p in tigramite_coeffs[j].keys():
            if tigramite_coeffs[j][p] != 0:
                graph_link[p[0], j, abs(p[1])] = '-->' 
    
    return graph_link


def amat_to_tigramite(amat):
    tau_max,Nx,_ = amat.shape
    
    tigramite_coeffs = {}
    
    graph_link = np.zeros((Nx, Nx, tau_max+1), dtype='object')
    graph_link[graph_link==0] = ''
    
    for tau in range(tau_max):
        for i in range(Nx):
            par = (i, -tau)
            for j in range(Nx):
                if amat[tau,j,i] != 0:
                    
                    if (j in list(tigramite_coeffs.keys())) == False:
                        tigramite_coeffs[j] = {}
                    
                    tigramite_coeffs[j][par] = amat[tau,j,i]               
                    graph_link[i, j, tau] = '-->' 
            
    return tigramite_coeffs, graph_link


def generate_intervened_graph(coeffs, graphs, amat, Ntri, k):
    tau,Nx,_ = amat.shape
    tau_max  = tau - 1
    
    
    amat_copy       = deepcopy(amat)
    amat_copy       = amat_copy + np.diag(np.nan*np.ones(Nx))
    amat_flatten    = deepcopy(amat.flatten())
    
    idx_edge        = np.where((amat_copy.flatten()!=0) & (np.isnan(amat_copy.flatten())==False))[0]
    idx_noedge      = np.where((amat_copy.flatten()==0) & (np.isnan(amat_copy.flatten())==False))[0]
    
    
    coeffs_add = []
    graphs_add = []
    amat_add   = []
    
    coeffs_change = []
    graphs_change = []
    amat_change   = []
    
    # generate graphs with edge-weight intervention
    cnt = 0
    while cnt < 100:
        flag = False
        while flag==False:
            tmp_flatten = deepcopy(amat_flatten)
            perm        = np.random.permutation(len(idx_edge))
            
            idx_target  = idx_edge[perm[0:k]]
            tmp_flatten[idx_target] = np.random.uniform(low=0, high=1, size=k)
            
            amat_tmp = tmp_flatten.reshape(amat.shape)
            
            if is_DAG(amat_tmp[0,:,:])==True:
                coeffs_tmp, graphs_tmp = amat_to_tigramite(amat_tmp)
                
                flag=True
                break
                
        amat_change.append(amat_tmp)
        graphs_change.append(graphs_tmp)
        coeffs_change.append(coeffs_tmp)
        
        cnt += 1
        print('(drop-edge) %d'%(cnt))
    
    # generate graphs with random edge-adding
    cnt = 0
    while cnt < 100:
        flag = False
        while flag==False:
            tmp_flatten = deepcopy(amat_flatten)
            perm        = np.random.permutation(len(idx_noedge))
            idx_target  = idx_noedge[perm[0:k]]
            
            tmp_flatten[idx_target] = np.random.uniform(low=0, high=1, size=k)
            amat_tmp = tmp_flatten.reshape(amat.shape)
            
            if is_DAG(amat_tmp[0,:,:])==True:
                coeffs_tmp, graphs_tmp = amat_to_tigramite(amat_tmp)
                
                flag=True
                break
                
        amat_add.append(amat_tmp)
        graphs_add.append(graphs_tmp)
        coeffs_add.append(coeffs_tmp)
        
        cnt += 1
        
        print('(add-edge) %d'%(cnt))
    
    return amat_add, graphs_add, coeffs_add, amat_change, graphs_change, coeffs_change

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

def generate_noise_distorted_data(Nnode, Ntrain, amat, causal_order, z_i, z_scale, anomaly_sample):
    #### make unobserved value Z
    Z             = np.zeros((Ntrain,Nnode))
    Z[anomaly_sample, z_i] += z_scale#np.sqrt(z_SD) * np.random.randn(100)
    
    #### draw distorted data by Z 
    # Nx             = np.random.uniform(low=0, high=0.5, size=(Ntrain,Nnode))
    Nx             = np.random.uniform(low=0, high=1, size=(Ntrain,Nnode))
    data_err       = generate_data(causal_order, Nnode, Nx, Ntrain, amat, 
                                   confounder=True, Z=Z)
    
    return data_err

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

def do_root_cause_analysis(data_err, z_i, anomaly_sample, amat_rca, 
                           labels, Nnode, target_node = 3, 
                           num_distribution_samples=10000):
    import pandas as pd
    #### put the causal graph into dowhy.gcm
    causal_model   = put_graph_into_dowhy(amat_rca)
    dataframe_err  = pd.DataFrame(data_err, columns = np.array(range(Nnode)))
    anomaly_score, attribution_list = root_cause_analysis(dataframe_err, 
                                                          causal_model, 
                                                          target_node, 
                                                          anomaly_sample,
                                                          num_distribution_samples)
    attribution = attribution_list[0]
    keys        = attribution.keys()
    key_id      = [int(key) for key in keys]
    att_val     = np.array([attribution[key][0] for key in keys])
    att_labels  = np.array(labels)[key_id]
    
    if key_id[att_val.argmax()] == z_i:
        correct = 1
    else:
        correct = 0
    
    return correct, att_val, att_labels



    