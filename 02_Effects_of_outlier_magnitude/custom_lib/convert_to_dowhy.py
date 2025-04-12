import numpy as np
import networkx as nx
#%%
def convert_tigramite_to_networkx(tigramite_dag):
    # convert the DAG returned by pcmci to the format of networkx used in dowhy
    p       = tigramite_dag.shape[0]
    tau_max = tigramite_dag.shape[2] - 1
    amat    = np.full((p * (tau_max + 1), p * (tau_max + 1)),-1)
    G       = nx.DiGraph()
    for i in range(p * (tau_max + 1)):
        G.add_node(i)
    for i in range(p):
        for j in range(p):
            tau = 0
            if tigramite_dag[i, j, 0] == '-->' and tigramite_dag[j, i, 0] == '<--':
                # i --> j
                amat[i,j] = 1
                G.add_edge(i, j)
            if tigramite_dag[i, j, 0] == '<--' and tigramite_dag[j, i, 0] == '-->':
                # j  --> i
                amat[j,i] = 1
                G.add_edge(j, i)
            if tigramite_dag[i, j, 0] == '' and tigramite_dag[j, i, 0] == '':
                amat[i,j] = 0

            # time-invariance
            for delta_tau in range(1,tau_max + 1):
                amat[i + delta_tau*p, j + delta_tau * p] = amat[i, j]
                amat[j + delta_tau * p, i + delta_tau * p] = amat[j, i]
                if amat[i,j] == 1:
                    G.add_edge(i + delta_tau * p, j + delta_tau * p)
                if amat[j,i] == 1:
                    G.add_edge(j + delta_tau * p, j + delta_tau * p)

            for tau in range(1,tau_max + 1):
                amat[j, i + tau*p] = 0 # arrow of time
                if tigramite_dag[i,j,tau] == '-->':
                    # i-tau --> j
                    amat[i + tau * p,j] = 1
                    G.add_edge(i + tau*p, j)
                if tigramite_dag[i,j,tau] == '':
                    amat[i + tau * p,j] = 0

                # time-invariance
                for delta_tau in range(1,tau_max - tau + 1):
                    if delta_tau > 0:
                        amat[i + (tau + delta_tau) * p, j + delta_tau * p] = amat[i + tau * p, j]
                        amat[j + delta_tau * p, i + (tau + delta_tau) * p] = amat[j, i + tau * p]
                        if amat[i + tau*p,j] == 1:
                            G.add_edge(i + (tau + delta_tau) * p, j + delta_tau * p)
                        if amat[j, i + tau * p] == 1:
                            G.add_edge(j + delta_tau * p, i + (tau + delta_tau) * p)

    return G,amat
#%%
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

