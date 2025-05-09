"""This module provides functionality for fitting probabilistic causal models and drawing samples from them.

Functions in this module should be considered experimental, meaning there might be breaking API changes in the future.
"""

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from dowhy.gcm import config
from dowhy.gcm.cms import ProbabilisticCausalModel
from dowhy.gcm.graph import (
    PARENTS_DURING_FIT,
    get_ordered_predecessors,
    is_root_node,
    validate_causal_dag,
    validate_causal_model_assignment,
)


def fit(causal_model: ProbabilisticCausalModel, data: pd.DataFrame):
    """Learns generative causal models of nodes in the causal graph from data.

    :param causal_model: The causal model containing the mechanisms that will be fitted.
    :param data: Observations of nodes in the causal model.
    """
    progress_bar = tqdm(
        causal_model.graph.nodes,
        desc="Fitting causal models",
        position=0,
        leave=True,
        disable=not config.show_progress_bars,
    )
    for node in progress_bar:
        if node not in data:
            raise RuntimeError(
                "Could not find data for node %s in the given training data! There should be a column "
                "containing samples for node %s." % (node, node)
            )

        progress_bar.set_description("Fitting causal mechanism of node %s" % node)

        fit_causal_model_of_target(causal_model, node, data)


def fit_causal_model_of_target(
    causal_model: ProbabilisticCausalModel, target_node: Any, training_data: pd.DataFrame
) -> None:
    """Fits only the causal mechanism of the given target node based on the training data.

    :param causal_model: The causal model containing the target node.
    :param target_node: Target node for which the mechanism is fitted.
    :param training_data: Training data for fitting the causal mechanism.
    :return: None
    """
    validate_causal_model_assignment(causal_model.graph, target_node)

    if is_root_node(causal_model.graph, target_node):
        X_temp = training_data[target_node].to_numpy()
        X_temp = X_temp[~np.isnan(X_temp)]
        causal_model.causal_mechanism(target_node).fit(X = X_temp)
    else:
        X_temp = training_data[get_ordered_predecessors(causal_model.graph, target_node)].to_numpy()
        if len(X_temp.shape) == 1:
            X_temp = X_temp.reshape(-1,1)
        Y_temp = training_data[target_node].to_numpy()
        is_nan_Y = np.isnan(Y_temp)
        is_any_column_nan_X = np.any(np.isnan(X_temp),axis = 1)
        ok_index = ~is_nan_Y
        ok_index[is_any_column_nan_X] = False
        X_temp = X_temp[ok_index]
        Y_temp = Y_temp[ok_index]
        causal_model.causal_mechanism(target_node).fit(
            X= X_temp,
            Y= Y_temp,
        )

    # To be able to validate that the graph structure did not change between fitting and causal query, we store the
    # parents of a node during fit. That way, before sampling, we can verify the parents are still the same. While
    # this would automatically fail when the number of parents is different, there are other more subtle cases,
    # where the number is still the same, but it's different parents, and therefore different data. That would yield
    # wrong results, but would not fail.
    causal_model.graph.nodes[target_node][PARENTS_DURING_FIT] = get_ordered_predecessors(
        causal_model.graph, target_node
    )


def draw_samples(causal_model: ProbabilisticCausalModel, num_samples: int) -> pd.DataFrame:
    """Draws new joint samples from the given graphical causal model. This is done by first generating random samples
    from root nodes and then propagating causal downstream effects through the graph.

    :param causal_model: New samples are generated based on the given causal model.
    :param num_samples: Number of samples to draw.
    :return: A pandas data frame where columns correspond to the nodes in the graph and rows to the drawn joint samples.
    """
    validate_causal_dag(causal_model.graph)

    sorted_nodes = list(nx.topological_sort(causal_model.graph))
    drawn_samples = pd.DataFrame(np.empty((num_samples, len(sorted_nodes))), columns=sorted_nodes)

    for node in sorted_nodes:
        causal_mechanism = causal_model.causal_mechanism(node)

        if is_root_node(causal_model.graph, node):
            drawn_samples[node] = causal_mechanism.draw_samples(num_samples)
        else:
            drawn_samples[node] = causal_mechanism.draw_samples(_parent_samples_of(node, causal_model, drawn_samples))

    return drawn_samples


def _parent_samples_of(node: Any, scm: ProbabilisticCausalModel, samples: pd.DataFrame) -> np.ndarray:
    return samples[get_ordered_predecessors(scm.graph, node)].to_numpy()
