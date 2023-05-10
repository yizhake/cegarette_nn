from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..datastructures import (
    BiasTable,
    Network,
    Scaling,
    WeightsTable,
    Sign,
)


def preprocess_layer_pos_neg(
    incoming: WeightsTable,
    outgoing: WeightsTable,
    biases: BiasTable,
    discard_if_no_out_edges=True,
) -> Tuple[WeightsTable, WeightsTable, BiasTable]:
    ids = outgoing.srcs
    assert np.logical_and((incoming.dests == ids).all(), (ids == biases.ids).all())

    # create outgoing
    pos_ids = ids.map(lambda n: replace(n, name=f"{n.name}+", sign=Sign.Pos))
    pos_outgoing_data = np.where(outgoing.table.values >= 0, outgoing.table.values, 0)
    pos_outgoing = pd.DataFrame(
        index=pos_ids, columns=outgoing.dests, data=pos_outgoing_data
    )

    neg_ids = ids.map(lambda n: replace(n, name=f"{n.name}-", sign=Sign.Neg))
    neg_outgoing_data = np.where(outgoing.table.values <= 0, outgoing.table.values, 0)
    neg_outgoing = pd.DataFrame(
        index=neg_ids, columns=outgoing.dests, data=neg_outgoing_data
    )

    new_outgoing = pd.concat([pos_outgoing, neg_outgoing], axis=0)

    # create incoming
    pos_incoming = incoming.table.copy()
    pos_incoming.columns = pos_ids
    neg_incoming = incoming.table.copy()
    neg_incoming.columns = neg_ids
    new_incoming = pd.concat([pos_incoming, neg_incoming], axis=1)

    # create biases
    pos_biases = biases.table.copy()
    pos_biases.columns = pos_ids
    neg_biases = biases.table.copy()
    neg_biases.columns = neg_ids
    new_biases = pd.concat([pos_biases, neg_biases], axis=1)

    if discard_if_no_out_edges:
        all_outgoing_are_zeros = ~new_outgoing.any(1)
        all_zeros_neurons = new_outgoing[all_outgoing_are_zeros].index
        new_outgoing = new_outgoing.drop(all_zeros_neurons, axis=0)
        new_incoming = new_incoming.drop(all_zeros_neurons, axis=1)
        new_biases = new_biases.drop(all_zeros_neurons, axis=1)

    new_incoming = WeightsTable(new_incoming)
    new_outgoing = WeightsTable(new_outgoing)
    new_biases = BiasTable(new_biases)

    return new_incoming, new_outgoing, new_biases


def preprocess_network_pos_neg(
    network: Network, layer_indices: Optional[Sequence[int]] = None
) -> Network:
    if layer_indices is None:
        # we don't split first and last layers
        to_split_indices = np.arange(2, network.layers_count() - 1, dtype=int)
    else:
        to_split_indices = layer_indices

    new_weights = network.weights.copy()
    new_biases = network.biases.copy()
    new_activations = network.activations.copy()

    for i in to_split_indices[::-1]:

        new_weights[i - 1], new_weights[i], new_biases[i] = preprocess_layer_pos_neg(
            new_weights[i - 1], new_weights[i], new_biases[i]
        )

    return Network(new_weights, new_biases, new_activations)


def preprocess_layer_inc_dec(
    incoming: WeightsTable,
    outgoing: WeightsTable,
    biases: BiasTable,
    discard_if_no_out_edges=True,
) -> Tuple[WeightsTable, WeightsTable, BiasTable]:
    ids = outgoing.srcs
    assert np.logical_and((incoming.dests == ids).all(), (ids == biases.ids).all())

    # calculate increasing edges
    inc_pos = np.outer(
        outgoing.dests.map(lambda n: n.scaling == Scaling.Inc),
        ids.map(lambda n: n.sign == Sign.Pos),
    )
    dec_neg = np.outer(
        outgoing.dests.map(lambda n: n.scaling == Scaling.Dec),
        ids.map(lambda n: n.sign == Sign.Neg),
    )
    increasing_edges = np.logical_or(inc_pos, dec_neg).astype(bool).T

    # calculate decreasing edges
    inc_neg = np.outer(
        outgoing.dests.map(lambda n: n.scaling == Scaling.Inc),
        ids.map(lambda n: n.sign == Sign.Neg),
    )
    dec_pos = np.outer(
        outgoing.dests.map(lambda n: n.scaling == Scaling.Dec),
        ids.map(lambda n: n.sign == Sign.Pos),
    )
    decreasing_edges = np.logical_or(inc_neg, dec_pos).astype(bool).T

    # create outgoing
    inc_ids = ids.map(lambda n: replace(n, name=f"{n.name}I", scaling=Scaling.Inc))
    inc_outgoing_data = np.where(increasing_edges, outgoing.table.values, 0)
    inc_outgoing = pd.DataFrame(
        index=inc_ids, columns=outgoing.dests, data=inc_outgoing_data
    )

    dec_ids = ids.map(lambda n: replace(n, name=f"{n.name}D", scaling=Scaling.Dec))
    dec_outgoing_data = np.where(decreasing_edges, outgoing.table.values, 0)
    dec_outgoing = pd.DataFrame(
        index=dec_ids, columns=outgoing.dests, data=dec_outgoing_data
    )

    new_outgoing = pd.concat([inc_outgoing, dec_outgoing], axis=0)

    # create incoming
    inc_incoming = incoming.table.copy()
    inc_incoming.columns = inc_ids
    dec_incoming = incoming.table.copy()
    dec_incoming.columns = dec_ids
    new_incoming = pd.concat([inc_incoming, dec_incoming], axis=1)

    # create biases
    inc_biases = biases.table.copy()
    inc_biases.columns = inc_ids
    dec_biases = biases.table.copy()
    dec_biases.columns = dec_ids
    new_biases = pd.concat([inc_biases, dec_biases], axis=1)

    if discard_if_no_out_edges:
        all_outgoing_are_zeros = ~new_outgoing.any(1)
        all_zeros_neurons = new_outgoing[all_outgoing_are_zeros].index
        new_outgoing = new_outgoing.drop(all_zeros_neurons, axis=0)
        new_incoming = new_incoming.drop(all_zeros_neurons, axis=1)
        new_biases = new_biases.drop(all_zeros_neurons, axis=1)

    new_incoming = WeightsTable(new_incoming)
    new_outgoing = WeightsTable(new_outgoing)
    new_biases = BiasTable(new_biases)

    return new_incoming, new_outgoing, new_biases


def preprocess_network_inc_dec(
    network: Network, layer_indices: Optional[Sequence[int]] = None
) -> Network:
    if layer_indices is None:
        # we don't split first and last layers
        to_split_indices = np.arange(2, network.layers_count() - 1, dtype=int)
    else:
        to_split_indices = layer_indices

    new_weights = network.weights.copy()
    new_biases = network.biases.copy()
    new_activations = network.activations.copy()

    # output layer is always `increasing`, we need to set it manually
    new_output_ids = new_weights[-1].dests.map(
        lambda n: replace(n, scaling=Scaling.Inc)
    )
    new_weights[-1] = new_weights[-1].copy()
    new_weights[-1].table.columns = new_output_ids
    new_biases[-1] = new_biases[-1].copy()
    new_biases[-1].table.columns = new_output_ids

    for i in to_split_indices[::-1]:

        new_weights[i - 1], new_weights[i], new_biases[i] = preprocess_layer_inc_dec(
            new_weights[i - 1], new_weights[i], new_biases[i]
        )

    return Network(new_weights, new_biases, new_activations)


def preprocess(
    network: Network, layer_indices: Optional[Sequence[int]] = None
) -> Network:
    network = preprocess_network_pos_neg(network, layer_indices)
    network = preprocess_network_inc_dec(network, layer_indices)
    return network
