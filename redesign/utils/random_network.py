from typing import Callable
import numpy as np
from more_itertools import pairwise

import numpy as np
import random

from ..datastructures import (
    ActivationFunction,
    BiasTable,
    Network,
    WeightsTable,
    NeuronId,
)


def random_network(
    num_inputs=2, hidden_layers_num_nodes=[3, 2], num_outputs=1, *, seed=1234
):
    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        # seed is a random generator
        assert isinstance(seed, (np.random.RandomState, random.Random))
        rng = seed
    return random_network_custom(
        num_inputs=num_inputs,
        hidden_layers_num_nodes=hidden_layers_num_nodes,
        num_outputs=num_outputs,
        weights_generator=lambda: rng.randint(-5, 5),
        bias_generator=lambda: rng.randint(-5, 5),
    )


def random_network_custom(
    *,
    num_inputs,
    hidden_layers_num_nodes,
    num_outputs,
    weights_generator: Callable[[], float],
    bias_generator: Callable[[], float],
):

    random_weight = weights_generator
    random_bias = bias_generator

    input_ids = [NeuronId(f"x{i}") for i in range(num_inputs)]

    hidden_neurons = []
    for l, hidden_size in enumerate(hidden_layers_num_nodes, 1):
        hidden_neurons.append(
            [NeuronId(f"v{l}:{i}") for i in range(1, hidden_size + 1)]
        )

    output_ids = [NeuronId(f"y{i}") for i in range(num_outputs)]

    biases = []
    biases.append(BiasTable({n: 0 for n in input_ids}))
    for l in hidden_neurons:
        biases.append(BiasTable({n: random_bias() for n in l}))
    biases.append(BiasTable({n: 0 for n in output_ids}))

    layers = [input_ids, *hidden_neurons, output_ids]

    weights = []
    for l1, l2 in pairwise(layers):
        weights.append(
            WeightsTable({n1: {n2: random_weight() for n2 in l2} for n1 in l1})
        )

    activations = (
        [ActivationFunction.Id]
        + [ActivationFunction.Relu] * (len(biases) - 2)
        + [ActivationFunction.Id]
    )
    network = Network(weights, biases, activations)

    return network
