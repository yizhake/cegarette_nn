import operator
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .basic_property import BasicConstraint, BasicProperty, LowerBound
from ..datastructures import BiasTable, Network, NeuronId, WeightsTable


@dataclass
class AdversarialProperty:
    input_constraints: List[BasicConstraint]
    output_constraints: List[BasicConstraint]
    minimal_is_the_winner: bool
    slack: float = 0.0


def find_winner_and_runnerup(
    property: AdversarialProperty,
) -> Tuple[NeuronId, NeuronId]:
    """
    this function check which nodes are the winner (minimal/maximal) and runner_up (the
    second minimal/maximal) a.t. specific test_property
    """
    cmp = operator.lt if property.minimal_is_the_winner else operator.gt
    winner = runnerup = (float("inf") if property.minimal_is_the_winner else float("-inf"), None)
    lowerbounds = [c for c in property.output_constraints if isinstance(c, LowerBound)]
    for lowerbound in lowerbounds:
        current = (lowerbound.lower, lowerbound.nid)
        if cmp(current, winner):
            runnerup = winner
            winner = current
        elif cmp(current, runnerup):
            runnerup = current
    return winner[1], runnerup[1]  # type: ignore


ADVERSARIAL_PROPERTY_OUTPUT_NEURON_ID = NeuronId("c")


def prepare_network_adversarial(network: Network, property: AdversarialProperty) -> Tuple[Network, BasicProperty]:
    if not isinstance(property, AdversarialProperty):
        raise TypeError("property should be of AdversarialProperty")

    winner, runnerup = find_winner_and_runnerup(property)

    last_layer_weights = network.weights[-1]
    last_layer_biases = network.biases[-1]

    output_neuron = ADVERSARIAL_PROPERTY_OUTPUT_NEURON_ID

    if property.minimal_is_the_winner:
        # the property is to find: y_winner > y_runnerup, which is equivalent to:
        #    y_winner - y_runnerup > slack
        #    sum(x * w_winner) + b_winner - sum(x * w_runnerup) - b_runnerup > slack
        #    sum(x * (w_winner - w_runnerup)) + (b_winner - b_runnerup) > slack
        #    sum(x * (w_winner - w_runnerup)) + (b_winner - b_runnerup) - slack > 0
        new_output_layer_weights = WeightsTable(
            pd.DataFrame({output_neuron: (last_layer_weights.table[winner] - last_layer_weights.table[runnerup])})
        )
        new_output_layer_biases = BiasTable(
            {output_neuron: last_layer_biases[winner] - last_layer_biases[runnerup] - property.slack}
        )
    else:
        # the property is to find: y_runnerup > y_winner, which is equivalent to:
        #    y_runnerup - y_winner > slack
        #    sum(x * w_runnerup) + b_runnerup - sum(x * w_winner) - b_winner > slack
        #    sum(x * (w_runnerup - w_winner)) + (b_runnerup - b_winner) > slack
        #    sum(x * (w_runnerup - w_winner)) + (b_runnerup - b_winner) - slack > 0
        new_output_layer_weights = WeightsTable(
            pd.DataFrame({output_neuron: (last_layer_weights.table[runnerup] - last_layer_weights.table[winner])})
        )
        new_output_layer_biases = BiasTable(
            {output_neuron: last_layer_biases[runnerup] - last_layer_biases[winner] - property.slack}
        )

    new_network = Network(
        weights=[*network.weights[:-1], new_output_layer_weights],
        biases=[*network.biases[:-1], new_output_layer_biases],
        activations=network.activations.copy(),
    )
    new_property = BasicProperty(
        property.input_constraints,
        [LowerBound(output_neuron, 0.0)],
    )
    return new_network, new_property
