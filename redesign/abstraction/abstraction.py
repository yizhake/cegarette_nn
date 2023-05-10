from dataclasses import dataclass
from typing import Callable, Generator, List, Sequence, Tuple, Union
from abc import abstractmethod

from more_itertools import first
import pandas as pd

from ..datastructures import (
    BiasTable,
    Network,
    NeuronId,
    Scaling,
    WeightsTable,
)


@dataclass(frozen=True, init=False)
class AbstractionStep:
    new_name: NeuronId
    nodes: Sequence[NeuronId]

    def __init__(
        self, new_name: Union[str, NeuronId], nodes: Sequence[NeuronId]
    ):
        if len(nodes) < 2:
            raise ValueError(f"`nodes` must have at least 2 nodes")

        object.__setattr__(self, "nodes", nodes)

        if isinstance(new_name, str):
            sign, scaling = first(self.nodes).type
            new_name_str = new_name

        elif isinstance(new_name, NeuronId):
            if new_name.type != first(self.nodes).type:
                raise ValueError(
                    f"given `new_name` is a NeuronId with different type then it's nodes"
                )
            new_name_str = new_name.name
            sign, scaling = new_name.type

        else:
            raise TypeError("`new_name` must be of `str` type or `NeuronId`")

        new_id = NeuronId(new_name_str, sign, scaling, frozenset(self.nodes))
        object.__setattr__(self, "new_name", new_id)


LayerIndex = int
Step = Tuple[LayerIndex, AbstractionStep]

AbstractionGenerator = Generator[Step, Network, None]


class AbstractionStrategy:
    @abstractmethod
    def steps(self) -> AbstractionGenerator:
        # start by receiving the initial network
        network = yield
        # write strategy logic in a loop, and `yield` the steps.
        # you can if you need the network created after the last abstraction step,
        # you can accept it in the `yield` expression.
        # you may return `None` at any point to stop the abstraction process
        #
        # example:
        #   while True:
        #       # use the network to decide the next step
        #       ...
        #       step = (layer_index, AbstractionStep(...))
        #       network = yield step

    @staticmethod
    def from_generator_function(
        gen_func: Callable[[], AbstractionGenerator]
    ) -> "AbstractionStrategy":
        class AbstractionStrategyFromGenerator(AbstractionStrategy):
            def steps(self) -> AbstractionGenerator:
                steps = gen_func()
                next(steps)
                network = yield
                while True:
                    try:
                        network = yield steps.send(network)
                    except StopIteration:
                        return

        return AbstractionStrategyFromGenerator()


# internal functions for abstracting. no check for validity.
def _abstract_weights_as_incoming(
    incoming: WeightsTable, step: AbstractionStep
) -> WeightsTable:
    scaling = first(step.nodes).scaling
    new_id = step.new_name

    aggregation_func = max if scaling == Scaling.Inc else min

    # aggregate on incoming edges: axis=1
    abstracted_incoming = (
        incoming.table[step.nodes].aggregate(aggregation_func, axis=1).rename(new_id)
    )
    new_incoming = WeightsTable(
        pd.concat(
            [incoming.table.drop(step.nodes, axis=1), abstracted_incoming], axis=1
        ).sort_index(axis=1)
    )
    new_incoming._orig_table = (
        incoming if incoming._orig_table is None else incoming._orig_table
    )
    new_incoming._abstraction_steps.extend(incoming._abstraction_steps)
    new_incoming._abstraction_steps.append((1, step))

    return new_incoming


def _abstract_weights_as_outgoing(
    outgoing: WeightsTable, step: AbstractionStep
) -> WeightsTable:
    new_id = step.new_name

    abstracted_outgoing = (
        outgoing.table.loc[step.nodes].aggregate(sum, axis=0).rename(new_id)
    )
    new_outgoing = WeightsTable(
        pd.concat(
            [outgoing.table.drop(step.nodes, axis=0), abstracted_outgoing.to_frame().T],
            axis=0,
        ).sort_index(axis=0)
    )
    new_outgoing._orig_table = (
        outgoing if outgoing._orig_table is None else outgoing._orig_table
    )
    new_outgoing._abstraction_steps.extend(outgoing._abstraction_steps)
    new_outgoing._abstraction_steps.append((0, step))

    return new_outgoing


def _abstract_biases(biases: BiasTable, step: AbstractionStep) -> BiasTable:
    scaling = first(step.nodes).scaling
    new_id = step.new_name

    aggregation_func = max if scaling == Scaling.Inc else min

    abstracted_biases = (
        biases.table[step.nodes].aggregate(aggregation_func, axis=1).rename(new_id)
    )
    new_biases = BiasTable(
        pd.concat(
            [biases.table.drop(step.nodes, axis=1), abstracted_biases], axis=1
        ).sort_index(axis=1)
    )
    new_biases._orig_table = (
        biases if biases._orig_table is None else biases._orig_table
    )
    new_biases._abstraction_steps.extend(biases._abstraction_steps)
    new_biases._abstraction_steps.append((1, step))

    return new_biases


def abstract_layer(
    incoming: WeightsTable,
    outgoing: WeightsTable,
    biases: BiasTable,
    step: AbstractionStep,
) -> Tuple[WeightsTable, WeightsTable, BiasTable]:
    sign = first(step.nodes).sign
    scaling = first(step.nodes).scaling
    # fmt:off
    assert sign is not None, 'abstract called with not classified node. maybe forgot to call `preprocess`?'
    assert scaling is not None, 'abstract called with not classified node. maybe forgot to call `preprocess`?'
    # fmt:on

    new_incoming = _abstract_weights_as_incoming(incoming, step)
    new_outgoing = _abstract_weights_as_outgoing(outgoing, step)
    new_biases = _abstract_biases(biases, step)

    return new_incoming, new_outgoing, new_biases


def abstract_network(
    network: Network, abstraction_strategy: AbstractionStrategy
) -> Network:
    steps = abstraction_strategy.steps()

    new_weights = network.weights.copy()
    new_biases = network.biases.copy()
    new_activations = network.activations.copy()
    current_network = Network(new_weights, new_biases, new_activations)

    try:
        next(steps)  # starting the generator
        while True:
            layer_index, step = steps.send(current_network)
            if layer_index <= 1 or layer_index >= network.layers_count() - 1:
                raise ValueError(
                    f"all layer indices must be between `2` and `network.layers_count()-2` "
                )

            (
                new_weights[layer_index - 1],
                new_weights[layer_index],
                new_biases[layer_index],
            ) = abstract_layer(
                new_weights[layer_index - 1],
                new_weights[layer_index],
                new_biases[layer_index],
                step,
            )

    except StopIteration:
        # no more steps
        return current_network
