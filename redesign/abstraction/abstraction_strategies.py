from typing import FrozenSet, Generator, Iterable, Sequence, Tuple
from itertools import groupby

import numpy as np

from ..datastructures import Network, NeuronId, Scaling, Sign, NeuronType
from .abstraction import AbstractionStrategy, Step, AbstractionStep


def group_by_type(
    neuron_ids: Iterable[NeuronId],
) -> Iterable[Tuple[NeuronType, FrozenSet[NeuronId]]]:
    nodes = sorted(neuron_ids, key=lambda n: n.type)
    groups_by_type = groupby(nodes, lambda n: n.type)
    return ((t, frozenset(group)) for t, group in groups_by_type)


def simple_steps_generator(network: Network, layers_order: Iterable[int]):
    neuron_ids = [b.ids for b in network.biases]
    for layer_index in layers_order:
        groups = group_by_type(neuron_ids[layer_index])
        for (sign, scaling), group in groups:
            group = frozenset(group)
            if len(group) < 2:
                # can't abstract less than 2 nodes
                continue
            new_name = f"a{layer_index}"
            new_name += "+" if sign == Sign.Pos else "-"
            new_name += "I" if scaling == Scaling.Inc else "D"
            step = AbstractionStep(new_name, group)
            yield (layer_index, step)


class CompleteAbstractionLeftToRight(AbstractionStrategy):
    def steps(self) -> Generator[Step, Network, None]:
        # start by receiving the initial network
        network = yield
        layers_order = range(2, network.layers_count() - 1)
        yield from simple_steps_generator(network, layers_order)


class CompleteAbstractionRightToLeft(AbstractionStrategy):
    def steps(self) -> Generator[Step, Network, None]:
        # start by receiving the initial network
        network = yield
        layers_order = reversed(range(2, network.layers_count() - 1))
        yield from simple_steps_generator(network, layers_order)


def randomly(seed=None) -> Generator[Step, Network, None]:
    import random

    # start by receiving the initial network
    network = yield

    rng = random.Random(seed)
    layers_order = range(2, network.layers_count() - 1)
    all_steps = simple_steps_generator(network, layers_order)
    all_steps_2 = []
    for layer, step in all_steps:
        indices = [0]
        while indices[-1] != len(step.nodes):
            indices.append(
                rng.randint(min(indices[-1] + 2, len(step.nodes)), len(step.nodes))
            )

        groups = np.split(list(step.nodes), indices[1:-1])
        assert sum(len(g) for g in groups) == len(step.nodes)
        names = [f"{step.new_name.name}:{i}" for i in range(len(groups))]
        new_steps = [
            (layer, AbstractionStep(n, frozenset(g))) for n, g in zip(names, groups)
        ]
        all_steps_2.extend(new_steps)

    rng.shuffle(all_steps_2)
    for step in all_steps_2:
        yield step


def from_explicit_steps(
    steps: Sequence[AbstractionStep],
) -> AbstractionStrategy:
    def gen() -> Generator[Step, Network, None]:
        network = yield  # type: ignore
        for step in steps:
            neuron = next(iter(step.nodes))
            # find the layer index for the neuron
            layer_index = next(
                i
                for i in range(network.layers_count())
                if neuron in network.layers[i].nodes
            )
            yield (layer_index, step)
    return AbstractionStrategy.from_generator_function(gen)
            


if __name__ == "__main__":
    from new_random_network import random_network
    from new_preprocess import preprocess

    net = random_network(1, [1, 10, 10, 10], 1)
    net = preprocess(net)
    steps = randomly()
    next(steps)
    steps.send(net)
    print(*list(steps), sep="\n")
