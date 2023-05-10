from collections import defaultdict
from itertools import groupby, repeat
from typing import Dict, Iterable, List, Tuple, TypeVar, Union, cast
from random import Random

import numpy as np

from ..datastructures import Network, NeuronId
from .refine import RefinementStep, Step
from .refinement_utils import (
    grouped_refinement_steps_from_original_nodes,
    network_abstracted_neurons,
    original_to_abstract_mapping,
    refinement_steps_from_original_nodes,
)

T = TypeVar("T")
LayerIndex = int


class RandomRefine:
    def __init__(self, max_per_step: int, seed: Union[int, Random] = 1234) -> None:

        if isinstance(seed, int):
            self.rng = Random(seed)
        else:
            self.rng = seed

        self.max_per_step = max_per_step

    def sample(self, iterable: Iterable[T], n: int) -> List[T]:

        reservoir = []
        for t, item in enumerate(iterable):
            if t < n:
                reservoir.append(item)
            else:
                m = self.rng.randint(0, t)
                if m < n:
                    reservoir[m] = item
        return reservoir

    def refinement_step(self, network: Network) -> Iterable[Step]:
        abstracted_neurons = network_abstracted_neurons(network)
        # sample original neurons to refine
        original_to_abstract = {}
        for l, a in abstracted_neurons:
            for o in a.original_neurons:
                original_to_abstract[o] = l, a
        sample = self.sample(original_to_abstract, self.max_per_step)
        # construct the steps
        neurons_to_refine_per_abstracted = defaultdict(list)
        for original in sample:
            abstract_neuron = original_to_abstract[original]
            neurons_to_refine_per_abstracted[abstract_neuron].append(original)
        steps = [
            (l, RefinementStep(abstraced, parts=originals))
            for (l, abstraced), originals in neurons_to_refine_per_abstracted.items()
        ]
        return steps

    def __call__(self, network: Network):
        return self.refinement_step(network)


class RefineByMaxLoss:
    """same as the original refinement strategy used in `core.refinement.refine.refine`"""

    def __init__(self, sequence_length: int = 1) -> None:
        self.sequence_len = sequence_length

    @staticmethod
    def original_edges_weights_diff(network: Network) -> Dict[NeuronId, float]:
        original_edges_weights_diff = defaultdict(float)
        for wt in network.weights:
            owt = wt._orig_table

            for s in wt.srcs:
                for d in wt.dests:
                    w = wt[s, d]
                    # consider only abstracted neurons
                    for os in s.original_neurons:
                        dests = d.original_neurons if d.original_neurons else [d]
                        for od in dests:
                            ow = owt[os, od]
                            original_edges_weights_diff[os] += abs(ow - w)

        return dict(original_edges_weights_diff)

    def __call__(self, network: Network):
        part2loss_map = RefineByMaxLoss.original_edges_weights_diff(network)
        top_part_loss = sorted(part2loss_map.items(), key=lambda x: x[1], reverse=True)
        top_part_loss = top_part_loss[: self.sequence_len]

        # p2l stands for "part2loss", pl[0] is part name
        parts = [p2l[0] for p2l in top_part_loss]

        steps = refinement_steps_from_original_nodes(parts, network)

        return steps


class RefineByMaxActivations:
    def __init__(self, max_per_step: int) -> None:
        self._all_examples = []
        self.max_per_step = max_per_step

    def refinement_step(
        self, network: Network, evaluated_network_activations: Dict[NeuronId, float]
    ) -> Iterable[Step]:
        abstracted_neurons = network_abstracted_neurons(network)
        # select original neurons to refine
        original_to_abstract = {}
        for l, a in abstracted_neurons:
            for o in a.original_neurons:
                original_to_abstract[o] = l, a
        evaluated_network_activations = {
            k: v
            for k, v in evaluated_network_activations.items()
            if k in original_to_abstract
        }
        sorted_by_max_activation = sorted(
            evaluated_network_activations.keys(),
            key=evaluated_network_activations.__getitem__,
            reverse=True,
        ) 
        selected = sorted_by_max_activation[: self.max_per_step]
        # construct the steps
        neurons_to_refine_per_abstracted = defaultdict(list)
        for original in selected:
            abstract_neuron = original_to_abstract[original]
            neurons_to_refine_per_abstracted[abstract_neuron].append(original)
        steps = [
            (l, RefinementStep(abstraced, parts=originals))
            for (l, abstraced), originals in neurons_to_refine_per_abstracted.items()
        ]
        return steps

    def __call__(self, network, evaluated_network_activations):
        return self.refinement_step(network, evaluated_network_activations)



class RefineByMaxLossClustered:
    def __init__(self, sequence_len: int) -> None:
        self.sequence_len = sequence_len

    def cluster(self, data_1d, num_clusters) -> List[int]:
        if len(data_1d) > num_clusters:
            from jenkspy import JenksNaturalBreaks

            jnb = JenksNaturalBreaks(num_clusters)
            jnb.fit(data_1d)
            clusters = jnb.predict(data_1d)
            return clusters  # type: ignore ; I know it's a List[int] like object
        else:
            # not enough clusters, return all differnet clusters
            return list(range(len(data_1d)))

    def __call__(self, network):
        part2loss_map = RefineByMaxLoss.original_edges_weights_diff(network)
        top_part_loss = sorted(part2loss_map.items(), key=lambda x: x[1], reverse=True)
        top_part_loss = top_part_loss[: self.sequence_len]

        if not top_part_loss:
            # no more refinement steps can be done
            return []

        original_to_abstract = original_to_abstract_mapping(network)
        # the number of cluster will be twice the current abstracted nodes we want to refine
        nodes, loss = zip(*top_part_loss)
        nodes = cast(Tuple[NeuronId, ...], nodes)
        num_clusters = len(set(original_to_abstract[n] for n in nodes)) * 2
        clusters = self.cluster(loss, num_clusters)
        clusters = dict(zip(nodes, clusters))

        parts = [(None, list(c)) for _, c in groupby(clusters.keys(), key=clusters.get)]
        parts2: List[Tuple[None, List[NeuronId]]] = []
        for _, p in parts:
            d = defaultdict(list)
            for pp in p:
                d[original_to_abstract[pp]].append(pp)
            parts2.extend(zip(repeat(None), d.values()))

        steps = grouped_refinement_steps_from_original_nodes(
            network=network,
            original_nodes_grouped=parts2,
        )
        return steps

