from itertools import groupby, islice, repeat
from redesign.abstraction.abstraction import AbstractionStep
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

from more_itertools import flatten

from ..datastructures import Network, NeuronId
from .refine import LayerIndex, RefinementStep, Step
from ..utils.neuronid_generator import generate_neuronid

NeuronName = Union[NeuronId, str]


def construct_steps(
    steps_info: Dict[Tuple[LayerIndex, NeuronId], List[NeuronId]]
) -> Iterable[Step]:
    steps = [
        (l, RefinementStep(abstraced, parts=originals))
        for (l, abstraced), originals in steps_info.items()
    ]
    return steps


def network_abstracted_neurons(
    network: Network,
) -> Iterable[Tuple[LayerIndex, NeuronId]]:
    # layers participating in the refinement are [2..n-1]
    layers = islice(network.layers, 2, network.layers_count() - 1)
    # all possible nodes in the network
    neurons = flatten(zip(repeat(i), l.nodes) for i, l in enumerate(layers, 2))
    # only abstracted nodes
    abstracted_neurons = filter(lambda n: bool(n[1].original_neurons), neurons)

    return abstracted_neurons


def original_to_abstract_mapping(
    network: Network,
) -> Dict[NeuronId, Tuple[LayerIndex, NeuronId]]:
    original_to_abstract = {}
    for l, a in network_abstracted_neurons(network):
        for o in a.original_neurons:
            original_to_abstract[o] = (l, a)
    return original_to_abstract


def refinement_steps_from_original_nodes(
    original_nodes: Iterable[NeuronId], network: Network
) -> Iterable[Step]:
    original_to_abstract = original_to_abstract_mapping(network)
    abstract_to_refinement_parts = defaultdict(list)
    for o in original_nodes:
        abstract_to_refinement_parts[original_to_abstract[o]].append(o)

    return construct_steps(abstract_to_refinement_parts)


def name_or_autogen(nol: Union[LayerIndex, NeuronName], sign, scaling):
    if isinstance(nol, LayerIndex):
        return generate_neuronid(nol, sign, scaling)
    elif isinstance(nol, str):
        return NeuronId(nol, sign, scaling)
    elif isinstance(nol, NeuronId):
        return nol
    else:
        raise ValueError(f"invalid nol type {type(nol)}")


def make_groups_refinement_step(
    target_neuron: NeuronId,
    parts: Sequence[Tuple[Union[LayerIndex, NeuronName], Sequence[NeuronId]]],
) -> Sequence[Union[RefinementStep, AbstractionStep]]:

    # fully refine the given neuron
    all_parts = [nid for new_neuron in parts for nid in new_neuron[1]]
    refinement_step = RefinementStep(target_neuron, all_parts)

    # create abstraction step for each group, thus making this refine "grouped"
    abstraction_steps = [
        AbstractionStep(name_or_autogen(nid, *parts[0].type), parts)
        for nid, parts in parts
        if len(parts) > 1  # otherwise we don't need abstraction step
    ]

    return (refinement_step, *abstraction_steps)


def grouped_refinement_steps_from_original_nodes(
    original_nodes_grouped: Sequence[Tuple[Optional[NeuronName], Sequence[NeuronId]]],
    network: Network,
) -> Iterable[Step]:
    original_to_abstract = original_to_abstract_mapping(network)
    abstract_to_refinement_parts = defaultdict(list)

    # verify uniqueness of nodes
    _all_nodes = set()
    for _, og in original_nodes_grouped:
        for o in og:
            if o in _all_nodes:
                raise ValueError(f"{o} has more than one occurrence")
            _all_nodes.add(o)

    for _, og in original_nodes_grouped:
        if len(og) < 1:
            raise ValueError("must have at least on neuron in the group")
        if not all(
            original_to_abstract[og[0]] == original_to_abstract[o] for o in og[1:]
        ):
            raise ValueError("grouped neurons must share the same abstract neuron")
        for o in og:
            abstract_to_refinement_parts[original_to_abstract[o]].append(o)

    by_abstracted = lambda t: original_to_abstract[t[1][0]]
    grouped = groupby(
        sorted(original_nodes_grouped, key=by_abstracted), key=by_abstracted
    )

    steps = []
    for (layer_index, abstract), groups in grouped:
        # replace all None new-nid for groups with layer_index
        groups = [
            (maybe_nid if maybe_nid is not None else layer_index, group_neurons)
            for maybe_nid, group_neurons in groups
        ]
        group_steps = make_groups_refinement_step(abstract, groups)
        steps.extend((layer_index, step) for step in group_steps)
    return steps
