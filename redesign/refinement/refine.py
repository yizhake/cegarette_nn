from dataclasses import dataclass
import logging
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Sequence,
    Tuple,
    Optional,
    Union,
)

from more_itertools import spy

from redesign.marabou_properties.basic_property import BasicProperty

from ..utils.functions_utils import call_with_needed_arguments
from ..datastructures import (
    BiasTable,
    Network,
    NeuronId,
    WeightsTable,
)
from ..abstraction.abstraction import (
    _abstract_biases,
    _abstract_weights_as_incoming,
    _abstract_weights_as_outgoing,
    AbstractionStep,
    abstract_layer,
)


@dataclass
class RefinementStep:
    target_neuron: NeuronId
    # the neuron parts we want to refine. if None, refine all
    # abstracted neuron under target_neuron
    parts: List[NeuronId] = None


@dataclass
class RefinementStatitistics:
    did_refine: bool
    num_steps: int
    num_neurons_refined: List[int]
    steps: List[RefinementStep]
    neuron_mappings: Dict[NeuronId, NeuronId]


def make_partial_refinement_step(
    target_neuron: NeuronId,
    parts: Sequence[Tuple[Union[str, NeuronId], Sequence[NeuronId]]],
) -> Iterable[Union[RefinementStep, AbstractionStep]]:
    all_parts = [nid for new_neuron in parts for nid in new_neuron[1]]
    refinement_step = RefinementStep(target_neuron, all_parts)
    abstraction_steps = [AbstractionStep(nid, parts) for nid, parts in parts]
    yield refinement_step
    for step in abstraction_steps:
        yield step


def step_without(
    step: AbstractionStep,
    to_discard: Collection[NeuronId],
    new_name: Optional[str] = None,
) -> Optional[AbstractionStep]:
    rest_nodes = [n for n in step.nodes if n not in to_discard]
    if len(rest_nodes) > 1:
        new_name = new_name if new_name is not None else step.new_name  # type: ignore
        return AbstractionStep(new_name, rest_nodes)  # type: ignore
    else:
        return None


def _filter_abstraction_steps(
    weights: WeightsTable, to_discard: Collection[NeuronId], axis: Literal[0, 1]
) -> List[Tuple[Literal[0, 1], AbstractionStep]]:
    # here we use the axis to identify whether the abstraction step
    # was performed on the weights as on sources or destinations neurons.
    # we need it for knowing which steps we should remove and which to keep as is
    to_discard = set(to_discard)
    new_steps = []
    for step_axis, abs_step in weights._abstraction_steps:
        if step_axis == axis:
            if (s := step_without(abs_step, to_discard)) is not None:
                new_steps.append((step_axis, s))
            else:
                # the abstraction step is completely ignored, we should
                # ignore the resulted neuron from this step for the rest
                # of the steps also
                to_discard.add(abs_step.new_name)
        else:
            new_steps.append((step_axis, abs_step))
    return new_steps


def refine_layer(
    incoming: WeightsTable,
    outgoing: WeightsTable,
    biases: BiasTable,
    step: RefinementStep,
) -> Tuple[WeightsTable, WeightsTable, BiasTable]:

    target_neuron: NeuronId = outgoing.table.loc[step.target_neuron].name
    if step.parts:
        new_ids = frozenset(step.parts)
    else:
        new_ids = target_neuron.original_neurons

    incoming_new_steps = _filter_abstraction_steps(incoming, new_ids, axis=1)
    outgoing_new_steps = _filter_abstraction_steps(outgoing, new_ids, axis=0)
    biases_new_steps = _filter_abstraction_steps(biases, new_ids, axis=1)

    new_incoming, new_outgoing, new_biases = (
        incoming._orig_table,
        outgoing._orig_table,
        biases._orig_table,
    )
    for axis, step in incoming_new_steps:
        if axis == 0:
            new_incoming = _abstract_weights_as_outgoing(new_incoming, step)
        else:  # axis == 1
            new_incoming = _abstract_weights_as_incoming(new_incoming, step)

    for axis, step in outgoing_new_steps:
        if axis == 0:
            new_outgoing = _abstract_weights_as_outgoing(new_outgoing, step)
        else:  # axis == 1
            new_outgoing = _abstract_weights_as_incoming(new_outgoing, step)

    for axis, step in biases_new_steps:
        new_biases = _abstract_biases(new_biases, step)

    return new_incoming, new_outgoing, new_biases


LayerIndex = int
Step = Tuple[LayerIndex, Union[RefinementStep, AbstractionStep]]
# when writing a RefinementStrategy function, you may assume to get some
# objects in the call. You may set your function to accept zero or more of
# them, by their name. Only the specified arguments will be passed to your
# implementation.
# (current) possible arguments for RefinementStrategy are:
#   network: Network
#   spurious_example: Mapping[NeuronId, float]
#   evaluated_network_activations: Mapping[NeuronId, float]
#
# example:
#   def my_refinement(activations, network):
#       # note we didn't accept `spurious_example` as an argument,
#       # nor we kept the order, but it still works.
#       ...
#
# NOTE: this design allow us to extend the possibilities of parameters we
#   send to refinement function, without modifying existing code (hopefully),
#   while keeping a clean code where the parameters aren't needed.
#   It is possible to overcome it by requiring adding `**_` to each refinement
#   function, but I think this method is cleaner.
RefinementStrategy = Callable[..., Iterable[Step]]


def refine_network(
    network: Network, refinement_strategy: RefinementStrategy, **refine_kwargs
) -> Tuple[Network, RefinementStatitistics]:

    # TODO: change call to this. need to add **kwargs to existing strategies.
    # steps = refinement_strategy(
    #     network=network, **refine_kwargs
    # )
    steps = call_with_needed_arguments(
        refinement_strategy, network=network, **refine_kwargs
    )
    head, steps = spy(steps)
    if not head:
        # no refinement steps
        stats = RefinementStatitistics(
            did_refine=False,
            num_steps=0,
            steps=[],
            num_neurons_refined=[],
            neuron_mappings={},
        )
        return network, stats

    new_weights = network.weights.copy()
    new_biases = network.biases.copy()
    new_activations = network.activations.copy()

    # statistics data
    num_refine_steps = 0
    steps_for_stats = []
    num_neurons_refined = []
    # each after-refine neuron to before the refinement-steps
    neuron_mappings = {}

    for layer_index, step in steps:
        if isinstance(step, RefinementStep):
            logging.debug(f"refinement step: ({layer_index}, {step})")
            (
                new_weights[layer_index - 1],
                new_weights[layer_index],
                new_biases[layer_index],
            ) = refine_layer(
                new_weights[layer_index - 1],
                new_weights[layer_index],
                new_biases[layer_index],
                step,
            )

            # stats collection
            num_refine_steps += 1
            steps_for_stats.append(step)
            num_neurons = (
                len(step.parts)
                if step.parts is not None
                else len(step.target_neuron.original_neurons)
            )
            num_neurons_refined.append(num_neurons)
            parts = []
            if step.parts is None:
                parts = step.target_neuron.original_neurons
            else:
                parts = step.parts
                if step.target_neuron in new_weights[layer_index].srcs:
                    # the network still contains the target neuron because we didn't fully refined it
                    parts.append(step.target_neuron)
            for part in parts:
                neuron_mappings[part] = step.target_neuron

        elif isinstance(step, AbstractionStep):
            logging.debug(f"abstraction step: ({layer_index}, {step})")
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
        else:
            raise TypeError(
                "`step` must be either a RefinementStep or an AbstractionStep"
            )

    new_network = Network(new_weights, new_biases, new_activations)
    stats = RefinementStatitistics(
        did_refine=True,
        num_steps=num_refine_steps,
        steps=steps_for_stats,
        num_neurons_refined=num_neurons_refined,
        neuron_mappings=neuron_mappings,
    )
    return new_network, stats


def refine_until_not_satisfying(
    network: Network,
    refinement_strategy: RefinementStrategy,
    spurious_examples,
    test_property: BasicProperty,
    update_property: Optional[Callable[[Network], BasicProperty]] = None,
    **refine_kwargs,
) -> Tuple[Network, RefinementStatitistics]:
    """
    same as refine, but stops when the given `spurious_example` is not satisfied on the refined network
    """
    from ..marabou_properties.basic_property import is_satisfying_assignment

    new_network = network
    # statistics data
    num_refine_steps = 0
    steps_for_stats = []
    num_neurons_refined = []
    neuron_mappings = {}

    while True:
        # TODO: change call to this. need to add **kwargs to existing strategies.
        # steps = refinement_strategy(
        #     network=network, **refine_kwargs
        # )
        steps = call_with_needed_arguments(
            refinement_strategy, network=new_network, **refine_kwargs
        )
        head, steps = spy(steps)
        if not head:
            # no refinement steps
            stats = RefinementStatitistics(
                did_refine=False,
                num_steps=0,
                steps=[],
                num_neurons_refined=[],
                neuron_mappings={},
            )
            return new_network, stats

        new_weights = new_network.weights.copy()
        new_biases = new_network.biases.copy()
        new_activations = network.activations.copy()
        new_network = Network(new_weights, new_biases, new_activations)

        for layer_index, step in steps:
            if isinstance(step, RefinementStep):
                logging.debug(f"refinement step: ({layer_index}, {step})")
                (
                    new_weights[layer_index - 1],
                    new_weights[layer_index],
                    new_biases[layer_index],
                ) = refine_layer(
                    new_weights[layer_index - 1],
                    new_weights[layer_index],
                    new_biases[layer_index],
                    step,
                )

                # stats collection
                num_refine_steps += 1
                steps_for_stats.append(step)
                num_neurons = (
                    len(step.parts)
                    if step.parts is not None
                    else len(step.target_neuron.original_neurons)
                )
                num_neurons_refined.append(num_neurons)
                parts = []
                if step.parts is None:
                    parts = step.target_neuron.original_neurons
                else:
                    parts = step.parts
                    if step.target_neuron in new_weights[layer_index].srcs:
                        # the network still contains the target neuron because we didn't fully refined it
                        parts.append(step.target_neuron)
                for part in parts:
                    neuron_mappings[part] = step.target_neuron

            elif isinstance(step, AbstractionStep):
                logging.debug(f"abstraction step: ({layer_index}, {step})")
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
            else:
                raise TypeError(
                    "`step` must be either a RefinementStep or an AbstractionStep"
                )

            if update_property is not None:
                test_property = update_property(new_network)

            for spurious_example in spurious_examples:
                is_satisfying, why = is_satisfying_assignment(
                    new_network, spurious_example, test_property
                )
                if not is_satisfying:
                    stats = RefinementStatitistics(
                        did_refine=True,
                        num_steps=num_refine_steps,
                        steps=steps_for_stats,
                        num_neurons_refined=num_neurons_refined,
                        neuron_mappings=neuron_mappings,
                    )
                    return new_network, stats
