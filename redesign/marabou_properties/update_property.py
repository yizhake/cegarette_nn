from dataclasses import replace
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, overload

from redesign.datastructures import Network
from .basic_property import (
    BasicProperty,
    LowerBound,
    UpperBound,
)
from ..bounds_calculation.propagate_bounds import (
    NeuronBounds,
    OptionalNeuronBounds,
    evaluate_bounds_naive,
    evaluate_bounds_using_marabou_preprocessor,
)
import logging

_INF = 10e38


@overload
def property_to_input_bounds(prop, insert_infs: Literal[False] = ...) -> OptionalNeuronBounds:
    ...


@overload
def property_to_input_bounds(prop, insert_infs: Literal[True]) -> NeuronBounds:
    ...


def property_to_input_bounds(prop: BasicProperty, insert_infs=False):
    if insert_infs:
        bounds: NeuronBounds = defaultdict(lambda: {"min": -_INF, "max": _INF})
    else:
        bounds: NeuronBounds = defaultdict(lambda: {"min": None, "max": None})
    for constraint in prop.input_constraints:
        if isinstance(constraint, LowerBound):
            bounds[constraint.nid]["min"] = constraint.lower
        elif isinstance(constraint, UpperBound):
            bounds[constraint.nid]["max"] = constraint.upper
    return dict(bounds)


def property_to_bounds(prop: BasicProperty, insert_infs=False) -> OptionalNeuronBounds:
    if insert_infs:
        bounds: NeuronBounds = defaultdict(lambda: {"min": -_INF, "max": _INF})
    else:
        bounds: NeuronBounds = defaultdict(lambda: {"min": None, "max": None})
    for constraint in prop.input_constraints:
        if isinstance(constraint, LowerBound):
            bounds[constraint.nid]["min"] = constraint.lower
        elif isinstance(constraint, UpperBound):
            bounds[constraint.nid]["max"] = constraint.upper
    for constraint in prop.output_constraints:
        if isinstance(constraint, LowerBound):
            bounds[constraint.nid]["min"] = constraint.lower
        elif isinstance(constraint, UpperBound):
            bounds[constraint.nid]["max"] = constraint.upper
    return dict(bounds)


def update_property_interval_propagation(
    property: BasicProperty,
    network_before: Network,
    network_after: Network,
) -> BasicProperty:

    input_bounds = property_to_input_bounds(property, insert_infs=True)
    bounds_before = evaluate_bounds_naive(network_before, input_bounds)
    output_bounds_before = [bounds_before[c.nid] for c in property.output_constraints]
    logging.debug(f"bounds_before: {output_bounds_before}")
    bounds_after = evaluate_bounds_naive(network_after, input_bounds)
    output_bounds_after = [bounds_after[c.nid] for c in property.output_constraints]
    logging.debug(f"bounds_after: {output_bounds_after}")

    return update_property(property, bounds_before, bounds_after)


def update_property_with_marabou_preprocess(
    property: BasicProperty,
    network_before: Network,
    network_after: Network,
    query_dir: Optional[Path] = None,
    cache={},
) -> BasicProperty:

    from maraboupy.Marabou import createOptions

    def path_or_none(name: str):
        if query_dir is None:
            return None
        else:
            return query_dir / name

    options = createOptions(tighteningStrategy="sbt", preprocessorBoundTolerance=1e-6)
    neuron_bounds = property_to_bounds(property)

    if id(network_before) in cache:
        bounds_before = cache[id(network_before)]
    else:
        bounds_before = evaluate_bounds_using_marabou_preprocessor(
            network_before,
            neuron_bounds,
            options,
            input_query_save_path=path_or_none("bound_propagate_before.input_query"),
            output_query_save_path=path_or_none("bound_propagate_before.output_query"),
        )
        cache[id(network_before)] = bounds_before
    output_bounds_before = [bounds_before[c.nid] for c in property.output_constraints]
    logging.debug(f"bounds_before: {output_bounds_before}")

    if id(network_after) in cache:
        bounds_after = cache[id(network_after)]
    else:
        bounds_after = evaluate_bounds_using_marabou_preprocessor(
            network_after,
            neuron_bounds,
            options,
            input_query_save_path=path_or_none("bound_propagate_after.input_query"),
            output_query_save_path=path_or_none("bound_propagate_after.output_query"),
        )
        cache[id(network_after)] = bounds_after
    output_bounds_after = [bounds_after[c.nid] for c in property.output_constraints]
    logging.debug(f"bounds_after: {output_bounds_after}")

    return update_property(property, bounds_before, bounds_after)


def update_property(
    property: BasicProperty,
    bounds_before: NeuronBounds,
    bounds_after: NeuronBounds,
) -> BasicProperty:

    output_bounds_before = [bounds_before[c.nid] for c in property.output_constraints]
    output_bounds_after = [bounds_after[c.nid] for c in property.output_constraints]

    bound_diffs = [b_a["min"] - b_b["max"] for b_a, b_b in zip(output_bounds_after, output_bounds_before)]
    new_output_constraints = []
    for output_constraint, bound_diff in zip(property.output_constraints, bound_diffs):
        if isinstance(output_constraint, LowerBound):
            new_bound = max(output_constraint.lower + bound_diff, output_constraint.lower)
            new_constraint = replace(output_constraint, lower=new_bound)
            new_output_constraints.append(new_constraint)
        elif isinstance(output_constraint, UpperBound):
            new_bound = min(output_constraint.upper + bound_diff, output_constraint.upper)
            new_constraint = replace(output_constraint, upper=new_bound)
            new_output_constraints.append(new_constraint)
    return replace(property, output_constraints=new_output_constraints)


class PropertyUpdateMethod(str, Enum):
    IntervalPropagation = "IntervalPropagation"
    Marabou = "Marabou"


class UpdatePropertyHelper:
    def __init__(
        self,
        original_network: Network,
        original_property: BasicProperty,
        update_method: PropertyUpdateMethod = PropertyUpdateMethod.Marabou,
    ):
        self.original_network = original_network
        self.original_property = original_property
        self.update_method = update_method

    def update_property(self, current_network: Network, output_folder: Optional[Path] = None) -> BasicProperty:
        if self.update_method == PropertyUpdateMethod.IntervalPropagation:
            return update_property_interval_propagation(self.original_property, self.original_network, current_network)
        elif self.update_method == PropertyUpdateMethod.Marabou:
            return update_property_with_marabou_preprocess(
                self.original_property, self.original_network, current_network, query_dir=output_folder
            )
        else:
            raise ValueError(f"Unknown update method: {self.update_method}")
