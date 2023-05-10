import os
from pathlib import Path
from typing import Dict, Optional, TypedDict
from typing_extensions import TypeAlias
from maraboupy import MarabouCore

from ..datastructures import AnyNumber, Network, NeuronId


class Bounds(TypedDict):
    min: AnyNumber
    max: AnyNumber


class OptionalBounds(TypedDict):
    min: Optional[AnyNumber]
    max: Optional[AnyNumber]


NeuronBounds: TypeAlias = Dict[NeuronId, Bounds]
OptionalNeuronBounds: TypeAlias = Dict[NeuronId, OptionalBounds]


def evaluate_bounds_naive(network: Network, input_bounds: NeuronBounds) -> NeuronBounds:

    inputs = network.input_vector(input_bounds)
    current_bounds = inputs
    all_bounds = {}
    all_bounds.update(current_bounds.to_dict())
    for w, b in zip(network.weights[:], network.biases[1:]):

        pos = w.table.where(w.table.values >= 0, 0)
        neg = w.table.where(w.table.values <= 0, 0)

        by_pos = current_bounds @ pos

        by_neg = current_bounds @ neg
        by_neg = by_neg.rename(index={"min": "max", "max": "min"})

        bounds_before_relu = by_pos + by_neg + b.table.values
        current_bounds = bounds_before_relu.where(bounds_before_relu >= 0, 0)
        all_bounds.update(current_bounds.to_dict())

    return all_bounds


def evaluate_bounds_using_marabou_preprocessor(
    network: Network,
    neuron_bounds: OptionalNeuronBounds,
    options: MarabouCore.Options,
    input_query_save_path: Optional[Path] = None,
    output_query_save_path: Optional[Path] = None,
) -> NeuronBounds:
    from ..marabou_integration import ARMarabouNetwork

    mnet = ARMarabouNetwork(network)
    for nid, bounds in neuron_bounds.items():
        var = mnet.nid_to_marabou_var(nid)
        # print(nid, var)
        assert isinstance(var, int), "expected to be only input/output bounds"
        if (lower := bounds["min"]) is not None:
            mnet.setLowerBound(var, lower)
        if (upper := bounds["max"]) is not None:
            mnet.setUpperBound(var, upper)

    query = mnet.getMarabouQuery()
    
    if input_query_save_path is not None:
        MarabouCore.saveQuery(query, str(input_query_save_path))

    preprocessed_query = MarabouCore.preprocess(query, options, os.devnull, returnFullyProcessedQuery=True)

    if output_query_save_path is not None:
        MarabouCore.saveQuery(preprocessed_query, str(output_query_save_path))

    res_bounds: NeuronBounds = {}
    for nid in network.input_ids:
        marabou_var = mnet.nid_to_marabou_var(nid)
        lower = preprocessed_query.getLowerBound(marabou_var)
        upper = preprocessed_query.getUpperBound(marabou_var)
        res_bounds[nid] = {"min": lower, "max": upper}

    for nid in network.output_ids:
        marabou_var = mnet.nid_to_marabou_var(nid)
        lower = preprocessed_query.getLowerBound(marabou_var)
        upper = preprocessed_query.getUpperBound(marabou_var)
        res_bounds[nid] = {"min": lower, "max": upper}

    return res_bounds
