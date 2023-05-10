from typing import Iterable, List, Tuple
from more_itertools import unique_everseen
from more_itertools.recipes import flatten, pairwise
import json
from numbers import Number

from jsonconversion.encoder import JSONObjectEncoder
from redesign.datastructures import ActivationFunction, NeuronId, UNIT, WeightsTable, BiasTable, Network


class NumpyDictWriter(JSONObjectEncoder):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.values_ctor = lambda v: f"numpy.{dtype.name}({v})"

    def isinstance(self, obj, cls):
        if isinstance(obj, Number):
            return False
        return super(JSONObjectEncoder, self).isinstance(obj, cls)

    def default(self, obj):
        if isinstance(obj, Number):
            return self.values_ctor(obj)

        return super().default(obj)


def make_identifier(name: str) -> str:
    return name.replace(":", "_").replace('-', '_N_').replace('+', '_P_')


def neuron_id_name(neuron_id: "NeuronId") -> str:
    return make_identifier(neuron_id.name)


def network_name(network: Network) -> str:
    return make_identifier("network")


def weights_name(weights: "WeightsTable", indices: Tuple[int, int]) -> str:
    s, d = indices
    return make_identifier(f"w_{s}_{d}")


def biases_name(biases: "BiasTable", index: int) -> str:
    return make_identifier(f"b_{index}")


def neuron_id_code(neuron_id: "NeuronId", assignment=True) -> str:
    name = neuron_id_name(neuron_id)
    sign = f"Sign.{neuron_id.sign}" if neuron_id.sign is not None else None
    scaling = f"Scaling.{neuron_id.scaling}" if neuron_id.scaling is not None else None
    assignment = f"{name} = " if assignment else ""
    return (
        f"{assignment}"
        f'NeuronId("{neuron_id.name}", '
        f"sign={sign}, "
        f"scaling={scaling})"
    )


def weights_code(
    weights: "WeightsTable", indices: Tuple[int, int], assignment=True
) -> str:
    name = weights_name(weights, indices)
    assignment = f"{name} = " if assignment else ""

    d = (
        weights.table.T.rename(neuron_id_name, axis=0)
        .rename(neuron_id_name, axis=1)
        .to_dict()
    )

    underline_type = weights.table.values.dtype
    s = json.dumps(d, cls=NumpyDictWriter, indent=2, dtype=underline_type)
    s = "\n  ".join(s.replace('"', "").splitlines())
    return f"{assignment}" f"WeightsTable(\n" f"  {s}\n" f")"


def biases_code(biases: "BiasTable", index: int, assignment=True) -> str:
    name = biases_name(biases, index)
    assignment = f"{name} = " if assignment else ""

    d = biases.table.loc[UNIT].rename(neuron_id_name).to_dict()

    underline_type = biases.table.values.dtype
    s = json.dumps(d, cls=NumpyDictWriter, indent=2, dtype=underline_type)
    s = "\n  ".join(s.replace('"', "").splitlines())
    return f"{assignment}" f"BiasTable(\n" f"  {s}\n" f")"

def activations_code(activations: List[ActivationFunction]) -> str:
    s = "["
    s += ", ".join(f"ActivationFunction.{a.name}" for a in activations)
    s += "]"
    return s

def weights_enumerate_as_pairwise(
    weights: List[WeightsTable],
) -> Iterable[Tuple[Tuple[int, int], WeightsTable]]:
    return zip(pairwise(range(len(weights) + 1)), weights)


def network_code(network: "Network", assignment=True) -> str:
    name = network_name(network)
    assignment = f"{name} = " if assignment else ""

    weights_names = [
        weights_name(w, ij) for ij, w in weights_enumerate_as_pairwise(network.weights)
    ]
    biases_names = [biases_name(b, i) for i, b in enumerate(network.biases)]

    return (
        f"{assignment}"
        f"Network(\n"
        f'  weights=[{", ".join(weights_names)}],\n'
        f'  biases=[{", ".join(biases_names)}],\n'
        f'  activations={activations_code(network.activations)},\n'
        f")"
    )


def imports() -> str:
    return (
        f"import numpy\n"
        f"from redesign.datastructures import NeuronId, WeightsTable, BiasTable, Network, Sign, Scaling, ActivationFunction\n"
    )


def generate_network_code(network: "Network", *, to_clipboard: bool = False, add_main: bool = False) -> str:
    neuron_ids = flatten([l.nodes for l in network.layers])
    neuron_ids = list(unique_everseen(neuron_ids))

    s = ""
    s += imports()
    s += "\n\n"
    for n in neuron_ids:
        s += neuron_id_code(n)
        s += "\n"
    s += "\n"
    for ij, w in weights_enumerate_as_pairwise(network.weights):
        s += weights_code(w, ij)
        s += "\n"
    for i, b in enumerate(network.biases):
        s += biases_code(b, i)
        s += "\n"
    s += "\n"
    s += network_code(network)
    s += "\n"

    if to_clipboard:
        from pandas.io import clipboard
        clipboard.copy(s)

    return s
