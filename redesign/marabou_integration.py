from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Type, Union
from collections import ChainMap
from itertools import chain
from maraboupy.Marabou import createOptions, solve_query, load_query
from more_itertools import flatten
from bidict import bidict
import numpy as np
from enum import Enum
import tempfile

from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions
from maraboupy.MarabouNetwork import MarabouNetwork
from maraboupy.MarabouUtils import Equation

from .datastructures import ActivationFunction, Network, NeuronId
from .marabou_properties.basic_property import BasicProperty, LowerBound, UpperBound


@dataclass(frozen=True)
class BNeuron:
    neuron: NeuronId


@dataclass(frozen=True)
class FNeuron:
    neuron: NeuronId


class ARMarabouNetwork(MarabouNetwork):
    def __init__(
        self,
        network: Network,
    ):
        super().__init__()
        self.underline_network = network
        self.prepare_marabou_network()

    def input_neuron_to_marabou_var(self, nid: NeuronId) -> Tuple[FNeuron, int]:
        f_input = self._f_input_neurons.inv[nid]
        var = self.neurons_to_marabou_variables[f_input]
        return f_input, var

    def output_neuron_to_marabou_var(self, nid: NeuronId) -> Tuple[BNeuron, int]:
        b_output = self._b_output_neurons.inv[nid]
        var = self.neurons_to_marabou_variables[b_output]
        return b_output, var

    def hidden_neuron_to_marabou_var(
        self, nid: NeuronId
    ) -> Tuple[Tuple[BNeuron, int], Tuple[FNeuron, int]]:
        b_neuron = self._b_hidden_neurons.inv[nid]
        f_neuron = self._f_hidden_neurons.inv[nid]
        b_pair = b_neuron, self.neurons_to_marabou_variables[b_neuron]
        f_pair = f_neuron, self.neurons_to_marabou_variables[f_neuron]
        return b_pair, f_pair

    def nid_to_marabou_var(self, nid: NeuronId) -> Union[int, Tuple[int, int]]:
        if nid in self._f_input_neurons.inv:
            f_input = self._f_input_neurons.inv[nid]
            var = self.neurons_to_marabou_variables[f_input]
            return var
        elif nid in self._b_output_neurons.inv:
            b_output = self._b_output_neurons.inv[nid]
            var = self.neurons_to_marabou_variables[b_output]
            return var
        else:
            b_neuron = self._b_hidden_neurons.inv[nid]
            f_neuron = self._f_hidden_neurons.inv[nid]
            b_var = self.neurons_to_marabou_variables[b_neuron]
            f_var = self.neurons_to_marabou_variables[f_neuron]
            return b_var, f_var

    def prepare_marabou_network(self):
        network = self.underline_network

        network_neurons = [[nid for nid in layer.nodes] for layer in network.layers]
        network_neurons_activations = ChainMap(
            *(
                {nid: layer.activation for nid in layer.nodes}
                for layer in network.layers
            )
        )
        hidden_neurons = network_neurons[1:-1]

        input_neurons = network_neurons[0]
        output_neurons = network_neurons[-1]
        # input neurons behaves like F-nodes, because they're after (null) activation
        self._f_input_neurons = bidict({FNeuron(nid): nid for nid in input_neurons})
        # output neurons behaves like B-nodes, because they don't have an activation
        self._b_output_neurons = bidict({BNeuron(nid): nid for nid in output_neurons})
        # hidden layers has both B & F nodes
        self._b_hidden_neurons = bidict(
            {BNeuron(nid): nid for layer in hidden_neurons for nid in layer}
        )

        self._f_hidden_neurons = bidict(
            {FNeuron(nid): nid for layer in hidden_neurons for nid in layer}
        )

        all_neurons_for_marabou = chain(
            self._f_input_neurons.keys(),
            self._b_hidden_neurons.keys(),
            self._f_hidden_neurons.keys(),
            self._b_output_neurons.keys(),
        )
        self.neurons_to_marabou_variables = bidict(
            {nid: i for i, nid in enumerate(all_neurons_for_marabou)}
        )
        self.neurons_to_original_neurons = dict(
            ChainMap(
                self._f_input_neurons,
                self._b_hidden_neurons,
                self._f_hidden_neurons,
                self._b_output_neurons,
            )
        )

        self.numVars = len(self.neurons_to_marabou_variables)
        self.inputVars = np.array(
            [[self.neurons_to_marabou_variables[nid] for nid in self._f_input_neurons]]
        )
        self.outputVars = np.array(
            [[self.neurons_to_marabou_variables[nid] for nid in self._b_output_neurons]]
        )

        # add equations
        for ws, bs in zip(network.weights, network.biases[1:]):

            for dst in ws.dests:
                bias = bs[dst]

                # Add marabou equation and add addend for output variable
                e = Equation()
                e.addAddend(-1.0, self.neurons_to_marabou_variables[BNeuron(dst)])

                # Add addends for weighted input variables
                for src in ws.srcs:
                    weight = ws[src, dst]
                    e.addAddend(weight, self.neurons_to_marabou_variables[FNeuron(src)])

                e.setScalar(-bias)
                self.addEquation(e)

        # add relus
        for b, f in zip(self._b_hidden_neurons.keys(), self._f_hidden_neurons.keys()):
            if network_neurons_activations[b.neuron] == ActivationFunction.Relu:
                self.addRelu(
                    self.neurons_to_marabou_variables[b],
                    self.neurons_to_marabou_variables[f],
                )
            elif network_neurons_activations[b.neuron] == ActivationFunction.Id:
                self.addEquality(
                    [
                        self.neurons_to_marabou_variables[b],
                        self.neurons_to_marabou_variables[f],
                    ],
                    [1, -1],
                    0,
                )

    def evaluateWithoutMarabou(self, inputValues):
        if len(inputValues) != 1:
            raise NotImplementedError(
                f"ARMarabouNetwork has support only 1-dim inputs only"
            )
        inputValues = inputValues[0]
        if len(inputValues) != len(self.underline_network.input_ids):
            raise ValueError(
                f"input size does not match. given {len(inputValues)} != {len(self.underline_network.input_ids)} network's"
            )
        input_values = self.underline_network.input_vector(
            {nid: v for nid, v in zip(self.underline_network.input_ids, inputValues)}
        )
        # expected return value to be 2d
        return np.atleast_2d(
            self.underline_network.evaluate(input_values, return_as_numpy=True)  # type: ignore
        )


class MarabouValue(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    QUIT_REQUESTED = "QUIT_REQUESTED"
    NOT_DONE = "NOT_DONE"


@dataclass
class MarabouResults:
    value: MarabouValue
    vals_as_neurons: Union[Dict[Union[BNeuron, FNeuron], float], None]  # None of UNSAT
    inputs_only: Union[Dict[NeuronId, float], None]  # None of UNSAT

    # original marabou output
    vals: Dict[int, float]
    stats: MarabouCore.Statistics


def add_property_bounds_to_ar_marabou_network(
    mnet: ARMarabouNetwork,
    property: BasicProperty,
):
    for constraint in property.input_constraints:
        if isinstance(constraint, LowerBound):
            _, var = mnet.input_neuron_to_marabou_var(constraint.nid)
            mnet.setLowerBound(var, constraint.lower)
        elif isinstance(constraint, UpperBound):
            _, var = mnet.input_neuron_to_marabou_var(constraint.nid)
            mnet.setUpperBound(var, constraint.upper)

    for c in property.output_constraints:
        if isinstance(c, LowerBound):
            _, var = mnet.output_neuron_to_marabou_var(c.nid)
            mnet.setLowerBound(var, c.lower)

        elif isinstance(c, UpperBound):
            _, var = mnet.output_neuron_to_marabou_var(c.nid)
            mnet.setUpperBound(var, c.upper)


def query_marabou(
    mnet: ARMarabouNetwork,
    property: BasicProperty,
    marabou_log="",
    query_save_path="",
    gamma_unsat_input="",
    gamma_unsat_output="",
    tightening_strategy="sbt",
    splitting_strategy="auto",
    **options,
) -> MarabouResults:
    """
    kwargs may include:
    gamma: List[Dict[int:int]], list of unsat activations.
        activation is list of active/inactive nodes, Dict is used in order to use get() method
        ("int", since actually variable indices of nodes are used)
    and
    gamma_abstract: Dict[int, (int, int)],
        mapping from each node to te couple of nodes it was abstracted from
        ("int", since actually variable indices of nodes are used)
    """
    add_property_bounds_to_ar_marabou_network(mnet, property)

    if not query_save_path:
        query_save_path = Path(tempfile.gettempdir()) / next(
            tempfile._get_candidate_names()
        )

    mnet.saveQuery(str(query_save_path))

    options = createOptions(
        preprocessorBoundTolerance=1e-6,
        # gamma_unsat_input=gamma_unsat_input,
        # gamma_unsat_output=gamma_unsat_output,
        tighteningStrategy=tightening_strategy,
        splittingStrategy=splitting_strategy,
        **options,
    )

    # using mnet.solve has issues (same experiment, different result), so we use solve_query instead.
    # res, vals, stats = mnet.solve(verbose=False, filename=marabou_log, options=options)
    query = load_query(str(query_save_path))
    res, vals, stats = solve_query(
        query, verbose=False, filename=str(marabou_log), options=options
    )

    value = {
        "sat": MarabouValue.SAT,
        "unsat": MarabouValue.UNSAT,
        "UNKNOWN": MarabouValue.UNKNOWN,
        "ERROR": MarabouValue.ERROR,
        "TIMEOUT": MarabouValue.TIMEOUT,
        "QUIT_REQUESTED": MarabouValue.QUIT_REQUESTED,
    }[res]

    if value == MarabouValue.SAT:
        vals_as_neurons_mapping = {
            mnet.neurons_to_marabou_variables.inv[i]: v for i, v in vals.items()
        }
        inputs_only = {
            mnet._f_input_neurons[f_input]: vals_as_neurons_mapping[f_input]
            for f_input in mnet._f_input_neurons
        }
    else:
        inputs_only = None
        vals_as_neurons_mapping = None

    results = MarabouResults(
        value=value,
        vals_as_neurons=vals_as_neurons_mapping,
        inputs_only=inputs_only,
        vals=vals,
        stats=stats,
    )
    return results


def call_if_exists(obj, method_name, default, *args, **kwargs):
    if hasattr(obj, method_name):
        return getattr(obj, method_name)(*args, **kwargs)
    else:
        return default


@dataclass
class MarabouStats:
    had_timeout: bool

    @classmethod
    def read_marabou_stats(cls: Type["MarabouStats"], stats_obj: MarabouCore.Statistics) -> "MarabouStats":
        return cls(
            had_timeout=stats_obj.hasTimedOut(),
        )


def query_marabou_use_exe(
    mnet: ARMarabouNetwork,
    property: BasicProperty,
    query_save_path,
    marabou_log="",
    gamma_unsat_input="",
    gamma_unsat_output="",
    **options,
):
    add_property_bounds_to_ar_marabou_network(mnet, property)

    if query_save_path:
        mnet.saveQuery(query_save_path)
    options = createOptions(
        preprocessorBoundTolerance=1e-6,
        gamma_unsat_input=gamma_unsat_input,
        gamma_unsat_output=gamma_unsat_output,
        **options,
    )
    from subprocess import check_output, CalledProcessError
    from pathlib import Path
    import maraboupy
    import re
    from inspect import getfile

    class AllThingsNone:
        def __getattr__(self, _):
            return lambda *args, **kwargs: None

    marabou_exe = Path(getfile(maraboupy)).resolve().parent.parent / "build" / "Marabou"
    try:
        cmd = [str(marabou_exe), "--input-query", str(query_save_path)]
        print(cmd)
        output = check_output(cmd).decode("utf-8")
        with open(marabou_log, "w") as f:
            f.write(output)
        if "Engine::solve: unsat query" in output:
            value = MarabouValue.UNSAT
            inputs_only = None
            vals_as_neurons_mapping = None
            vals = {}
        elif "Input assignment:" in output:
            value = MarabouValue.SAT
            m = next(
                re.finditer(r"Input assignment:((?:\n.*)*)", output, re.MULTILINE)
            ).group(1)
            vals = {}
            for x in range(5):
                # pattern: x0 = 0.0
                x_value = float(
                    next(re.finditer(r"{} = (.*)".format(f"x{x}"), m)).group(1)
                )
                vals[x] = x_value
            y0 = float(next(re.finditer(r"y0 = (.*)", m)).group(1))
            vals_as_neurons_mapping = {
                mnet.neurons_to_marabou_variables.inv[i]: v for i, v in vals.items()
            }
            inputs_only = {
                mnet._f_input_neurons[f_input]: vals_as_neurons_mapping[f_input]
                for f_input in mnet._f_input_neurons
            }
        else:
            value = MarabouValue.UNKNOWN
            inputs_only = None
            vals_as_neurons_mapping = None
            vals = {}

    except CalledProcessError as e:
        with open(marabou_log, "w") as f:
            f.write(e.output.decode("utf-8"))
        value = MarabouValue.UNKNOWN
        inputs_only = None
        vals_as_neurons_mapping = None
        vals = {}

    results = MarabouResults(
        value=value,
        vals_as_neurons=vals_as_neurons_mapping,
        inputs_only=inputs_only,
        vals=vals,
        stats=AllThingsNone(),  # type: ignore
    )
    return results
