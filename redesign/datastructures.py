from functools import total_ordering
from dataclasses import dataclass, field
from collections import ChainMap
import sys
from typing import (
    Callable,
    Dict,
    ForwardRef,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import TypeAlias
from more_itertools import first, pairwise
import numpy as np
import numpy.typing as npt
import pandas as pd
from enum import Enum
from itertools import chain
from copy import copy

U = TypeVar("U")


class ActivationFunction(str, Enum):
    Relu = "relu"
    Id = "id"

    @property
    def function(self) -> Callable[[U], U]:
        if self == ActivationFunction.Relu:
            return lambda x: np.maximum(x, 0)  # type: ignore
        elif self == ActivationFunction.Id:
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation function: {self}")


class LayerType(str, Enum):
    Input = "input"
    Hidden = "hidden"
    Output = "output"


class Sign(str, Enum):
    Pos = "Pos"
    Neg = "Neg"


class Scaling(str, Enum):
    Inc = "Inc"
    Dec = "Dec"


NeuronType = Tuple[Sign, Scaling]


@total_ordering
@dataclass(frozen=True)
class NeuronId:
    name: str
    sign: Optional[Sign] = field(default=None, compare=False)
    scaling: Optional[Scaling] = field(default=None, compare=False)
    original_neurons: FrozenSet["NeuronId"] = field(
        default_factory=frozenset, compare=False
    )

    def __post_init__(self):
        object.__setattr__(self, "original_neurons", frozenset(self.original_neurons))

    def __repr__(self) -> str:
        if self.original_neurons:
            return f"AbstractedNeuronId({self.name}, {self.sign}, {self.scaling})"
        else:
            return f"NeuronId({self.name}, {self.sign}, {self.scaling})"

    @property
    def type(self):
        return self.sign, self.scaling

    def __eq__(self, other):
        return self.name == other.name and self.type == other.type

    def __lt__(self, other):
        return (self.name, self.type) < (other.name, other.type)


UNIT = NeuronId("_unit")
INPUT = NeuronId("_input")

T = TypeVar("T", bound=npt.NBitBase)
AnyNumber = None  # workaround for typing bug.
if sys.version_info >= (3, 9):
    AnyNumber = Union[np.floating[T], np.integer[T], int, float]
else:
    AnyNumber = Union[int, float]
NeuronToScalar: TypeAlias = Dict[NeuronId, AnyNumber]
NeuronToSequenceOfScalar: TypeAlias = Dict[NeuronId, Sequence[AnyNumber]]
NeuronToNamedScalar: TypeAlias = Dict[NeuronId, Dict[str, AnyNumber]]
NeuronValues: TypeAlias = Union[
    NeuronToScalar, NeuronToSequenceOfScalar, NeuronToNamedScalar
]


class WeightsTable:
    def __init__(
        self, weights: Union[Dict[NeuronId, NeuronToScalar], pd.DataFrame]
    ) -> None:
        # weights should be either a dict or a dataframe.
        # if it's a dataframe, then it should have sources on the 0-axis (index),
        #   and destinations on the 1-axis (columns)
        if isinstance(weights, pd.DataFrame):
            if not all(isinstance(s, NeuronId) for s in weights.index):
                raise TypeError(f"`weights` indices must be `NeuronId`")
            if not all(isinstance(s, NeuronId) for s in weights.columns):
                raise TypeError(f"`weights` columns must be `NeuronId`")
            self.table = weights

        # if it's a dict, then it should be { src: { dest: weight } }
        elif isinstance(weights, dict):
            dtype = type(first(first(weights.values()).values()))
            self.table = (
                pd.DataFrame(weights, dtype=dtype)
                .T.reindex(weights.keys(), axis=0)
                .reindex(first(weights.values()).keys(), axis=1)
            )

        else:
            raise TypeError(f"incorrect type for `weights`: {type(weights)}")

        # This is used for the abstraction/refinment
        self._orig_table: WeightsTable = None
        # Literal[0, 1] represnt the abstraction axis (on rows or cols)
        # 
        self._abstraction_steps: List[
            Tuple[Literal[0, 1], ForwardRef("AbstractionStep")]
        ] = []

    def __getitem__(self, index):
        assert isinstance(index, tuple)
        src, dest = index
        return self.table.loc[src][dest]

    @property
    def srcs(self) -> Sequence[NeuronId]:
        return self.table.index

    @property
    def dests(self) -> Sequence[NeuronId]:
        return self.table.columns

    # for IPython
    def _repr_html_(self):
        return self.table._repr_html_()

    def copy(self):
        obj = type(self)(self.table.copy(deep=True))
        obj._orig_table = copy(self._orig_table)
        obj._abstraction_steps = copy(self._abstraction_steps)
        return obj

    def __copy__(self):
        return self.copy()

    @property
    def matrix(self) -> np.ndarray:
        return self.table.values


class BiasTable(WeightsTable):
    def __init__(self, biases: Union[NeuronToScalar, pd.DataFrame]) -> None:
        # biases should be either a dict or a dataframe.
        # if it's a dataframe, then it should have only UNIT on the 0-axis (index),
        #   and NeuronId's on the 1-axis (columns)
        if isinstance(biases, pd.DataFrame):
            if not (len(biases.index) == 1 and biases.index[0] == UNIT):
                raise TypeError(f"`biases` index must be [UNIT]")
            if not all(isinstance(s, NeuronId) for s in biases.columns):
                raise TypeError(f"`biases` columns must be `NeuronId`")
            super().__init__(biases)

        # if it's a dict, then it should be { neuron_id: bias }
        elif isinstance(biases, dict):
            super().__init__({UNIT: biases})

        else:
            raise TypeError(f"incorrect type for `biases`: {type(biases)}")

    def __getitem__(self, node_id):
        assert isinstance(node_id, NeuronId)
        return super().__getitem__((UNIT, node_id))

    @property
    def ids(self) -> Sequence[NeuronId]:
        return self.dests

    @property
    def matrix(self) -> np.ndarray:
        return super().matrix.ravel()


class Layer:
    def __init__(
        self, type: LayerType, nodes: List[NeuronId], activation: ActivationFunction
    ) -> None:
        self.nodes = nodes
        self.type = type
        self.activation = activation

    @property
    def type_name(self):
        return self.type


class Network:
    def __init__(
        self,
        weights: List[WeightsTable],
        biases: List[BiasTable],
        activations: List[ActivationFunction],
    ):
        # for a network with N layers, there are `len(weights) - 1` weights tables,
        # one between two layers, and there are `len(biases)` biases tables, one
        # for each layer
        assert len(weights) + 1 == len(biases)
        assert len(activations) == len(biases)
        self.weights = weights
        self.biases = biases
        self.activations = activations
        self.activation_functions = [a.function for a in activations]

    @overload
    def evaluate(
        self,
        inputs: Union[NeuronValues, pd.DataFrame],
        verify: Optional[bool] = False,
        return_intermediate: Optional[bool] = False,
        return_as_numpy: Optional[Literal[False]] = False,
        return_as_dict: Optional[Literal[True]] = True,
    ) -> Dict[NeuronId, AnyNumber]:
        ...

    @overload
    def evaluate(
        self,
        inputs: Union[NeuronValues, pd.DataFrame],
        verify: Optional[bool] = False,
        return_intermediate: Optional[bool] = False,
        return_as_numpy: Optional[Literal[False]] = False,
        return_as_dict: Optional[Literal[False]] = False,
    ) -> pd.DataFrame:
        ...

    @overload
    def evaluate(
        self,
        inputs: Union[NeuronValues, pd.DataFrame],
        verify: Optional[bool] = False,
        return_intermediate: Optional[bool] = False,
        return_as_numpy: Optional[Literal[True]] = True,
        return_as_dict: Optional[Literal[False]] = False,
    ) -> np.ndarray:
        ...

    def evaluate(
        self,
        inputs: Union[NeuronValues, pd.DataFrame],
        verify=False,
        return_intermediate=False,
        return_as_numpy=False,
        return_as_dict=False,
    ):
        if sum((return_as_numpy, return_as_dict)) > 1:
            raise ValueError(
                f"parameters `return_as_numpy`, `return_as_dict` are mutually-exclusive"
            )
        if not isinstance(inputs, pd.DataFrame):
            inputs = self.input_vector(inputs, verify=verify)

        all_values = [inputs]
        current_values = self.activation_functions[0](
            inputs.values + self.biases[0].table.values
        )
        for w, b, act in zip(
            self.weights[:-1], self.biases[1:-1], self.activation_functions[1:-1]
        ):
            current_values = act(current_values @ w.table.values + b.table.values)
            if return_intermediate:
                all_values.append(
                    pd.DataFrame(current_values, index=inputs.index, columns=w.dests)
                )
        current_values = self.activation_functions[-1](
            current_values @ self.weights[-1].table.values
            + self.biases[-1].table.values
        )
        if return_intermediate:
            all_values.append(
                pd.DataFrame(
                    current_values, index=inputs.index, columns=self.weights[-1].dests
                )
            )

        if return_as_numpy:
            return current_values
        elif return_as_dict:
            if return_intermediate:
                if len(inputs.index) == 1 and inputs.index == [INPUT]:
                    # since the input uses INPUT as the neuron id, we want to extract it into a float
                    return dict(
                        ChainMap(*[res.loc[INPUT].to_dict() for res in all_values])
                    )
                else:
                    merge_rows_to_dict = (
                        lambda df: df.groupby(lambda _: True)
                        .agg(list)
                        .loc[True]
                        .to_dict()
                    )
                    return dict(
                        ChainMap(*[merge_rows_to_dict(res) for res in all_values])
                    )
            else:
                return dict(zip(self.output_ids, current_values.T))
        else:
            if return_intermediate:
                return all_values
            else:
                return pd.DataFrame(
                    current_values, index=inputs.index, columns=self.output_ids
                )

    def input_vector(self, data: NeuronValues, verify=False) -> pd.DataFrame:
        if verify:
            if s := data.keys() ^ self.input_ids:
                raise ValueError(
                    f"`data.keys()` does not match network inputs. differ at {s}"
                )
        if np.isscalar(next(iter(data.values()))):
            # single input
            return pd.DataFrame({INPUT: {**data}}).reindex(self.input_ids).T
        else:
            return pd.DataFrame(data)

    def layers_count(self):
        return len(self.weights) + 1

    @property
    def layers(self) -> Sequence[Layer]:
        return tuple(
            chain(
                (Layer(LayerType.Input, self.weights[0].srcs, self.activations[0]),),
                (
                    Layer(LayerType.Hidden, wt.srcs, a)
                    for wt, a in zip(self.weights[1:], self.activations[1:-1])
                ),
                (
                    Layer(
                        LayerType.Output, self.weights[-1].dests, self.activations[-1]
                    ),
                ),
            )
        )

    @property
    def input_ids(self) -> Sequence[NeuronId]:
        return self.weights[0].srcs

    @property
    def output_ids(self) -> Sequence[NeuronId]:
        return self.weights[-1].dests

    @property
    def neuron_ids(self) -> Sequence[NeuronId]:
        from more_itertools import flatten

        return tuple(
            chain(
                tuple(self.weights[0].srcs),
                flatten(tuple(wt.srcs) for wt in self.weights[1:]),
                tuple(self.weights[-1].dests),
            )
        )

    def verify(self):
        for l1, l2 in pairwise(self.weights):
            assert (l1.dests == l2.srcs).all()
        for w, b in zip(self.weights, self.biases):
            assert (w.srcs == b.ids).all()
        assert (self.weights[-1].dests == self.biases[-1].ids).all()

    def __getitem__(self, slc):
        if isinstance(slc, slice):
            if slc.step is not None and slc.step != 1:
                raise ValueError("step must be either `None` or `1`")

            start = slc.start % self.layers_count()
            stop = slc.stop % self.layers_count() if slc.stop is not None else None
            slc = slice(start, stop)
            w_stop = self.layers_count() - 1 if slc.stop is None else slc.stop - 1
            print(f"self.weights[{start} : {w_stop}]")
            print(f"self.biases[{slc}]")
            return Network(
                weights=self.weights[slc.start : w_stop],
                biases=self.biases[slc],
                activations=self.activations[slc],
            )
        else:
            raise ValueError(
                "only slices are legal when using [...] operator on a Network"
            )


class Activation(str, Enum):
    Active = "Active"
    Inactive = "Inactive"
