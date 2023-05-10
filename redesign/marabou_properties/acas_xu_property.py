from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import pandas as pd

from .basic_property import BasicProperty, BasicConstraint, LowerBound, UpperBound
from ..datastructures import ActivationFunction, Network, NeuronId, WeightsTable, BiasTable


@dataclass(frozen=True)
class Term:
    nid: NeuronId
    factor: float = 1.0


@dataclass(frozen=True)
class MultiVarConstraint:
    vars: Sequence[Union[Term, NeuronId]]


@dataclass(frozen=True)
class MultiVarLowerBound(MultiVarConstraint):
    lower: float

    @property
    def value(self):
        return self.lower


@dataclass(frozen=True)
class MultiVarUpperBound(MultiVarConstraint):
    upper: float

    @property
    def value(self):
        return self.upper


@dataclass
class ACASXuConjunctionProperty:
    input_constraints: Sequence[BasicConstraint]
    output_constraints: Sequence[Union[MultiVarLowerBound, MultiVarUpperBound]]

def prepare_network_acas_xu_conjunction(
    network: Network, property: ACASXuConjunctionProperty
) -> Tuple[Network, BasicProperty]:
    import sympy as sm
    from bidict import bidict

    if not isinstance(property, ACASXuConjunctionProperty):
        raise TypeError("property should be of ACASXuConjunctionProperty")

    def term_or_neuron_to_sympy(t: Union[Term, NeuronId]) -> sm.Expr:
        if isinstance(t, Term):
            return sm.Symbol(t.nid.name) * t.factor
        elif isinstance(t, NeuronId):
            return sm.Symbol(t.name)

    def convert_neuron_to_symbol(
        neuron: Union[Term, NeuronId]
    ) -> Tuple[float, sm.Symbol]:
        if isinstance(neuron, Term):
            return neuron.factor, convert_neuron_to_symbol(neuron.nid)[1]
        elif isinstance(neuron, NeuronId):
            return 1.0, sm.Symbol(neuron.name)

    def extract_neuron(neuron: Union[Term, NeuronId]) -> NeuronId:
        if isinstance(neuron, Term):
            return extract_neuron(neuron.nid)
        elif isinstance(neuron, NeuronId):
            return neuron

    neuron_to_symbol = bidict(
        {
            extract_neuron(nid): convert_neuron_to_symbol(nid)[1]
            for c in property.output_constraints
            for nid in c.vars
        }
    )

    def constraint_to_inequlity(
        constraint: Union[MultiVarLowerBound, MultiVarUpperBound]
    ):
        lhs = sm.Add(*[term_or_neuron_to_sympy(t) for t in constraint.vars])
        if isinstance(constraint, MultiVarLowerBound):
            return constraint.lower < lhs
        elif isinstance(constraint, MultiVarUpperBound):
            return lhs < constraint.upper

    constraints_as_inequalities = [
        constraint_to_inequlity(constraint) for constraint in property.output_constraints
    ]

    assert (
        len(set(c.rel_op for c in constraints_as_inequalities)) == 1
    ), "All constraints should have the same relational operator"

    def isolate_variables_to_lhs(ineq):
        lhs = sm.S.Zero
        rel_op = ineq.func
        rhs = ineq.rhs - ineq.lhs

        while len(rhs.free_symbols) > 0:
            s = rhs.free_symbols.pop()
            c = rhs.coeff(s)
            rhs -= c * s
            lhs -= c * s
        ineq = rel_op(lhs, rhs)
        return ineq

    isolated = [isolate_variables_to_lhs(ineq) for ineq in constraints_as_inequalities]

    # each inequality will become a new output neuron
    reduced_property_output_constraints = []
    weights = defaultdict(dict)
    output_neurons = []
    for i, ineq in  enumerate(isolated, 1):
        ConstraintType = LowerBound if ineq.rel_op == '>' else UpperBound
        coeff_dict = ineq.lhs.as_coefficients_dict()
        output_neuron = NeuronId(f"c{i}")
        output_neurons.append(output_neuron)
        for s in coeff_dict.keys():
            weights[neuron_to_symbol.inv[s]][output_neuron] = float(coeff_dict[s])
        reduced_property_output_constraints.append(
            ConstraintType(output_neuron, 0.0)
        )
    weights = pd.DataFrame(dict(weights)).fillna(0.0).T

    new_output_layer_weights = WeightsTable(weights)
    new_output_layer_biases = BiasTable({o: 0.0 for o in output_neurons})

    new_network = Network(
        weights=[
            *network.weights,
            new_output_layer_weights,
        ],
        biases=[
            *network.biases,
            new_output_layer_biases,
        ],
        activations=[
            *network.activations,
            ActivationFunction.Id,
        ]
    )
    new_property = BasicProperty(
        property.input_constraints,
        reduced_property_output_constraints,
    )
    return new_network, new_property
