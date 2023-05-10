from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union, List
import numpy as np

from ..datastructures import AnyNumber, Network, NeuronId, NeuronToScalar

FLOAT_T = np.float64
MIN: FLOAT_T = np.finfo(FLOAT_T).min
MAX: FLOAT_T = np.finfo(FLOAT_T).max


@dataclass(frozen=True)
class LowerBound:
    nid: NeuronId
    lower: AnyNumber = MIN


@dataclass(frozen=True)
class UpperBound:
    nid: NeuronId
    upper: AnyNumber = MAX


BasicConstraint = Union[LowerBound, UpperBound]


@dataclass
class BasicProperty:
    input_constraints: Sequence[BasicConstraint]
    output_constraints: Sequence[BasicConstraint]


def prepare_network_basic_property(
    network: Network, property: BasicProperty
) -> Tuple[Network, BasicProperty]:
    if not isinstance(property, BasicProperty):
        raise TypeError("property should be of BasicProperty")
    return network, property


def compare_with_precision(cmp_func, eps=1e-4, *, less_eps_is_eq):
    def func(a, b, eps=eps):
        diff = a - b
        if abs(diff) < eps:
            return less_eps_is_eq
        else:
            return cmp_func(diff, 0)

    return func


import operator

gt_with_precision = compare_with_precision(operator.gt, less_eps_is_eq=False)
lt_with_precision = compare_with_precision(operator.lt, less_eps_is_eq=False)


def is_constraint_satisfied(
    values: NeuronToScalar, constraint: BasicConstraint
) -> Tuple[bool, str]:
    value = values[constraint.nid]
    if isinstance(constraint, LowerBound):
        if lt_with_precision(value, constraint.lower):
            return (
                False,
                f"({value} = {constraint.nid.name}) < (lowerbound = {constraint.lower})",
            )
        else:
            return (
                True,
                f"({value} = {constraint.nid.name}) >= (lowerbound = {constraint.lower})",
            )
    elif isinstance(constraint, UpperBound):
        if gt_with_precision(value, constraint.upper):
            return (
                False,
                f"({value} = {constraint.nid.name}) > (upperbound = {constraint.upper})",
            )
        else:
            return (
                True,
                f"({value} = {constraint.nid.name}) <= (upperbound = {constraint.upper})",
            )


def are_constraints_satisfied(
    values: NeuronToScalar, constraints: Sequence[BasicConstraint]
) -> bool:
    return all(is_constraint_satisfied(values, c)[0] for c in constraints)


def is_satisfying_assignment(
    net: Network,
    values,
    test_property: BasicProperty,
    verify_input: bool = True,
) -> Tuple[bool, List[str]]:
    if not isinstance(test_property, BasicProperty):
        raise TypeError("test_property should be of BasicProperty")
    if verify_input:
        if not are_constraints_satisfied(values, test_property.input_constraints):
            return False, ["inputs"]

    if isinstance(net, Network):
        net_output = net.evaluate(values, return_as_dict=True)
        is_satisfied = True
        why_not = []
        for output_constraint in test_property.output_constraints:
            satisfied, why = is_constraint_satisfied(net_output, output_constraint)
            is_satisfied = is_satisfied and satisfied
            why_not.append(why)
        return is_satisfied, why_not
    else:
        raise Exception(f"net type {type(net)} is not supported")
