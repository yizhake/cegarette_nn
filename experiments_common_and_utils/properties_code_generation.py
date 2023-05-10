from redesign.datastructures import NeuronId
from redesign.marabou_properties.basic_property import (
    BasicProperty,
    LowerBound,
    UpperBound,
    BasicConstraint,
)
from redesign.marabou_properties.adverserial_property import AdversarialProperty


def nid_name(nid: NeuronId) -> str:
    return nid.name


def lowerbound_code(bound: LowerBound) -> str:
    return f"LowerBound({nid_name(bound.nid)}, {bound.lower})"


def upperbound_code(bound: UpperBound) -> str:
    return f"UpperBound({nid_name(bound.nid)}, {bound.upper})"


def bound_code(bound: BasicConstraint):
    if isinstance(bound, UpperBound):
        return upperbound_code(bound)
    elif isinstance(bound, LowerBound):
        return lowerbound_code(bound)


def indent(s: str, count=4):
    indentation = " " * count
    return indentation + indentation.join(s.splitlines(True))


def basic_property_code(property: BasicProperty):
    input_constraints = ",\n".join(
        bound_code(bound) for bound in property.input_constraints
    )
    output_constraints = ",\n".join(
        bound_code(bound) for bound in property.output_constraints
    )
    return (
        f"BasicProperty(\n"
        f"    input_constraints=[\n"
        f"{indent(input_constraints, count=8)}\n"
        f"    ],\n"
        f"    output_constraints=[\n"
        f"{indent(output_constraints, count=8)}\n"
        f"    ],\n"
        f")"
    )


def adversarial_property_code(property: AdversarialProperty):
    input_constraints = ",\n".join(
        [bound_code(bound) for bound in property.input_constraints]
    )
    output_constraints = ",\n".join(
        [bound_code(bound) for bound in property.output_constraints]
    )
    return (
        f"AdversarialProperty(\n"
        f"    input_constraints=[\n"
        f"{indent(input_constraints, count=8)}\n"
        f"    ],\n"
        f"    output_constraints=[\n"
        f"{indent(output_constraints, count=8)}\n"
        f"    ],\n"
        f"    minimal_is_the_winner={property.minimal_is_the_winner},\n"
        f"    slack={property.slack},\n"
        f")"
    )


def property_code(name, property):
    if isinstance(property, BasicProperty):
        s = ""
        s += "from redesign.marabou_properties.basic_property import LowerBound, UpperBound, BasicProperty\n"
        s += "\n"
        s += f"{name} = {basic_property_code(property)}"
        return s
    elif isinstance(property, AdversarialProperty):
        s = ""
        s += "from redesign.marabou_properties.basic_property import LowerBound, UpperBound\n"
        s += "from redesign.marabou_properties.adverserial_property import AdversarialProperty\n"
        s += "\n"
        s += f"{name} = {adversarial_property_code(property)}"
        return s
    else:
        raise NotImplementedError(f"{type(property)} is not supported")
