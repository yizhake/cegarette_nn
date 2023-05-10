from redesign.marabou_properties.acas_xu_property import Term
from typing import Union
from redesign.datastructures import NeuronId, Scaling, Sign

Pos = Sign.Pos
Neg = Sign.Neg
Inc = Scaling.Inc
Dec = Scaling.Dec


class NeuronId(NeuronId):
    def __neg__(self) -> Term:
        return Term(self, -1.0)

    def __rmul__(self, factor):
        return Term(self, factor)


def _add_operators(nid: NeuronId) -> NeuronId:
    nid = NeuronId(nid.name, nid.sign, nid.scaling)
    return nid


def NeuronLiteral(nid: Union[NeuronId, str]) -> NeuronId:
    if isinstance(nid, str):
        nid = NeuronId(nid)
    nid = _add_operators(nid)
    return nid


def neuron_from_qualified_name(name: str, remove_qualification=False) -> NeuronId:
    sign = None
    scaling = None
    if len(name) >= 2:
        if name[-2] == "+":
            sign = Pos
        elif name[-2] == "-":
            sign = Neg
        if remove_qualification and sign:
            name = name[:-2] + name[-1]
    if len(name) >= 1:
        if name[-1] == "I":
            scaling = Inc
        elif name[-1] == "D":
            scaling = Dec
        if remove_qualification and scaling:
            name = name[:-1]

    return NeuronId(name, sign, scaling)


x0 = NeuronLiteral("x0")
x1 = NeuronLiteral("x1")
x2 = NeuronLiteral("x2")
x3 = NeuronLiteral("x3")
x4 = NeuronLiteral("x4")
x5 = NeuronLiteral("x5")
x6 = NeuronLiteral("x6")
x7 = NeuronLiteral("x7")
x8 = NeuronLiteral("x8")
x9 = NeuronLiteral("x9")
x10 = NeuronLiteral("x10")

y0 = NeuronLiteral("y0")
y1 = NeuronLiteral("y1")
y2 = NeuronLiteral("y2")
y3 = NeuronLiteral("y3")
y4 = NeuronLiteral("y4")
y5 = NeuronLiteral("y5")
y6 = NeuronLiteral("y6")
y7 = NeuronLiteral("y7")
y8 = NeuronLiteral("y8")
y9 = NeuronLiteral("y9")
y10 = NeuronLiteral("y10")
