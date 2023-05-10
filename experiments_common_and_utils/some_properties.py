from redesign.marabou_properties.basic_property import (
    BasicProperty,
    LowerBound,
    UpperBound,
)
from redesign.marabou_properties.acas_xu_property import (
    ACASXuConjunctionProperty,
    MultiVarUpperBound,
    MultiVarLowerBound,
)
from .neuron_literals import *

#
# ACAs official properties, negated
#

acas_property_1 = BasicProperty(
    input_constraints=[
        LowerBound(x0, 0.6),
        UpperBound(x0, 0.6798577687),
        LowerBound(x1, -0.5),
        UpperBound(x1, 0.5),
        LowerBound(x2, -0.5),
        UpperBound(x2, 0.5),
        LowerBound(x3, 0.45),
        UpperBound(x3, 0.5),
        LowerBound(x4, -0.5),
        UpperBound(x4, -0.45),
    ],
    output_constraints=[LowerBound(y0, 3.9911256459)],
)

acas_property_2 = ACASXuConjunctionProperty(
    input_constraints=[
        LowerBound(x0, 0.6),
        UpperBound(x0, 0.6798577687),
        LowerBound(x1, -0.5),
        UpperBound(x1, 0.5),
        LowerBound(x2, -0.5),
        UpperBound(x2, 0.5),
        LowerBound(x3, 0.45),
        UpperBound(x3, 0.5),
        LowerBound(x4, -0.5),
        UpperBound(x4, -0.45),
    ],
    output_constraints=[
        MultiVarLowerBound([y0, -y1], 0),
        MultiVarLowerBound([y0, -y2], 0),
        MultiVarLowerBound([y0, -y3], 0),
        MultiVarLowerBound([y0, -y4], 0),
    ],
)

acas_property_3 = ACASXuConjunctionProperty(
    input_constraints=[
        LowerBound(x0, -0.3035311561),
        UpperBound(x0, -0.2985528119),
        LowerBound(x1, -0.0095492966),
        UpperBound(x1, 0.0095492966),
        LowerBound(x2, 0.4933803236),
        UpperBound(x2, 0.5),
        LowerBound(x3, 0.3),
        UpperBound(x3, 0.5),
        LowerBound(x4, 0.3),
        UpperBound(x4, 0.5),
    ],
    output_constraints=[
        MultiVarUpperBound([y0, -y1], 0),
        MultiVarUpperBound([y0, -y2], 0),
        MultiVarUpperBound([y0, -y3], 0),
        MultiVarUpperBound([y0, -y4], 0),
    ],
)

acas_property_4 = ACASXuConjunctionProperty(
    input_constraints=[
        LowerBound(x0, -0.3035311561),
        UpperBound(x0, -0.2985528119),
        LowerBound(x1, -0.0095492966),
        UpperBound(x1, 0.0095492966),
        LowerBound(x2, 0),
        UpperBound(x2, 0),
        LowerBound(x3, 0.3181818182),
        UpperBound(x3, 0.5),
        LowerBound(x4, 0.0833333333),
        UpperBound(x4, 0.1666666667),
    ],
    output_constraints=[
        MultiVarUpperBound([y0, -y1], 0),
        MultiVarUpperBound([y0, -y2], 0),
        MultiVarUpperBound([y0, -y3], 0),
        MultiVarUpperBound([y0, -y4], 0),
    ],
)


def acas_official_properties():
    return {
        "property_1": acas_property_1,
        "property_2": acas_property_2,
        "property_3": acas_property_3,
        "property_4": acas_property_4,
    }
