from dataclasses import replace

from redesign.datastructures import Scaling
from redesign.marabou_properties.basic_property import BasicProperty


def fix_property_after_preprocess(property: BasicProperty) -> BasicProperty:
    property = replace(
        property,
        output_constraints=[
            replace(c, nid=replace(c.nid, scaling=Scaling.Inc))
            for c in property.output_constraints
        ],
    )
    return property
