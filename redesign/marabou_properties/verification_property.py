from typing import Union

from .basic_property import BasicProperty
from .acas_xu_property import ACASXuConjunctionProperty
from .adverserial_property import AdversarialProperty
from ..datastructures import Network
from redesign.marabou_properties.basic_property import (
    BasicProperty,
)
from typing import Union

SupportedProperties = Union[
    BasicProperty,
    ACASXuConjunctionProperty,
    AdversarialProperty,
]

def prepare_network_for_property(network: Network, property: SupportedProperties):
    from .basic_property import BasicProperty, prepare_network_basic_property
    from .acas_xu_property import (
        ACASXuConjunctionProperty,
        prepare_network_acas_xu_conjunction,
    )
    from .adverserial_property import (
        AdversarialProperty,
        prepare_network_adversarial,
    )

    if isinstance(property, BasicProperty):
        return prepare_network_basic_property(network, property)
    elif isinstance(property, ACASXuConjunctionProperty):
        return prepare_network_acas_xu_conjunction(network, property)
    elif isinstance(property, AdversarialProperty):
        return prepare_network_adversarial(network, property)
    else:
        raise NotImplementedError(
            f"not implemtented for property of type `{type(property)}`"
        )
