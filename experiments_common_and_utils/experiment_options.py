from dataclasses import dataclass

from redesign.marabou_properties.verification_property import SupportedProperties

from .network_code_generation import generate_network_code
from .properties_code_generation import property_code
from .typing_utils import PathLike

from redesign.datastructures import Network


@dataclass
class ExperimentOptions:
    refine_until_not_satisfying: bool = False
    save_all_networks_as_code: bool = False
    save_properties_as_code: bool = False


class ExperimentTracingHelper:
    def __init__(self, options: ExperimentOptions) -> None:
        self.options = options

    def save_network_as_code(self, path: PathLike, network: Network):
        if self.options.save_all_networks_as_code:
            with open(path, "w") as f:
                f.write(generate_network_code(network))

    def save_property_as_code(self, path: PathLike, property: SupportedProperties):
        if self.options.save_properties_as_code:
            with open(path, "w") as f:
                f.write(property_code("test_property", property))

    def save_network_and_property_as_code(
        self, path: PathLike, network: Network, property: SupportedProperties
    ):
        save_any = (
            self.options.save_all_networks_as_code
            or self.options.save_properties_as_code
        )
        if save_any:
            with open(path, "w") as f:
                if self.options.save_all_networks_as_code:
                    f.write(generate_network_code(network))
                if self.options.save_properties_as_code:
                    f.write(property_code("test_property", property))
