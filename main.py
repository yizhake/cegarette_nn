from dataclasses import asdict, dataclass, make_dataclass
from more_itertools import first
import socket
from ast import literal_eval
from typing import Optional, Type, Union, ClassVar
from simple_parsing import ArgumentParser, subgroups, field
from simple_parsing.wrappers.field_wrapper import DashVariant
from pathlib import Path
from enum import Enum
import structlog
from redesign.marabou_properties.update_property import PropertyUpdateMethod
from redesign.marabou_properties.verification_property import SupportedProperties

logger = structlog.get_logger()

from redesign.refinement.refinement_strategies import (
    RandomRefine,
    RefineByMaxActivations,
    RefineByMaxLoss,
    RefineByMaxLossClustered,
)
from redesign.abstraction.abstraction_strategies import (
    CompleteAbstractionLeftToRight,
    CompleteAbstractionRightToLeft,
)

from experiments_common_and_utils.nnet_paths import NNET_ROOT_PATH, all_nnet_paths
from experiments_common_and_utils.read_network import read_network, SupportedNetworks
from experiments_common_and_utils.tee import Tee
from experiments_common_and_utils.some_properties import acas_official_properties
from experiments_common_and_utils.more_properties import acas_adversarial_properties
from experiments_common_and_utils.experiment_options import ExperimentOptions

properties = {
    **acas_official_properties(),
    **acas_adversarial_properties(),
}


class Method(Enum):
    marabou_vanilla = m = "marabou_vanilla"
    abstraction_refinement = ar = "abstraction_refinement"


class AbstractionStrategy(Enum):
    complete_left_to_right = l2r = "complete_left_to_right"
    complete_right_to_left = r2l = "complete_right_to_left"

    def get_class(self):
        return {
            AbstractionStrategy.complete_left_to_right: CompleteAbstractionLeftToRight,
            AbstractionStrategy.complete_right_to_left: CompleteAbstractionRightToLeft,
        }[self]


class RefinementStrategy(Enum):
    random = "random"
    by_max_loss = "by_max_loss"
    by_max_loss_clustered = "by_max_loss_clustered"
    by_max_activations = "by_max_activations"

    def get_class(self):
        return {
            RefinementStrategy.random: RandomRefine,
            RefinementStrategy.by_max_loss: RefineByMaxLoss,
            RefinementStrategy.by_max_loss_clustered: RefineByMaxLossClustered,
            RefinementStrategy.by_max_activations: RefineByMaxActivations,
        }[self]


@dataclass
class TypedPathArg:
    type: ClassVar[str]
    path: Path = field(positional=True)


def network_type(name: str) -> Type:
    return make_dataclass(
        "Network",
        [
            (
                "type",
                ClassVar[str],
                field(default=name),
            )
        ],
        bases=(TypedPathArg,),
    )


def network_subgroups():
    group = {type.name: network_type(type.name) for type in SupportedNetworks}
    return subgroups(group, default=group[first(group)])


def flag():
    return field(action="store_true")


@dataclass
class Args:
    output_path: Path = Path("/tmp/output_path")

    method: Method = Method.ar

    property: Union[Path, str] = "adversarial_1"

    network_path: TypedPathArg = network_subgroups()

    update_property: Optional[PropertyUpdateMethod] = None
    # If set, the property output will be updated according to this method

    abstraction_strategy: AbstractionStrategy = AbstractionStrategy.r2l
    abstraction_args: Optional[str] = ""  # comma-seperated arguments for the abstraction strategy constructor

    refinement_strategy: RefinementStrategy = RefinementStrategy.by_max_loss
    refinement_args: Optional[str] = "1"  # comma-seperated arguments for the refinement strategy constructor

    # options
    save_networks_as_code: bool = flag()
    # If set, all networks will be saved as python code. useful for debugging",
    save_properties_as_code: bool = flag()
    # If set, all properties will be saved as python code. useful for debugging,
    refine_until_not_satisfying: bool = flag()
    # If set, the refinement will stop when the property is not satisfied,


def load_property(property: Union[str, Path]) -> SupportedProperties:
    if Path(property).is_file() and Path(property).suffix == ".py":
        import importlib.util

        spec = importlib.util.spec_from_file_location("property", property)
        property_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(property_module)
        if not hasattr(property_module, "property"):
            raise ValueError(f"The property file {property} does not contain a property function")
        return property_module.property
    elif (property := str(property)) in properties:
        return properties[property]
    else:
        raise ValueError(f"Unknown property {property}")


def main(args: Args):
    experiment_options = ExperimentOptions()

    # >>> user-inputs
    property_to_do = args.property
    network_type = args.network_path.type
    network_path = args.network_path.path
    output_path = args.output_path
    # options
    experiment_options.refine_until_not_satisfying = args.refine_until_not_satisfying
    experiment_options.save_all_networks_as_code = args.save_networks_as_code
    experiment_options.save_properties_as_code = args.save_properties_as_code
    # <<< user-inputs

    # setup log
    output_folder = output_path
    output_folder.mkdir(exist_ok=True, parents=True)

    log_file = output_folder / f"log.jsonl"
    with Tee(log_file, "w") as log_fp:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper("iso", utc=False),
                structlog.stdlib.add_log_level,
                structlog.processors.JSONRenderer(sort_keys=True),
            ],
            logger_factory=structlog.PrintLoggerFactory(log_fp),
        )

        logger.info("method", name=str(args.method.value))
        logger.info("options", **asdict(experiment_options))

        if args.method == Method.marabou_vanilla:
            logger.info("user input", network_path=str(network_path))
            logger.info("user input", property=property_to_do)

            # acquire data
            network = read_network(network_type, network_path)
            test_property = load_property(property_to_do)
            logger.info("testing property", property=test_property)

            with open(output_folder / "hostname", "w") as fw:
                fw.write(socket.gethostname() + "\n")

            from experiments_common_and_utils.marabou_vanilla_experiment import (
                run_marabou_vanilla_experiment,
            )

            run_marabou_vanilla_experiment(network, test_property, output_folder, experiment_options)

        elif args.method == Method.abstraction_refinement:
            logger.info("user input", network_path=str(network_path))
            logger.info("user input", property=property_to_do)
            logger.info(
                "user input",
                abstraction=str(args.abstraction_strategy),
                arguments=args.abstraction_args,
            )
            logger.info(
                "user input",
                refinement=str(args.refinement_strategy),
                arguments=args.refinement_args,
            )
            logger.info("user input", update_property=str(args.update_property))

            # acquire data
            network = read_network(network_type, network_path)
            test_property = load_property(property_to_do)
            logger.info("testing property", property=test_property)

            abstraction_args = literal_eval(f"[{args.abstraction_args}]")
            abstraction_strategy = args.abstraction_strategy.get_class()(*abstraction_args)
            logger.info(
                "abstraction strategy",
                class_name=abstraction_strategy.__class__.__name__,
                arguments=abstraction_args,
            )

            refinement_args = literal_eval(f"[{args.refinement_args}]")
            refinement_strategy = args.refinement_strategy.get_class()(*refinement_args)
            logger.info(
                "refinement strategy",
                class_name=refinement_strategy.__class__.__name__,
                arguments=refinement_args,
            )

            from experiments_common_and_utils.single_experiment import run_experiment

            run_experiment(
                network,
                test_property,
                abstraction_strategy,
                refinement_strategy,
                output_folder,
                experiment_options,
                update_property=args.update_property,
            )


def example() -> Args:
    args = Args()

    args.output_path = Path("/tmp/afzal_net-1-6_prop_conj3")

    args.method = Method.ar

    # args.network_path = network_type(SupportedNetworks.nnet.value)(all_nnet_paths[0])
    args.network_path = network_type(SupportedNetworks.nnet.value)(
        "/home/yizhak/Research/Code/MarabouApplications/acas/nnet/ACASXU_run2a_1_2_batch_2000.nnet"
    )

    args.property = "adversarial_1"

    args.abstraction_strategy = AbstractionStrategy.complete_right_to_left
    args.abstraction_args = ""

    args.refinement_strategy = RefinementStrategy.by_max_loss
    args.refinement_args = "1"

    args.refine_until_not_satisfying = True

    args.update_property = PropertyUpdateMethod.IntervalPropagation  # or: PropertyUpdateMethod.Marabou

    return args


if __name__ == "__main__":

    parser = ArgumentParser(add_option_string_dash_variants=DashVariant.DASH)
    parser.add_arguments(Args, dest="args")

    use_example = False
    if use_example:
        args = example()
    else:
        args: Args = parser.parse_args().args

    main(args)
