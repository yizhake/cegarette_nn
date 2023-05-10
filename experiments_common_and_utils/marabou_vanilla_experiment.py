from dataclasses import asdict
from pathlib import Path
import structlog

logger = structlog.get_logger()

from redesign.datastructures import Network
from redesign.marabou_properties.verification_property import (
    prepare_network_for_property,
)
from redesign.marabou_integration import ARMarabouNetwork, MarabouStats, MarabouValue, query_marabou
from .experiment_options import ExperimentOptions
from .single_experiment import SupportedProperties
from .time_utils import Timer
from .report_utils import network_info_as_dict
from .typing_utils import PathLike


def run_marabou_vanilla_experiment(
    network: Network,
    test_property: SupportedProperties,
    output_folder: PathLike,
    experiment_options: ExperimentOptions,
):
    output_folder = Path(output_folder)

    experiment_time = Timer().start()

    logger.info("original network info", network_info=network_info_as_dict(network))
    logger.msg(
        "preparing network with property",
        property_type=str(type(test_property).__name__),
    )
    network, test_property = prepare_network_for_property(network, test_property)
    logger.info(
        "property-reduced network info", network_info=network_info_as_dict(network)
    )
    experiment_final_marabou_status: MarabouValue = MarabouValue.UNKNOWN

    mnet = ARMarabouNetwork(network)
    with Timer() as t:
        marabou_log = str((output_folder / f"marabou-log").absolute())
        query_save_path = str((output_folder / f"query").absolute())
        logger.msg(
            "querying marabou",
            network_info=network_info_as_dict(network),
            marabou_log=marabou_log,
            query_path=query_save_path,
        )
        query_result = query_marabou(
            mnet=mnet,
            property=test_property,
            marabou_log=marabou_log,
            query_save_path=query_save_path,
        )
    logger.msg("marabou query finished", time=t())
    logger.msg(
            "marabou stats", **asdict(MarabouStats.read_marabou_stats(query_result.stats))
        )

    experiment_final_marabou_status = query_result.value
    experiment_time = experiment_time.stop()

    logger.info(
        "summary",
        experiment_final_marabou_status=experiment_final_marabou_status,
        experiment_time=experiment_time,
    )
