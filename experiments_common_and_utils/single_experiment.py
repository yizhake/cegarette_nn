from copy import deepcopy
from dataclasses import asdict
from itertools import count
from pathlib import Path
from typing import List, Optional
import structlog
import pickle

logger = structlog.get_logger()

from redesign.datastructures import Network
from redesign.refinement.refine import (
    RefinementStatitistics,
    RefinementStrategy,
    refine_until_not_satisfying,
    refine_network,
)
from redesign.abstraction.abstraction import AbstractionStrategy, abstract_network
from redesign.marabou_properties.basic_property import (
    is_satisfying_assignment,
)
from redesign.preprocess.preprocess import preprocess
from redesign.marabou_properties.verification_property import (
    SupportedProperties,
    prepare_network_for_property,
)
from redesign.marabou_integration import (
    ARMarabouNetwork,
    MarabouStats,
    MarabouValue,
    query_marabou,
)
from redesign.marabou_properties.update_property import (
    PropertyUpdateMethod,
    UpdatePropertyHelper,
)

from .experiment_options import ExperimentOptions, ExperimentTracingHelper
from .fix_property_after_preprocess import fix_property_after_preprocess
from .time_utils import Timer
from .report_utils import network_info_as_dict
from .typing_utils import PathLike, cast_none


SAVE_NET = False


def run_experiment(
    network: Network,
    test_property: SupportedProperties,
    abstraction_strategy: AbstractionStrategy,
    refinement_strategy: RefinementStrategy,
    output_folder: PathLike,
    experiment_options: ExperimentOptions,
    update_property: Optional[PropertyUpdateMethod] = None,
    do_preprocess: bool = True,
):
    output_folder = Path(output_folder)
    experiment_tracing_helper = ExperimentTracingHelper(experiment_options)

    experiment_time = Timer().start()

    logger.info("original network info", network_info=network_info_as_dict(network))
    logger.msg(
        "preparing network with property",
        property_type=str(type(test_property).__name__),
    )
    network, test_property = prepare_network_for_property(network, test_property)
    logger.info("property-reduced network info", network_info=network_info_as_dict(network))
    experiment_tracing_helper.save_network_and_property_as_code(
        output_folder / "network_prepared_for_property.py", network, test_property
    )

    if do_preprocess:
        logger.msg(f"preprocessing network (splitting neurons)")
        orig_net = preprocess(network)
    else:
        logger.msg(f"network has already been preprocessed")
        orig_net = deepcopy(network)

    # temporary solution for the changes in the property output neuron ids. TODO fix
    test_property = fix_property_after_preprocess(test_property)
    original_property = deepcopy(test_property)
    logger.info("preprocessed network info", network_info=network_info_as_dict(orig_net))
    experiment_tracing_helper.save_network_and_property_as_code(
        output_folder / "network_preprocessed.py", orig_net, test_property
    )
    if update_property is not None:
        property_updater = UpdatePropertyHelper(
            original_network=orig_net,
            original_property=original_property,
            update_method=update_property,
        )
    else:
        property_updater = None

    logger.msg(f"abstracting network")
    abs_net = abstract_network(orig_net, abstraction_strategy)
    most_abstract = deepcopy(abs_net)
    logger.info(
        "abstracted (initial-step) network info",
        network_info=network_info_as_dict(most_abstract),
    )

    logger.msg(f"starting refinement process")
    all_spurious_examples_found = []
    all_stats: List[RefinementStatitistics] = []

    experiment_final_marabou_status: MarabouValue = MarabouValue.UNKNOWN

    for step in count(1):
        step_logger = logger.bind(step=step)
        step_logger.msg(f"starting step")
        step_folder = output_folder / f"step_{step}"
        step_folder.mkdir(exist_ok=True, parents=True)

        if SAVE_NET:
            net_pk_path = step_folder / "net.pk"
            with open(net_pk_path, "wb") as fp:
                pickle.dump(abs_net, fp)
            with open(net_pk_path, "rb") as fp:
                pickle.load(fp)

        if property_updater is not None:
            step_logger.msg(f"updating property")
            test_property = property_updater.update_property(
                current_network=abs_net,
                output_folder=step_folder,
            )
            step_logger.info("updated property output constraints", updated_property=test_property.output_constraints)

        experiment_tracing_helper.save_network_and_property_as_code(step_folder / "network.py", abs_net, test_property)
        # query marabou
        mnet = ARMarabouNetwork(abs_net)
        marabou_log = str((step_folder / f"marabou-log").absolute())
        query_save_path = str((step_folder / f"query").absolute())
        step_logger.msg(
            "querying marabou",
            network_info=network_info_as_dict(abs_net),
            marabou_log=marabou_log,
            query_path=query_save_path,
        )
        with Timer() as t:
            query_result = query_marabou(
                mnet=mnet,
                property=test_property,
                marabou_log=marabou_log,
                query_save_path=query_save_path,
            )
        step_logger.msg("marabou query finished", time=t())
        step_logger.msg("marabou stats", **asdict(MarabouStats.read_marabou_stats(query_result.stats)))

        if query_result.value == MarabouValue.UNSAT:
            step_logger.msg("UNSAT in abstract => UNSAT in original. we're done!")
            experiment_final_marabou_status = MarabouValue.UNSAT
            break

        elif query_result.value == MarabouValue.SAT:
            satisfying_example_on_abs = cast_none(query_result.inputs_only)
            logger.info("counter-example on abs", values=list(satisfying_example_on_abs.items()))

            # verify that the counter-example is actually a counter-example
            satisfies, why = is_satisfying_assignment(abs_net, satisfying_example_on_abs, test_property)
            step_logger.info(
                "constraint satisfaction status",
                net="current network",
                is_satisfied=satisfies,
                why=why,
            )
            if not satisfies:
                net_output = abs_net.evaluate(satisfying_example_on_abs, return_as_dict=True)
                step_logger.debug(
                    "the example found is not satisfying. it's a bug",
                    marabou_query_assignment=list(query_result.vals_as_neurons.items()),
                    network_evaluated=list(net_output.items()),  # type:ignore
                    output_names=list(abs_net.output_ids),
                )
                return

            step_logger.msg("checking example on original")
            is_satisfying_on_orig, why = is_satisfying_assignment(
                orig_net, satisfying_example_on_abs, original_property
            )
            step_logger.info(
                "constraint satisfaction status",
                net="original network",
                is_satisfied=is_satisfying_on_orig,
                why=why,
            )

            if is_satisfying_on_orig:
                step_logger.msg(
                    "also SAT in original, we're done!",
                    counter_example=list(satisfying_example_on_abs.items()),
                )
                experiment_final_marabou_status = MarabouValue.SAT
                break
            else:
                step_logger.msg("it's a spurious example. we should continue with refining")
                all_spurious_examples_found.append(satisfying_example_on_abs)

            orig_net_activations = orig_net.evaluate(
                satisfying_example_on_abs,
                return_as_dict=True,
                return_intermediate=True,
            )

            step_logger.msg("doing refinement...")
            if experiment_options.refine_until_not_satisfying:
                abs_net, stats = refine_until_not_satisfying(
                    abs_net,
                    refinement_strategy,
                    spurious_examples=all_spurious_examples_found,
                    test_property=test_property,
                    update_property=property_updater.update_property if property_updater is not None else None,
                    # refinement kwargs
                    evaluated_network_activations=orig_net_activations,
                )
            else:
                abs_net, stats = refine_network(
                    abs_net,
                    refinement_strategy,
                    # refinement kwargs
                    spurious_example=satisfying_example_on_abs,
                    evaluated_network_activations=orig_net_activations,
                )
            stats_as_dict = {
                "steps": [asdict(s) for s in stats.steps],
            }
            step_logger.msg("refine step stats", **stats_as_dict)
            step_logger.msg("refinement done", network_info=network_info_as_dict(abs_net))
            all_stats.append(stats)

        else:
            step_logger.msg("got marabou exit status", marabou_exit_status=query_result.value)
            experiment_final_marabou_status = query_result.value
            break

    experiment_time = experiment_time.stop()

    logger.info(
        "summary",
        experiment_final_marabou_status=experiment_final_marabou_status,
        experiment_time=experiment_time,
        num_spurious_examples_found=len(all_spurious_examples_found),
        sum_refine_steps=sum([s.num_steps for s in all_stats]),
        refine_steps=([s.num_steps for s in all_stats]),
        neuron_per_step=[sum(s.num_neurons_refined) for s in all_stats],
    )

    logger.info("final network info", network_info=network_info_as_dict(abs_net))
