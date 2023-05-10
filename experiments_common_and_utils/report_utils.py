from redesign.datastructures import Network


def network_info_as_dict(net: Network) -> dict:
    return dict(
        layers=[len(l.nodes) for l in net.layers],
        hidden_nodes=sum([len(l.nodes) for l in net.layers[1:-1]]),
    )


def pformat_network_info(net: Network) -> str:
    s = ""
    s += f"layers: {[len(l.nodes) for l in net.layers]}\n"
    s += f"hidden nodes: {sum([len(l.nodes) for l in net.layers[1:-1]])}\n"
    return s

def report_constraint(net_name: str, satisfied: bool, why: str) -> str:
    if satisfied:
        return f"constraint satisfied on {net_name}: {why}"
    else:
        return f"constraint unsatisfied on {net_name}: {why}"
