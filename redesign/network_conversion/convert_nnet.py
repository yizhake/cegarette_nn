from dataclasses import dataclass
from typing import Dict, List
from maraboupy.MarabouNetworkNNet import MarabouNetworkNNet
from more_itertools import pairwise
from ..datastructures import ActivationFunction, BiasTable, Network, NeuronId, WeightsTable
from ..marabou_properties.basic_property import LowerBound, UpperBound


def input_bounds_from_nnet(
    nnet: MarabouNetworkNNet, variable_to_neuron_id: Dict[int, NeuronId]
):
    input_bounds = []
    for var, nid in variable_to_neuron_id.items():
        input_bounds.append(LowerBound(nid, nnet.getInputMinimum(var)))
        input_bounds.append(UpperBound(nid, nnet.getInputMaximum(var)))

    return input_bounds


def network_from_nnet(nnet: MarabouNetworkNNet) -> Network:
    inputs = [NeuronId(f"x{i}") for i in range(nnet.layerSizes[0])]
    hidden = []
    for l, ls in enumerate(nnet.layerSizes[1:-1], 1):
        hidden.append([NeuronId(f"v{l}:{i}") for i in range(ls)])
    outputs = [NeuronId(f"y{i}") for i in range(nnet.layerSizes[-1])]

    net_weights = []
    all_layers = [inputs, *hidden, outputs]
    for (il1, l1), (_, l2) in pairwise(enumerate(all_layers)):
        weights = {
            s: {d: nnet.weights[il1][di][si] for di, d in enumerate(l2)}
            for si, s in enumerate(l1)
        }
        net_weights.append(WeightsTable(weights))

    net_biases = []
    # nnet.biases does not provide biases for input layer, but `Network` requires it
    net_biases.append(BiasTable({nid: 0 for nid in inputs}))
    for layer_ids, layer_biases in zip([*hidden, outputs], nnet.biases):
        bs = dict(zip(layer_ids, layer_biases))
        net_biases.append(BiasTable(bs))

    activations = [ActivationFunction.Id] + [ActivationFunction.Relu] * (len(net_biases) - 2) + [ActivationFunction.Id]
    net = Network(net_weights, net_biases, activations=activations)
    net.verify()

    return net


@dataclass
class NNetOptions:
    inputMinimums: List[float]
    inputMaximums: List[float]
    inputMeans: List[float]
    inputRanges: List[float]
    outputMean: float
    outputRange: float


def nnet_from_network(net: Network, nnet_options: NNetOptions) -> MarabouNetworkNNet:
    nnet = MarabouNetworkNNet()

    numLayers = len(net.weights)
    layerSizes = [len(l.nodes) for l in net.layers]
    maxLayersize = max(layerSizes)
    inputSize = layerSizes[0]
    outputSize = layerSizes[-1]

    weights = []
    for w in net.weights:
        weights.append(w.table.values.T.copy())
    biases = []
    # nnet skips input biases
    for b in net.biases[1:]:
        biases.append(b.table.values.T.copy())

    nnet.numLayers = numLayers
    nnet.layerSizes = layerSizes
    nnet.inputSize = inputSize
    nnet.outputSize = outputSize
    nnet.maxLayersize = maxLayersize
    nnet.inputMinimums = nnet_options.inputMinimums
    nnet.inputMaximums = nnet_options.inputMaximums
    nnet.inputMeans = nnet_options.inputMeans
    nnet.inputRanges = nnet_options.inputRanges
    nnet.outputMean = nnet_options.outputMean
    nnet.outputRange = nnet_options.outputRange
    nnet.weights = weights
    nnet.biases = biases

    nnet.computeNetworkAttributes()

    return nnet

