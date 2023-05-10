from more_itertools import pairwise
import pandas as pd
import tensorflow.keras as keras

from ..datastructures import Network, ActivationFunction, BiasTable, NeuronId, WeightsTable


def network_from_keras_sequentail(model: keras.models.Sequential) -> Network:
    _, num_inputs = model.input_shape
    inputs = [NeuronId(f"x{i}") for i in range(num_inputs)]
    hidden = []
    for l_num, l in enumerate(model.layers[:-1], 1):
        c = l.get_config()
        _, b = l.get_weights()
        hidden.append([NeuronId(f"v{l_num}:{i}") for i in range(len(b))])
    _, num_outputs = model.output_shape
    outputs = [NeuronId(f"y{i}") for i in range(num_outputs)]

    net_weights = []
    all_layers = [inputs, *hidden, outputs]
    for (li, lo), l in zip(pairwise(all_layers), model.layers):
        w, _ = l.weights
        assert w.shape == (len(li), len(lo)), f"{w.shape} != {(len(li), len(lo))}"
        df = pd.DataFrame(w.numpy(), index=li, columns=lo)
        net_weights.append(WeightsTable(df))

    net_biases = []
    # model biases does not provide biases for input layer, but `Network` requires it
    net_biases.append(BiasTable({nid: 0 for nid in inputs}))
    for layer_ids, l in zip([*hidden, outputs], model.layers):
        _, b = l.weights
        assert b.shape == (len(layer_ids)), f"{b.shape} != {(len(layer_ids))}"
        bs = dict(zip(layer_ids, b.numpy()))
        net_biases.append(BiasTable(bs))

    activations = [ActivationFunction.Id] + [ActivationFunction.Relu] * (len(net_biases) - 2) + [ActivationFunction.Id]
    for act, l in zip(activations[1:], model.layers):
        as_tf_act_name = {ActivationFunction.Id: "linear", ActivationFunction.Relu: "relu"}[act]
        model_act = l.get_config()["activation"]
        assert as_tf_act_name == model_act, f"{as_tf_act_name} == {model_act}"

    net = Network(net_weights, net_biases, activations=activations)
    net.verify()

    return net
