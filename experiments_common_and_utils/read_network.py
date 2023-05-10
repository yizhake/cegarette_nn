from enum import Enum
from maraboupy.MarabouNetworkNNet import MarabouNetworkNNet
from redesign.datastructures import Network
from redesign.nnet.nnet_conversions import network_from_nnet


class SupportedNetworks(str, Enum):
    nnet = "nnet"
    tf = "tf"

    def get_reader(self):
        return {
            SupportedNetworks.nnet: read_nnet,
            SupportedNetworks.tf: read_tf,
        }[self]


def read_nnet(nnet_path) -> Network:
    nnet = MarabouNetworkNNet(nnet_path)
    network = network_from_nnet(nnet)
    return network


def read_tf(tf_path) -> Network:
    import tensorflow.keras as keras
    from redesign.network_conversion.convert_tensorflow import network_from_keras_sequentail

    model = keras.models.load_model(tf_path)
    net = network_from_keras_sequentail(model)
    return net


def read_network(network_type, network_path):
    return SupportedNetworks[network_type].get_reader()(network_path)
