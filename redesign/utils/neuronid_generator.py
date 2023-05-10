from itertools import count
from ..datastructures import Sign, Scaling, NeuronId


class NeuronIdGenerator:
    def __init__(self, layer_index) -> None:
        self.counter = count(1)
        self.layer_index = layer_index

    def gen(self, sign: Sign, scaling: Scaling):
        # "ag" stands for auto-generated
        name = f"ag{self.layer_index}:{next(self.counter)}"
        return NeuronId(name, sign, scaling)


_generators = {}


def name_generator(layer_index) -> NeuronIdGenerator:
    return _generators.setdefault(layer_index, NeuronIdGenerator(layer_index))


def generate_neuronid(layer_index: int, sign: Sign, scaling: Scaling):
    return name_generator(layer_index).gen(sign, scaling)
gen_nid = generate_neuronid 
