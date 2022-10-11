import pytest
import numpy as np

from utils.config_reader import ConfigReader
from simulations.simulator import Simulator
from client_server.client import FederatedClient

CONFIG = ConfigReader("config_test_cifar10.yaml")



def test_create_simulator():
    assert Simulator(CONFIG) is not None


def test_process_data():
    simulator = Simulator(CONFIG)
    simulator.process_data()
    assert len(simulator.mnist_train_data_splits) == CONFIG.data["client_count"]


@pytest.mark.parametrize("model_type", ["tlp", "cnn", "tlcnn"])
def test_setup_model(model_type):
    simulator = Simulator(CONFIG)
    assert simulator.plain_model is None

    simulator.process_data()
    CONFIG.data["model_type"] = model_type
    simulator.setup_model()

    assert simulator.plain_model is not None


def test_setup_server():
    simulator = Simulator(CONFIG)
    simulator.process_data()  # test passed
    simulator.setup_model()  # test passed

    assert simulator.server is None
    simulator.setup_server()
    assert simulator.server is not None
    assert simulator.server.global_gradients is not None


@pytest.mark.parametrize("client_count", [1, 10, 50, 100, 1000])
def test_setup_clients(client_count):
    CONFIG.data["client_count"] = client_count
    simulator = Simulator(CONFIG)
    simulator.process_data()  # test passed
    simulator.setup_model()  # test passed

    assert len(simulator.clients) == 0
    simulator.setup_clients()
    assert len(simulator.clients) == CONFIG.data["client_count"]
