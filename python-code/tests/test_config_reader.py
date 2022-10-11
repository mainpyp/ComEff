import pytest
from utils.config_reader import ConfigReader

CONFIG = ConfigReader("config_test_cifar10.yaml")

def test_assert_fail_if_wrong_path():
    with pytest.raises(AssertionError):
        ConfigReader("dummy.yaml")


def test_if_file_is_accepted():
    ConfigReader("config_test_cifar10.yaml")


def test_if_format_is_dict():
    assert type(CONFIG.data) is dict


