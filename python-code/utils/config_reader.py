"""
    Copyright (C) 2019 Adrian Edward Thomas Henkel, Reza NasiriGerdeh

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Author: Adrian Edward Thomas Henkel
"""
import os
import yaml


class ConfigReader:
    path: str

    def __init__(self, path: str):
        assert os.path.exists(path), "Config file is not a file"
        assert path.endswith("yaml"), "Config file has not the yaml suffix"
        self.path = path
        self.data = self.parse_config()

    def parse_config(self):
        """
        Opens the yaml file and processes it

        Returns
        -------
        data : dict
            Data is a dictionary with the yaml file data
        """
        with open(self.path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data


if __name__ == "__main__":
    config_reader = ConfigReader(
        "../configs/fashion_mnist/100clients/mlu2_sparse80.yaml"
    )
