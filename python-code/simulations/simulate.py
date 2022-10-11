"""
    Copyright (C) 2019 Adrian Edward Thomas Henkel

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
import argparse
import sys
sys.path.insert(0, "../")
from utils.config_reader import ConfigReader
from simulations.simulator import Simulator


def parse_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    args = parser.parse_args()
    return args.config

if __name__ == '__main__':
    config_path = parse_path()
    print(f"Simulation starts with {config_path}")
    conf = ConfigReader(config_path)
    simulator = Simulator(config=conf)
    print("Start running simulation")
    simulator.run()
