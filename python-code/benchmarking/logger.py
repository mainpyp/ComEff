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
import pandas as pd
from datetime import datetime
from os import path
from typing import Dict
from utils.config_reader import ConfigReader

TIMESTAMP = datetime.now().strftime("%d.%m.%y-%H:%M:%S")
COLUMNS = ["Iteration", "Accuracy", "Loss", "Grads"]


class Logger:
    def __init__(self, config: ConfigReader, sep: str = None):
        # format round: value
        self.data: pd.DataFrame
        self.data = pd.DataFrame(columns=COLUMNS)
        self.output_path = config.data["output_path"]
        self.sep = sep
        self.quantification_flag = config.data["quantification_flag"]
        self.quantification_options = config.data["quantification_options"]
        self.more_local_updates_flag = config.data["more_local_updates_flag"]
        self.more_local_updates_options = config.data["more_local_updates_options"]
        self.sparsification_flag = config.data["sparsification_flag"]
        self.sparsification_options = config.data["sparsification_options"]
        self.export_path = self._decide_filename()
        assert path.exists(self.output_path), f"The export destination is not valid {self.output_path}"


    def add_metrics(self, iteration: int, accuracy: float, loss:float, grads: int):
        """Adds an entry to the dataframe. It is asserted that no entry for one iterations
            exists

        Parameters
        ----------
        iteration
            The number of iteration
        accuracy
            The accuracy of one iteration
        """
        assert (
            iteration not in self.data.index
        ), f"Error: Iteration {iteration} already exists in the data"
        row = [int(iteration), accuracy, loss, grads]
        self.data.loc[iteration] = row
        self.export_data()

    def get_data(self) -> pd.DataFrame:
        return self.data

    def export_data(self):
        print(f"Exporting data to {self.export_path}")
        if self.sep:
            self.data.to_csv(path_or_buf=self.export_path, sep=self.sep, index=False)
        else:
            self.data.to_csv(path_or_buf=self.export_path, sep=";", index=False)

    def _decide_filename(self, name: str = None):
        """Decides a suited filename. To make the name unique, a timestamp is added as a suffix.
            mlu[n] -> n more local updates
            q -> quantified weights
            sparse[n] -> The nth percentile was used for sending the data

        Parameters
        ----------
        name
            If given, this name is used instead of determine the suited name

        """


        filename = ""
        if self.more_local_updates_flag:
            filename += f"mlu{self.more_local_updates_options['local_iterations']}"
        if self.quantification_flag:
            filename += "_q_"
        if self.sparsification_flag:
            filename += f"_sparse{self.sparsification_options['percentile']}"
        if not (self.sparsification_flag or self.more_local_updates_flag or self.quantification_flag):
            filename += "regular_fed"

        if name:
            filename = name


        if filename.startswith("_"):
            filename = filename[1:]

        print(f"{self.output_path}{filename}_{TIMESTAMP}.csv")
        return f"{self.output_path}{filename}_{TIMESTAMP}.csv".replace("__", "_")

    def set_filename(self, name: str):
        self.export_path = self._decide_filename(name)
        print(f"New filename: {self.export_path}")
