import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from math import floor
from numpy import random

#  DATASET/MODEL_ABBREVIATION
SETTINGS = "mnist/tlp"

"""
OUTPUT_PATH = f"/Users/adrianhenkel/Documents/Uni/com-eff/repos/com-eff/storage/produced-data/plots/{SETTINGS}"

NO_FED_PATH = f"../../storage/produced-data/logs/{SETTINGS}/non_fed_17.08.20-20:23:25.csv"
REGULAR_FED_PATH = f"../../storage/produced-data/logs/{SETTINGS}/regular_fed_06.12.20-12:22:19.csv"

UPDATES2_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_mlu/mlu2_17.08.20-11:43:38.csv"
UPDATES5_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_mlu/mlu5_17.08.20-15:28:22.csv"
UPDATES10_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_mlu/mlu10_17.08.20-16:11:28.csv"
UPDATES20_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_mlu/mlu20_17.08.20-17:38:08.csv"
UPDATES50_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_mlu/mlu50_17.08.20-22:03:55.csv"

QUANTIFIED_PATH = f"../../storage/produced-data/logs/{SETTINGS}/q_17.08.20-12:03:57.csv"
SPARSE_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse80_17.08.20-23:43:27.csv"
QUANTIFIED_UPDATES2_PATH = f"../../storage/produced-data/logs/{SETTINGS}/mlu2_q_17.08.20-12:19:49.csv"
QUANTIFIED_UPDATES2_SPARSE_PATH = f"../../storage/produced-data/logs/{SETTINGS}/mlu2_q_sparse80_18.08.20-09:37:24.csv"
QUANTIFIED_SPARSE_PATH = f"../../storage/produced-data/logs/{SETTINGS}/q_sparse90_17.08.20-20:33:03.csv"
UPDATES2_SPARSE_PATH = f"../../storage/produced-data/logs/{SETTINGS}/mlu2_sparse80_18.08.20-11:02:27.csv"

SPARSE0_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse0_17.08.20-15:17:57.csv"
SPARSE10_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse10_17.08.20-16:29:45.csv"
SPARSE20_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse20_17.08.20-17:42:36.csv"
SPARSE30_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse30_17.08.20-18:54:54.csv"
SPARSE40_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse40_17.08.20-20:06:37.csv"
SPARSE50_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse50_17.08.20-21:18:26.csv"
SPARSE60_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse60_17.08.20-22:29:14.csv"
SPARSE70_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse70_19.08.20-09:59:45.csv"
SPARSE80_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse80_17.08.20-23:43:27.csv"
SPARSE90_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse90_18.08.20-00:52:58.csv"
SPARSE95_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse95_18.08.20-02:00:10.csv"
SPARSE99_PATH = f"../../storage/produced-data/logs/{SETTINGS}/only_sparse/sparse99_18.08.20-03:06:42.csv"

"""
OUTPUT_PATH = f"/Users/adrianhenkel/Documents/Uni/com-eff/repos/com-eff/storage/produced-data/plots/fashion_mnist/cnn"

NO_FED_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/non_federated.csv"
REGULAR_FED_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/regular_fed_15.11.20-18:38:50.csv"

UPDATES2_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_mlu/mlu2_16.11.20-17:25:49.csv"
UPDATES5_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_mlu/mlu5_17.11.20-12:58:48.csv"
UPDATES10_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_mlu/mlu10_16.11.20-22:43:26.csv"
UPDATES20_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_mlu/mlu20_16.11.20-17:59:38.csv"
UPDATES50_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_mlu/mlu50_17.11.20-01:12:02.csv"

QUANTIFIED_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/q_16.11.20-03:46:00.csv"
SPARSE_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse80_20.11.20-12:01:12.csv"
QUANTIFIED_UPDATES2_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/mlu2_q_16.11.20-03:11:39.csv"
QUANTIFIED_UPDATES2_SPARSE_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/mlu2_q_sparse80_16.11.20-04:08:10.csv"
QUANTIFIED_SPARSE_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/q_sparse80_15.11.20-10:43:25.csv"
UPDATES2_SPARSE_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/mlu2_sparse80_15.11.20-18:58:59.csv"

SPARSE0_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse0_18.11.20-02:35:21.csv"
SPARSE10_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse10_17.11.20-18:00:21.csv"
SPARSE20_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse20_19.11.20-19:10:42.csv"
SPARSE30_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse30_20.11.20-03:41:04.csv"
SPARSE40_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse40_19.11.20-10:52:25.csv"
SPARSE50_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse50_19.11.20-02:44:27.csv"
SPARSE60_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse60_18.11.20-11:04:15.csv"
SPARSE70_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse70_20.11.20-19:53:56.csv"
SPARSE80_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse80_20.11.20-12:01:12.csv"
SPARSE90_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse90_21.11.20-03:55:45.csv"
SPARSE95_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse95_17.11.20-09:56:30.csv"
SPARSE99_PATH = f"../../storage/produced-data/logs/fashion_mnist/cnn/only_sparse/sparse99_18.11.20-19:05:47.csv"

NO_FED_DF = pd.read_csv(NO_FED_PATH, sep=";")
REGULAR_FED_DF = pd.read_csv(REGULAR_FED_PATH, sep=";")

UPDATES2_DF = pd.read_csv(UPDATES2_PATH, sep=";")
UPDATES5_DF = pd.read_csv(UPDATES5_PATH, sep=";")
UPDATES10_DF = pd.read_csv(UPDATES10_PATH, sep=";")
UPDATES20_DF = pd.read_csv(UPDATES20_PATH, sep=";")
UPDATES50_DF = pd.read_csv(UPDATES50_PATH, sep=";")


QUANTIFIED_DF = pd.read_csv(QUANTIFIED_PATH, sep=";")
SPARSE_DF = pd.read_csv(SPARSE_PATH, sep=";")
QUANTIFIED_UPDATES2_DF = pd.read_csv(QUANTIFIED_UPDATES2_PATH, sep=";")
QUANTIFIED_UPDATES2_SPARSE_DF = pd.read_csv(QUANTIFIED_UPDATES2_SPARSE_PATH, sep=";")
QUANTIFIED_SPARSE_DF = pd.read_csv(QUANTIFIED_SPARSE_PATH, sep=";")
UPDATES2_SPARSE_DF = pd.read_csv(UPDATES2_SPARSE_PATH, sep=";")

SPARSE0_DF = pd.read_csv(SPARSE0_PATH, sep=";")
SPARSE10_DF = pd.read_csv(SPARSE10_PATH, sep=";")
SPARSE20_DF = pd.read_csv(SPARSE20_PATH, sep=";")
SPARSE30_DF = pd.read_csv(SPARSE30_PATH, sep=";")
SPARSE40_DF = pd.read_csv(SPARSE40_PATH, sep=";")
SPARSE50_DF = pd.read_csv(SPARSE50_PATH, sep=";")
SPARSE60_DF = pd.read_csv(SPARSE60_PATH, sep=";")
SPARSE70_DF = pd.read_csv(SPARSE70_PATH, sep=";")
SPARSE80_DF = pd.read_csv(SPARSE80_PATH, sep=";")
SPARSE90_DF = pd.read_csv(SPARSE90_PATH, sep=";")
SPARSE95_DF = pd.read_csv(SPARSE95_PATH, sep=";")
SPARSE99_DF = pd.read_csv(SPARSE99_PATH, sep=";")

ALL_DATAFRAMES = {
    "No federated": NO_FED_DF,
    "Regular federated": REGULAR_FED_DF,
    "2 local updates": UPDATES2_DF,
    "Quantified": QUANTIFIED_DF,
    "Sparse": SPARSE_DF,
    "Quantified, 2 local updates": QUANTIFIED_UPDATES2_DF,
    "Quantified, sparse": QUANTIFIED_SPARSE_DF,
    "2 local updates, sparse": UPDATES2_SPARSE_DF,
    "Quantified, sparse, 2 local updates": QUANTIFIED_UPDATES2_SPARSE_DF,
}

LEGEND = [
    "No federated",
    "Regular federated",
    "2 local updates",
    "Quantified",
    "Sparse",
    "Quantified and 2 local updates",
    "Quantified, sparse",
    "2 local updates, sparse",
    "Quantified, sparse, 2 local updates",
]

ALL_MLU_DATAFRAMES = {"1": REGULAR_FED_DF,
                      "2": UPDATES2_DF,
                      "5": UPDATES5_DF,
                      "10": UPDATES10_DF,
                      "20": UPDATES20_DF,
                      "50": UPDATES50_DF}

LEGEND_MLU = [
    "regular federated",
    "2 local updates",
    "5 local updates",
    "10 local updates",
    "20 local updates",
    "50 local updates"
]

ALL_SPARSE = {
    "0": SPARSE0_DF,
    "10": SPARSE10_DF,
    "20": SPARSE20_DF,
    "30": SPARSE30_DF,
    "40": SPARSE40_DF,
    "50": SPARSE50_DF,
    "60": SPARSE60_DF,
    "70": SPARSE70_DF,
    "80": SPARSE80_DF,
    "90": SPARSE90_DF,
    "95": SPARSE95_DF,
    "99": SPARSE99_DF
}

LEGEND_SPARSE = [
    "0 - percentile",
    "10 - percentile",
    "20 - percentile",
    "30 - percentile",
    "40 - percentile",
    "50 - percentile",
    "60 - percentile",
    "70 - percentile",
    "80 - percentile",
    "90 - percentile",
    "95 - percentile",
    "99 - percentile"
]


def plot_compare_approaches(save: bool=False,
                            start_with_zero: bool=False,
                            palette: str="bright",
                            line_at:list=None,
                            suffix: str=None):
    """
    This method plots the comparison of all three approaches.
    quantified 2, sparse 80 and mlu 2 is used
    :return: None
    """
    plot_name = "compare_all_approaches_acc_annotated"
    sns.set(style="darkgrid")
    sns.set_palette(palette=palette)

    first_row = pd.DataFrame({"Iteration": [0,], "Accuracy": [0.0,]})
    for df_name in ALL_DATAFRAMES:
        #  ALL_DATAFRAMES[df_name] = ALL_DATAFRAMES[df_name].apply(np.log10)
        #  ALL_DATAFRAMES[df_name] = ALL_DATAFRAMES[df_name].iloc[250:]
        if start_with_zero:
            start_with_zero = pd.concat(
                [first_row, ALL_DATAFRAMES[df_name]], ignore_index=True
            )
        sns.lineplot(x="Iteration", y="Accuracy", data=ALL_DATAFRAMES[df_name])
    plt.legend(LEGEND)
    plt.grid(color="white", linestyle="-", linewidth=1, axis="y")
    if line_at:
        for line in line_at:
            plt.plot([x for x in range(5, 95)], [line for _ in range(5, 95)], "--", c="grey")
            plt.text(25, (line + 0.005), f"Accuracy {round(line, 2)}%", c="black", alpha=0.5)
    plt.title("Compare different combinations of approaches")
    if save:
        if not suffix:
            plt.savefig(f"{OUTPUT_PATH}/{plot_name}.png")
        else:
            plt.savefig(f"{OUTPUT_PATH}/{plot_name}_{suffix}.png")
    plt.show()


def plot_mlu_compare(save: bool = False,
                     start_with_zero: bool = False,
                     line_at: float = None,
                     palette: str = "bright",
                     suffix: str = None):
    plot_name = "compare_mlu"
    sns.set(style="darkgrid")
    sns.set_palette(palette=palette)
    first_row = pd.DataFrame({"Iteration": [0,], "Accuracy": [0.0,]})
    for df_name in ALL_MLU_DATAFRAMES:
        #  ALL_DATAFRAMES[df_name] = ALL_DATAFRAMES[df_name].apply(np.log10)
        #  ALL_DATAFRAMES[df_name] = ALL_DATAFRAMES[df_name].iloc[250:]
        if start_with_zero:
            start_with_zero = pd.concat(
                [first_row, ALL_MLU_DATAFRAMES[df_name]], ignore_index=True
            )
        sns.lineplot(x="Iteration", y="Accuracy", data=ALL_MLU_DATAFRAMES[df_name])
    plt.legend(LEGEND_MLU)
    plt.grid(color="white", linestyle="-", linewidth=1, axis="y")
    if line_at:
        for line in line_at:
            plt.plot([x for x in range(5, 95)], [line for _ in range(5, 95)], "--", c="grey")
            plt.text(25, (line + 0.005), f"Accuracy {round(line, 2)}%", c="black", alpha=0.5)
    plt.title("Compare amount of local iterations")
    if save:
        if not suffix:
            plt.savefig(f"{OUTPUT_PATH}/{plot_name}.png")
        else:
            plt.savefig(f"{OUTPUT_PATH}/{plot_name}_{suffix}.png")
    plt.show()


def plot_sparse_compare(save: bool=False,
                        every_second_file: tuple=None,
                        start_with_zero: bool=False,
                        line_at:float=None,
                        palette: str="bright",
                        suffix: str=None):
    skip = every_second_file[0]
    start_at = every_second_file[1]
    add_to_index = every_second_file[2]

    plot_name = "compare_sparsification10+20"
    sns.set(style="darkgrid")
    sns.set_palette(palette=palette)
    first_row = pd.DataFrame({"Iteration": [0, ], "Accuracy": [0.0, ]})
    plt.grid(color="white", linestyle="-", linewidth=1, axis="y")
    plt.title("Comparison of sparsification percentiles")

    already_added = False
    for index, df_name in enumerate(list(ALL_SPARSE)):
        if not already_added:
            index += add_to_index
            already_added = True

        if start_with_zero:
            start_with_zero = pd.concat(
                [first_row, ALL_MLU_DATAFRAMES[df_name]], ignore_index=True
            )

        if skip and index % start_at == 0:
            sns.lineplot(x="Iteration", y="Accuracy", data=ALL_SPARSE[df_name])
            continue

    if skip:
        plt.legend(LEGEND_SPARSE[add_to_index::start_at])
    else:
        plt.legend(LEGEND_SPARSE)


    if line_at:
        for line in line_at:
            plt.plot([x for x in range(5, 95)], [line for _ in range(5, 95)], "--", c="grey")
            plt.text(25, (line + 0.005), f"Accuracy {round(line, 2)}%", c="black", alpha=0.5)

    if save:
        if not suffix:
            plt.savefig(f"{OUTPUT_PATH}/{plot_name}.png")
        else:
            plt.savefig(f"{OUTPUT_PATH}/{plot_name}_{suffix}.png")

    plt.show()

def mlu_iterations():
    for key in ALL_MLU_DATAFRAMES:
        print(f"{key} : ")
        print(len(ALL_MLU_DATAFRAMES[key][ALL_MLU_DATAFRAMES[key].Accuracy < 0.97 ]))


def plot_heatmap():
    heat = np.array([
        [0.5, 0.25, 0.125],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5]
    ])
    mask = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, False]
    ])
    sns.heatmap(heat, linewidths=0.5, linecolor="w", annot=True, mask=mask, square=True, cmap="Blues",
                cbar_kws={'label': 'Data traffic in GB'})
    plt.xticks([0.5, 1.5, 2.5], ["gradQ", "MCsU", "gradS"])
    plt.yticks([0.5, 1.5, 2.5], ["gradQ", "MCsU", "gradS"])
    for label in plt.gca().yaxis.get_majorticklabels():
        print(label.get_position()[1])
        label.set_position((0, label.get_position()[1] + 0.5))
        print(label.get_position()[1])
    #plt.savefig("data_traffic_in_gb.png")
    plt.show()


def plot_dirichlet(dirichlet_params: list, n_clients, total_data, color=None, name=None):
    amounts = []
    for param in dirichlet_params:
        distribution = np.random.dirichlet(np.ones(n_clients) / param, size=1)[0]
        amount_of_data = [floor(percentage * total_data) for percentage in distribution]
        if sum(amount_of_data) != n_clients:
            delta = n_clients - sum(amount_of_data)
            for _ in range(delta):
                random_index = random.randint(0, len(distribution) - 1)
                amount_of_data[random_index] += 1
        amounts.append(amount_of_data)

    plt.figure(figsize=(20, len(dirichlet_params)+3))
    x_data = list(range(n_clients))
    for index, (a, param) in enumerate(zip(amounts, dirichlet_params)):
        plt.subplot(131 + index)
        if index is 0:
            plt.ylabel("Amount of data", fontsize=16)
        elif index is 1:
            plt.xlabel("Number of clients", fontsize=16)
        if not color:
            splot = sns.barplot(x_data, a)
        else:
            splot = sns.barplot(x_data, a, palette=color)
        for i, label in enumerate(splot.xaxis.get_ticklabels()):
            if not i % (n_clients / 10) == 0:
                if i == n_clients:
                    continue
                label.set_visible(False)
        plt.title(f"Parameter = {param}")

    plt.suptitle('Distribution of Client Data with different dirichlet parameters', fontsize=18)
    if name:
        plt.savefig(f"../../storage/produced-data/plots/other/{name}")
    plt.show()


if __name__ == "__main__":
    heat_palette = [
        "#ff0000",
        "#ffbb00",
        "#ffff00",
        "#bfff00",
        "#6fff00",
        "#00ff2f",
        "#00ffd5",
        "#00bfff",
        "#0059ff",
        "#1e00ff",
        "#7b00ff",
        "#e600ff",
        "#ff00bf"
    ]

    palette_choices = [
        "deep",
        "muted",
        "pastel",
        "bright",
        "dark",
        "colorblind"
    ]
    plot_heatmap()