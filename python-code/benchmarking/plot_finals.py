import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_PATH = f"/Users/adrianhenkel/Documents/Uni/com-eff/repos/com-eff/storage/produced-data/plots/further_work"

###########
# CIFAR10 #
###########
NON_FED_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/non_federated_26.02.21-12:09:15.csv"
REGULAR_FED_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/regular_fed_24.02.21-22:31:12.csv"

MLU2_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/only_mlu/mlu2_25.02.21-11:07:56.csv"
MLU10_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/only_mlu/mlu10_25.02.21-14:45:31.csv"
MLU20_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/only_mlu/mlu20_25.02.21-11:40:26.csv"

SPARSE10_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/only_sparse/sparse10_08.03.21-16:29:50.csv"
SPARSE50_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/only_sparse/sparse50_09.03.21-01:11:21.csv"
SPARSE90_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/only_sparse/sparse90_09.03.21-09:16:54.csv"

SPARSE50_MLU10_Q_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/mlu10_q_sparse50_01.03.21-19:29:53.csv"
SPARSE50_MLU2_Q_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/mlu2_q_sparse50_09.03.21-10:18:09.csv"

Q_CIFAR = f"../../storage/produced-data/logs/final_run/cifar10/tlcnn/q_25.02.21-03:53:21.csv"

# Dataframes
NON_FED_CIFAR_DF = pd.read_csv(NON_FED_CIFAR, sep=";")
REGULAR_FED_CIFAR_DF = pd.read_csv(REGULAR_FED_CIFAR, sep=";")

MLU2_CIFAR_DF = pd.read_csv(MLU2_CIFAR, sep=";")
MLU10_CIFAR_DF = pd.read_csv(MLU10_CIFAR, sep=";")
MLU20_CIFAR_DF = pd.read_csv(MLU20_CIFAR, sep=";")

SPARSE10_CIFAR_DF = pd.read_csv(SPARSE10_CIFAR, sep=";")
SPARSE50_CIFAR_DF = pd.read_csv(SPARSE50_CIFAR, sep=";")
SPARSE90_CIFAR_DF = pd.read_csv(SPARSE90_CIFAR, sep=";")

SPARSE50_MLU10_Q_CIFAR_DF = pd.read_csv(SPARSE50_MLU10_Q_CIFAR, sep=";")
SPARSE50_MLU2_Q_CIFAR_DF = pd.read_csv(SPARSE50_MLU2_Q_CIFAR, sep=";")

Q_CIFAR_DF = pd.read_csv(Q_CIFAR, sep=";")

###########
# FASHION #
###########
NON_FED_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/non_federated_24.02.21-16:55:50.csv"
REGULAR_FED_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/regular_fed_16.02.21-08:49:37.csv"

MLU2_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/only_mlu/mlu2_15.02.21-17:54:10.csv"
MLU10_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/only_mlu/mlu10_15.02.21-19:01:24.csv"
MLU20_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/only_mlu/mlu20_17.02.21-04:46:50.csv"

SPARSE10_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/only_sparse/sparse10_08.03.21-16:30:34.csv"
SPARSE50_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/only_sparse/sparse50_08.03.21-18:52:07.csv"
SPARSE90_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/only_sparse/sparse90_08.03.21-21:07:57.csv"

SPARSE50_MLU10_Q_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/mlu10_q_sparse50_09.03.21-10:04:57.csv"

Q_FASHION = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/q_23.02.21-12:18:56.csv"


# Dataframes
NON_FED_FASHION_DF = pd.read_csv(NON_FED_FASHION, sep=";")
REGULAR_FED_FASHION_DF = pd.read_csv(REGULAR_FED_FASHION, sep=";")

MLU2_FASHION_DF = pd.read_csv(MLU2_FASHION, sep=";")
MLU10_FASHION_DF = pd.read_csv(MLU10_FASHION, sep=";")
MLU20_FASHION_DF = pd.read_csv(MLU20_FASHION, sep=";")

SPARSE10_FASHION_DF = pd.read_csv(SPARSE10_FASHION, sep=";")
SPARSE50_FASHION_DF = pd.read_csv(SPARSE50_FASHION, sep=";")
SPARSE90_FASHION_DF = pd.read_csv(SPARSE90_FASHION, sep=";")

SPARSE50_MLU10_Q_FASHION_DF = pd.read_csv(SPARSE50_MLU10_Q_FASHION, sep=";")

Q_FASHION_DF = pd.read_csv(Q_FASHION, sep=";")

# Non IID

NON_FED_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/cnn/non_federated_24.02.21-16:55:50.csv"
REGULAR_FED_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/regular_fed_25.02.21-23:11:03.csv"

MLU2_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/only_mlu/mlu2_26.02.21-15:52:25.csv"
MLU10_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/only_mlu/mlu10_28.02.21-01:37:59.csv"
MLU20_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/only_mlu/mlu20_26.02.21-16:58:31.csv"

SPARSE10_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/only_sparse/sparse10_26.02.21-09:12:57.csv"
SPARSE50_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/only_sparse/sparse50_26.02.21-11:30:31.csv"
SPARSE90_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/only_sparse/sparse90_26.02.21-13:39:32.csv"

SPARSE50_MLU10_Q_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/mlu10_q_sparse50_25.02.21-23:46:53.csv"

Q_FASHION_NON_IID = f"../../storage/produced-data/logs/final_run/fashion_mnist/non_iid/q_26.02.21-06:32:34.csv"

NON_FED_FASHION_DF_NON_IID = pd.read_csv(NON_FED_FASHION_NON_IID, sep=";")
REGULAR_FED_FASHION_DF_NON_IID = pd.read_csv(REGULAR_FED_FASHION_NON_IID, sep=";")

MLU2_FASHION_DF_NON_IID = pd.read_csv(MLU2_FASHION_NON_IID, sep=";")
MLU10_FASHION_DF_NON_IID = pd.read_csv(MLU10_FASHION_NON_IID, sep=";")
MLU20_FASHION_DF_NON_IID = pd.read_csv(MLU20_FASHION_NON_IID, sep=";")

SPARSE10_FASHION_DF_NON_IID = pd.read_csv(SPARSE10_FASHION_NON_IID, sep=";")
SPARSE50_FASHION_DF_NON_IID = pd.read_csv(SPARSE50_FASHION_NON_IID, sep=";")
SPARSE90_FASHION_DF_NON_IID = pd.read_csv(SPARSE90_FASHION_NON_IID, sep=";")

SPARSE50_MLU10_Q_FASHION_DF_NON_IID = pd.read_csv(SPARSE50_MLU10_Q_FASHION_NON_IID, sep=";")

Q_FASHION_DF_NON_IID = pd.read_csv(Q_FASHION_NON_IID, sep=";")

##############
# COLORECTAL #
##############
NON_FED_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/non_federated_01.03.21-17:58:53.csv"
REGULAR_FED_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/regular_fed_11.03.21-11:52:04.csv"

MLU2_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/only_mlu/mlu2_11.03.21-21:55:56.csv"
MLU5_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/only_mlu/mlu5_13.03.21-22:18:11.csv"
MLU10_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/only_mlu/mlu10_10.03.21-19:00:36.csv"
MLU20_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/only_mlu/mlu20_11.03.21-10:11:14.csv"

SPARSE10_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/only_sparse/sparse10_13.03.21-15:42:06.csv"
SPARSE50_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/sparse50_14.03.21-09:47:38.csv"
SPARSE90_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/only_sparse/sparse90_13.03.21-15:42:17.csv"

SPARSE50_MLU5_Q_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/mlu5_q_sparse50_10.03.21-17:01:33.csv"

Q_COLORECTAL = f"../../storage/produced-data/logs/final_run/colorectal/vgg16/q_11.03.21-14:59:13.csv"

NON_FED_COLORECTAL_DF = pd.read_csv(NON_FED_COLORECTAL, sep=";")
REGULAR_FED_COLORECTAL_DF = pd.read_csv(REGULAR_FED_COLORECTAL, sep=";")

MLU2_COLORECTAL_DF = pd.read_csv(MLU2_COLORECTAL, sep=";")
MLU5_COLORECTAL_DF = pd.read_csv(MLU5_COLORECTAL, sep=";")
MLU10_COLORECTAL_DF = pd.read_csv(MLU10_COLORECTAL, sep=";")
MLU20_COLORECTAL_DF = pd.read_csv(MLU20_COLORECTAL, sep=";")

SPARSE10_COLORECTAL_DF = pd.read_csv(SPARSE10_COLORECTAL, sep=";")
SPARSE50_COLORECTAL_DF = pd.read_csv(SPARSE50_COLORECTAL, sep=";")
SPARSE90_COLORECTAL_DF = pd.read_csv(SPARSE90_COLORECTAL, sep=";")

SPARSE50_MLU5_Q_COLORECTAL_DF = pd.read_csv(SPARSE50_MLU5_Q_COLORECTAL, sep=";")

Q_COLORECTAL_DF = pd.read_csv(Q_COLORECTAL, sep=";")

DATAFRAMES = {
    "fashion": {
        "sparse":
            {
                "P=10": SPARSE10_FASHION_DF,
                "P=50": SPARSE50_FASHION_DF,
                "P=90": SPARSE90_FASHION_DF,
            },
        "mlu":
            {
                "E=2": MLU2_FASHION_DF,
                "E=10": MLU10_FASHION_DF,
                "E=20": MLU20_FASHION_DF
            },
        "other":
            {
                "GQ + GS(P=50) + MU(E=10)": SPARSE50_MLU10_Q_FASHION_DF,
                "GQ": Q_FASHION_DF,
                "regular": REGULAR_FED_FASHION_DF,
                "Centralized": NON_FED_FASHION_DF
            }
    },
    "cifar10": {
        "sparse":
            {
                "P=10": SPARSE10_CIFAR_DF,
                "P=50": SPARSE50_CIFAR_DF,
                "P=90": SPARSE90_CIFAR_DF
            },
        "mlu":
            {
                "E=2": MLU2_CIFAR_DF,
                "E=10": MLU10_CIFAR_DF,
                "E=20": MLU20_CIFAR_DF
            },
        "other":
            {
                "GQ + GS(P=50) + MU(E=10)": SPARSE50_MLU10_Q_CIFAR_DF,
                "GQ + GS(P=50) + MU(E=2)": SPARSE50_MLU2_Q_CIFAR_DF,
                "GQ": Q_CIFAR_DF,
                "regular": REGULAR_FED_CIFAR_DF,
                "Centralized": NON_FED_CIFAR_DF
            }
    },
    "colorectal": {
        "sparse":
            {
                "P=10": SPARSE10_COLORECTAL_DF,
                "P=50": SPARSE50_COLORECTAL_DF,
                "P=90": SPARSE90_COLORECTAL_DF
            },
        "mlu":
            {
                "E=2": MLU2_COLORECTAL_DF,
                "E=5": MLU5_COLORECTAL_DF,
                "E=10": MLU10_COLORECTAL_DF,
                "E=20": MLU20_COLORECTAL_DF
            },
        "other":
            {
                "GQ + GS(P=50) + MU(E=5)": SPARSE50_MLU5_Q_COLORECTAL_DF,
                "GQ": Q_COLORECTAL_DF,
                "regular": REGULAR_FED_COLORECTAL_DF,
                "Centralized": NON_FED_COLORECTAL_DF
            }
    },
"fashion_non_iid": {
        "sparse":
            {
                "P=10": SPARSE10_FASHION_DF_NON_IID,
                "P=50": SPARSE50_FASHION_DF_NON_IID,
                "P=90": SPARSE90_FASHION_DF_NON_IID
            },
        "mlu":
            {
                "E=2": MLU2_FASHION_DF_NON_IID,
                "E=10": MLU10_FASHION_DF_NON_IID,
                "E=20": MLU20_FASHION_DF_NON_IID
            },
        "other":
            {
                "GQ + GS(P=50) + MU(E=10)": SPARSE50_MLU10_Q_FASHION_DF_NON_IID,
                "GQ": Q_FASHION_DF_NON_IID,
                "regular": REGULAR_FED_FASHION_DF_NON_IID,
                "Centralized": NON_FED_FASHION_DF_NON_IID
            }
    },
}

sns.set_palette("bright")

def plot_q(dataset: str, metric, cut_after: int = None, save: str = None):
    data = DATAFRAMES[dataset]["other"]
    data.pop('GQ + GS(P=50) + MU(E=10)', None)
    data.pop('GQ + GS(P=50) + MU(E=5)', None) # for colorectal
    if "GQ + GS(P=50) + MU(E=2)" in data:
        data.pop("GQ + GS(P=50) + MU(E=2)")  # for cifar
    regular = data.pop('regular', None)
    data.pop("Centralized", None)
    fig, ax = plt.subplots()
    if cut_after:
        ax.hlines(regular[metric].max(), 0, cut_after, linestyles="dashed", colors="black", linewidth=2)
    else:
        ax.hlines(regular[metric].max(), 0, regular["Iteration"].max(), linestyles="dashed", colors="black", linewidth=2)

    for datapoint in data:
        if cut_after:
            sns.lineplot(x="Iteration", y=metric, data=data[datapoint][:cut_after])
        else:
            sns.lineplot(x="Iteration", y=metric, data=data[datapoint])

    plt.xlabel("Communication round", fontsize=20)
    plt.ylabel(f"Test {metric.lower()}", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_PATH}/q_{dataset}_{save}", format="eps")
        print(f"Saving to: {OUTPUT_PATH}/q_{dataset}_{save}")
        plt.close()
    else:
        plt.show()

def plot_mlu(dataset: str, metric, cut_after: int = None, save: str = None):
    data = DATAFRAMES[dataset]["mlu"]
    regular = DATAFRAMES[dataset]["other"]["regular"]

    if cut_after:
        plt.hlines(regular[metric].max(), 0, cut_after, linestyles="dashed", colors="black", linewidth=2)
    else:
        plt.hlines(regular[metric].max(), 0, regular["Iteration"].max(), linestyles="dashed", colors="black",
                   linewidth=2)
    for datapoint in data:
        if datapoint == "regular":
            if cut_after:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint][:cut_after], color="black", linewidth=1.0)
            else:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint])
        else:
            if cut_after:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint][:cut_after])
            else:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint])

    plt.xlabel("Communication round", fontsize=20)
    plt.ylabel(f"Test {metric.lower()}", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(data, fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_PATH}/mlu_{dataset}_{save}", format="eps")
        print(f"Saving to: {OUTPUT_PATH}/mlu_{dataset}_{save}")
        plt.close()
    else:
        plt.show()


def plot_sparse(dataset: str, metric, cut_after: int = None, save: str = None, smoothen_by:int=None):
    data = DATAFRAMES[dataset]["sparse"]
    regular = DATAFRAMES[dataset]["other"].pop("regular")
    if cut_after:
        plt.hlines(regular[metric].max(), 0, cut_after, linestyles="dashed", colors="black", linewidth=2)
    else:
        plt.hlines(regular[metric].max(), 0, regular["Iteration"].max(), linestyles="dashed", colors="black",
                   linewidth=2)
    for datapoint in data:
        if cut_after:
            if smoothen_by:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint][:cut_after:smoothen_by])
            else:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint][:cut_after])
        else:
            if smoothen_by:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint][::smoothen_by])
            else:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint])

    plt.xlabel("Communication round", fontsize=20)
    plt.ylabel(f"Test {metric.lower()}", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(data, fontsize=13, loc=4)
    plt.tight_layout()

    if save:
        if smoothen_by:
            save_path = f"{OUTPUT_PATH}/sparse_{dataset}_s{smoothen_by}_{save}"
        else:
            save_path = f"{OUTPUT_PATH}/sparse_{dataset}_{save}"
        plt.savefig(save_path, format="eps")
        print(f"Saving to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_combinations(dataset: str, metric, cut_after: int = None, save: str = None):
    data = DATAFRAMES[dataset]["other"]
    data["MU(E=10)"] = DATAFRAMES[dataset]["mlu"]["E=10"]
    if dataset != "colorectal": # No P=50 in colorectal
        data["GS(P=50)"] = DATAFRAMES[dataset]["sparse"]["P=50"]
    else:
        data["GS(P=10)"] = DATAFRAMES[dataset]["sparse"]["P=10"]
        data["MU(E=5)"] = DATAFRAMES[dataset]["mlu"]["E=5"]
    regular = DATAFRAMES[dataset]["other"].pop("regular")
    DATAFRAMES[dataset]["other"].pop("Centralized")
    if cut_after:
        plt.hlines(regular[metric].max(), 0, cut_after, linestyles="dashed", colors="black", linewidth=2)
    else:
        plt.hlines(regular[metric].max(), 0, regular["Iteration"].max(), linestyles="dashed", colors="black",
                   linewidth=2)
    data["Conventional Federated Learning"] = regular

    if dataset == "colorectal":
        sns.lineplot(x="Iteration", y=metric, data=data["Conventional Federated Learning"][:cut_after:10], color="black")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ"][:cut_after:10], color="red")
        sns.lineplot(x="Iteration", y=metric, data=data["MU(E=5)"][:cut_after:10], color="blue")
        sns.lineplot(x="Iteration", y=metric, data=data["GS(P=10)"][:cut_after:10], color="green")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ + GS(P=50) + MU(E=5)"][:cut_after:10], color="orange")

        plt.legend(["Conventional Federated Learning", "GQ", "MU(E=5)", "GS(P=10)", "GQ + GS(P=50) + MU(E=5)"], fontsize=13)
    elif dataset == "fashion":
        sns.lineplot(x="Iteration", y=metric, data=data["Conventional Federated Learning"][:cut_after], color="black")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ"][:cut_after], color="red")
        sns.lineplot(x="Iteration", y=metric, data=data["MU(E=10)"][:cut_after], color="blue")
        sns.lineplot(x="Iteration", y=metric, data=data["GS(P=50)"][:cut_after], color="green")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ + GS(P=50) + MU(E=10)"][:cut_after], color="orange")

        plt.legend(["Conventional Federated Learning", "GQ", "MU(E=10)", "GS(P=50)", "GQ + GS(P=50) + MU(E=10)"], fontsize=13)
    elif dataset == "cifar10":
        sns.lineplot(x="Iteration", y=metric, data=data["Conventional Federated Learning"][:cut_after], color="black")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ"][:cut_after], color="red")
        sns.lineplot(x="Iteration", y=metric, data=data["MU(E=10)"][:cut_after], color="blue")
        sns.lineplot(x="Iteration", y=metric, data=data["GS(P=50)"][:cut_after], color="green")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ + GS(P=50) + MU(E=10)"][:cut_after], color="orange")
        sns.lineplot(x="Iteration", y=metric, data=data["GQ + GS(P=50) + MU(E=2)"][:cut_after], color="#B5A966")

        plt.legend(["Conventional Federated Learning", "GQ", "MU(E=10)", "GS(P=50)",
                    "GQ + GS(P=50) + MU(E=10)", "GQ + GS(P=50) + MU(E=2)"], fontsize=13)
    else:
        for datapoint in data:
            if cut_after:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint][:cut_after])
            else:
                sns.lineplot(x="Iteration", y=metric, data=data[datapoint])
        plt.legend(data, fontsize=13)

    plt.xlabel("Communication round", fontsize=20)
    plt.ylabel(f"Test {metric.lower()}", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_PATH}/combinations_{dataset}_{save}", format="eps")
        print(f"Saving to: {OUTPUT_PATH}/combinations_{dataset}_{save}")
        plt.close()
    else:
        plt.show()


def plot_heatmap():
    heat = np.array([
        [50, 25, 81.2],
        [75, 50, 25],
        [62.5, 62.5, 25]
    ])
    mask = np.array([
        [False, True, False],
        [False, False, True],
        [False, False, False]
    ])
    sns.heatmap(heat, linewidths=0.5, linecolor="w", annot=True, mask=mask, square=True, cmap="Blues",
                cbar_kws={'label': 'Bandwidth savings in percent'})
    plt.xticks([0.5, 1.5, 2.5], ["gradQ", "MCsU", "gradS"])
    plt.yticks([0.5, 1.5, 2.5], ["gradQ", "MCsU", "gradS"])
    for label in plt.gca().yaxis.get_majorticklabels():
        print(label.get_position()[1])
        label.set_position((0, label.get_position()[1] + 0.5))
        print(label.get_position()[1])

    plt.text(2.05, 1.1, "                   |   \ngradQ, sparseS and MCsU", horizontalalignment='left', size='x-small', color='black', weight='light')

    plt.savefig("../../storage/produced-data/plots/final_plots/data_savings_in_percent.eps", format="eps")
    #plt.show()


def create_limit_approach_table(dataset: str, limits: list):
    import math
    all_data = DATAFRAMES[dataset]
    maximum_value = all_data["other"]["regular"].max()["Accuracy"]
    maximum_value = math.floor(maximum_value * 1000) / 1000.0

    limits.append(maximum_value)

    dataframe_with_all_indices = pd.DataFrame(columns=limits)
    for category in all_data:
        for approach in all_data[category]:
            list_of_indices_of_first_appearance = []
            dataframe = all_data[category][approach]
            accuracy_series = dataframe["Accuracy"]
            for limit in limits:
                if not (accuracy_series >= limit).any():
                    list_of_indices_of_first_appearance.append(None)
                    continue
                first_value_over_limit = accuracy_series[accuracy_series >= limit].iloc[0]
                index = dataframe.loc[accuracy_series == first_value_over_limit, 'Iteration'].iloc[0]
                list_of_indices_of_first_appearance.append(index)
            dataframe_with_all_indices.loc[approach] = list_of_indices_of_first_appearance
    return dataframe_with_all_indices


def create_all_limit_approach_tables(dataset: str):
    print(f"Creating table for {dataset}")
    if dataset == "fashion":
        limits = [0.84, 0.86, 0.88, 0.90, 0.92]
    elif dataset == "cifar10":
        limits = [0.5, 0.6, 0.68, 0.7, 0.72]
    elif dataset == "colorectal":
        limits = [0.55, 0.65, 0.7, 0.72, 0.76]
    elif dataset == "colorectal_mlu":
        limits = [0.55, 0.6, 0.65, 0.7, 0.715, 0.73]
        #limits = [0.1, 0.2, 0.3, 0.4, 0.5]
        dataset = "colorectal"
    else:
        limits = []
    print(create_limit_approach_table(dataset, limits))


def calculate_bandwitdh_savage(iteration_regular, iteratio_approach, number_of_clients, model_size, p:tuple = None, E = None, q = False):
    regular_bit_traffic_till_convergence = 2 * iteration_regular * number_of_clients * model_size * 32
    total_traffic = regular_bit_traffic_till_convergence / 8 / 1000 / 1000

    if p and E and q:
        if len(p) == 2:
            combined_bit_traffic =  ((p[1] * 32) +
                                    (iteratio_approach * model_size) +
                                    iteratio_approach * number_of_clients * model_size * 32) / 2
        else:
            combined_bit_traffic = ((iteratio_approach * number_of_clients * model_size * 32 * (1 - p / 100)) +
                                    (iteratio_approach * model_size) +
                                    iteratio_approach * number_of_clients * model_size * 32) / 2
        print(f"E={E}, Q, P={p}: {combined_bit_traffic}")
        combined_percentage_savings = (1 - (combined_bit_traffic / regular_bit_traffic_till_convergence)) * 100
        print(f"Percentage savings for E={E}, Q, P={p}: {combined_percentage_savings}")
        combined_total = combined_bit_traffic / 8 / 1000 / 1000
        combined_total_savings = total_traffic - combined_total
        print("total_traffic | combined_total | combined_percentage_savings")
        print(f"{total_traffic:.2f} | {combined_total:.2f} | {combined_percentage_savings:.2f}")

    elif p:
        if len(p) == 1:
            sparse_bit_traffic = (iteratio_approach * number_of_clients * model_size * 32) + \
                                 (iteratio_approach * model_size * number_of_clients) + \
                                 iteratio_approach * number_of_clients * model_size * 32
        else:
            sparse_bit_traffic = (p[1] * 32) + \
                                 (iteratio_approach * model_size * number_of_clients) + \
                                 iteratio_approach * number_of_clients * model_size * 32
        print(f"P={p}: {sparse_bit_traffic}")
        sparse_percentage_savings = (1 - (sparse_bit_traffic / regular_bit_traffic_till_convergence)) * 100
        print(f"Percentage savings for P={p}: {sparse_percentage_savings}")
        sparse_total = sparse_bit_traffic / 8 / 1000 / 1000
        print("total_traffic | sparse_total | sparse_percentage_savings")
        print(f"{total_traffic:.2f} | {sparse_total:.2f} | {sparse_percentage_savings:.2f}")

    elif q:
        q_bit_traffic = iteratio_approach * number_of_clients * model_size * 32
        print(f"Q: {q_bit_traffic}")
        q_percentage_savings = (1 - (q_bit_traffic / regular_bit_traffic_till_convergence)) * 100
        print(f"Percentage savings for Q: {q_percentage_savings}")
        q_total = q_bit_traffic / 8 / 1000 / 1000
        q_total_savings = total_traffic - q_total
        print("total_traffic | q_total | q_percentage_savings")
        print(f"{total_traffic:.2f} | {q_total:.2f} | {q_percentage_savings:.2f}")

    elif E:
        mu_bit_traffic = 2 * iteratio_approach * number_of_clients * model_size * 32
        print(f"E={E}: {mu_bit_traffic}")
        mu_percentage_savings = (1 - (mu_bit_traffic / regular_bit_traffic_till_convergence)) * 100
        print(f"Percentage savings for E={E}: {mu_percentage_savings}")
        mu_total = mu_bit_traffic / 8 / 1000 / 1000
        mu_total_savings = total_traffic - mu_total
        print("total_traffic | mu_total | mu_percentage_savings")
        print(f"{total_traffic:.2f} | {mu_total:.2f} | {mu_percentage_savings:.2f}")
    else:
        print(f"Regular bit traffic: {regular_bit_traffic_till_convergence}")
        regular_percentage_savings = (1 - (regular_bit_traffic_till_convergence / regular_bit_traffic_till_convergence)) * 100
        print(f"Regular percentage savings: {regular_percentage_savings}")
        total_regular_traffic = regular_bit_traffic_till_convergence / 8 / 1000 / 1000
        total_regular_savings = total_traffic - total_regular_traffic
        print("total_traffic | total_regular_traffic | regular_percentage_savings")
        print(f"{total_traffic:.2f} | {total_regular_traffic:.2f} | {regular_percentage_savings:.2f}")



if __name__ == '__main__':
    #FMNIST section
    #plot_q(dataset="fashion", metric="Accuracy", cut_after=90, save="acc.eps")
    #plot_sparse(dataset="fashion", metric="Accuracy", cut_after=90, save="acc.eps")
    #plot_mlu(dataset="fashion", metric="Accuracy", cut_after=90, save="acc.eps")
    #plot_combinations(dataset="fashion", metric="Accuracy", cut_after=90, save="acc.eps")

    #CIFAR section
    #plot_q(dataset="cifar10", metric="Accuracy", cut_after=80, save="acc.eps")
    #plot_sparse(dataset="cifar10", metric="Accuracy", cut_after=70, save="acc.eps")
    #plot_mlu(dataset="cifar10", metric="Accuracy", cut_after=70, save="acc.eps")
    #plot_combinations(dataset="cifar10", metric="Accuracy", cut_after=70, save="acc.eps")

    # CCH section
    #plot_q(dataset="colorectal", metric="Accuracy", cut_after=330, save="acc.eps")
    plot_sparse(dataset="colorectal", metric="Accuracy", cut_after=400, smoothen_by=30, save="acc.eps")
    #plot_mlu(dataset="colorectal", metric="Accuracy", cut_after=340, save="acc.eps")
    #plot_combinations(dataset="colorectal", metric="Accuracy", cut_after=300) #, save="acc.eps")

    # LIMIT TABLE
    #create_all_limit_approach_tables(dataset="colorectal")
    #cnn_size = 1663370
    #tlcnn_size = 9878794
    #vgg16_size = 65087304
    #calculate_bandwitdh_savage(iteratio_approach=121, iteration_regular=108, number_of_clients=10, model_size=vgg16_size,
                               #q=True)
    #calculate_bandwitdh_savage(iteratio_approach=96, iteration_regular=108, number_of_clients=10, model_size=tlcnn_size,
    #                           p=50)


    # For Sparse
    #iteration_approach = 5
    #iteration_reg = 5
    #calculate_bandwitdh_savage(iteratio_approach=iteration_approach, iteration_regular=iteration_reg,
    #                           number_of_clients=10, model_size=cnn_size,
    #                           p=(50, SPARSE10_FASHION_DF['Grads'].values[iteration_approach-1]))

    #plot_q(dataset="colorectal", metric="Accuracy",cut_after=400, save="acc.eps")
    #plot_mlu(dataset="colorectal", metric="Accuracy", cut_after=100, save="acc.eps")
    #plot_sparse(dataset="fashion", metric="Accuracy", cut_after=100, save="acc.eps")
    #print(SPARSE90_FASHION_DF)
    #print(SPARSE90_FASHION_DF['Grads'].values[199])
    #save_all_plots_of("fashion_non_iid")

"""
    plot_reg_and_non(dataset="fashion", metric="Accuracy", save="cnn_acc.eps")
    plot_reg_and_non(dataset="fashion", metric="Loss", save="cnn_loss.eps")

    plot_mlu(dataset="fashion", metric="Accuracy", save="cnn_acc_mlu.eps")
    plot_mlu(dataset="fashion", metric="Loss", save="cnn_loss_mlu.eps")
    plot_sparse(dataset="fashion", metric="Accuracy", save="cnn_acc_sparse.eps")
    plot_sparse(dataset="fashion", metric="Loss", save="cnn_loss_sparse.eps")

    plot_mlu(dataset="cifar10", metric="Accuracy", save="tlcnn_acc_mlu.eps")
    plot_mlu(dataset="cifar10", metric="Loss", save="tlcnn_loss_mlu.eps")
    plot_sparse(dataset="cifar10", metric="Accuracy", save="tlcnn_acc_sparse.eps")
    plot_sparse(dataset="cifar10", metric="Loss", save="tlcnn_loss_sparse.eps")"""
