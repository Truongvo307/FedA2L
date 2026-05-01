import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


def plot_accuracy_granularity(data, save_path="heatmap.png", figsize=(14, 8)):
    data = {row[0]: list(row[1:]) for row in data.rows()}

    # Create a Polars DataFrame from the dictionary
    df = pl.DataFrame(data)

    # Convert the Polars DataFrame to a long format for heatmap compatibility
    df_long = df.melt(
        id_vars=["accuracy"], variable_name="nodes_and_servers", value_name="value"
    )

    # Convert back to a wide format for seaborn heatmap compatibility
    df_pivot = df_long.pivot(
        index="nodes_and_servers", columns="accuracy", values="value"
    )

    # Convert to a format that seaborn can use directly
    df_pivot_pd = df_pivot.to_pandas()

    # Set the index to be the 'nodes_and_servers' column
    df_pivot_pd.set_index("nodes_and_servers", inplace=True)

    # Create the heatmap
    plt.figure(figsize=figsize)

    # Set the plot background color
    plt.rcParams["axes.facecolor"] = "black"
    plt.rcParams["savefig.facecolor"] = "black"
    plt.rcParams["figure.facecolor"] = "black"

    # Set the font properties
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Create the heatmap with appropriate settings for a black background
    ax = sns.heatmap(
        df_pivot_pd,
        annot=True,
        cmap="viridis",
        cbar=True,
        fmt=".0f",
        annot_kws={"size": 8, "color": "white"},
        linewidths=0.5,
        linecolor="black",
    )

    plt.title("Heatmap of Epochs to Reach Different Accuracy Levels", color="white")
    plt.xlabel("Accuracy (%)", color="white")
    plt.ylabel("nodes and Servers", color="white")

    # Set tick labels color
    plt.xticks(color="white")
    plt.yticks(color="white")

    # Change color bar (legend) label to white
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Save the plot as a PNG file with 300 DPI
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_participant_rate(data, figsize=(10, 5), save_path="participant.png"):
    # Extract keys and values
    client_ids = list(data.keys())
    iterations = list(data.values())

    # Create the plot
    plt.figure(figsize=figsize)

    # Set the background color
    plt.rcParams["axes.facecolor"] = "black"
    plt.rcParams["savefig.facecolor"] = "black"

    # Plot the data
    plt.bar(client_ids, iterations, color="skyblue")

    # Add titles and labels with larger font size and white color
    plt.title(
        "Number of Iterations Each Client Participates In", fontsize=14, color="white"
    )
    plt.xlabel("Client ID", fontsize=12, color="white")
    plt.ylabel("Number of Iterations", fontsize=12, color="white")

    # Set x-ticks and y-ticks to white color
    plt.xticks(client_ids, fontsize=10, color="white")
    plt.yticks(fontsize=10, color="white")

    # Remove border (spines)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Save the plot with higher DPI
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_data_distribution(
    file_path: str,
    data_type: str = "test",
    output_path: str = "",
    num_classes: int = 10,
    normalize: bool = False,
    min_size: int = 30,
    max_size: int = 1500,
    ylabel_step: int = 1,
):

    output_path = output_path or file_path
    data = pl.read_csv(os.path.join(file_path, f"{data_type}_statistics.csv")).sort(
        "client_id"
    )
    label_sum = data.groupby("label").agg(pl.col("count").sum())
    labels, counts = label_sum["label"].to_list(), label_sum["count"].to_list()
    grouped_data = [
        list(zip(group["label"], group["count"]))
        for _, group in data.groupby("client_id")
    ]

    if normalize:
        all_sizes = np.array(
            [sample for client in grouped_data for _, sample in client]
        )
        normalized_sizes = (all_sizes - all_sizes.min()) / (
            all_sizes.max() - all_sizes.min()
        )
        normalized_sizes = normalized_sizes * (max_size - min_size) + min_size
    else:
        normalized_sizes = None

    plt.figure(figsize=(10, 6))
    bubble_idx = 0
    for client_id, client_samples in enumerate(grouped_data):
        for label, sample_size in client_samples:
            size = normalized_sizes[bubble_idx] if normalize else sample_size
            bubble_idx += 1 if normalize else 0
            plt.scatter(client_id, label, s=size, c="blue", alpha=0.5)

    plt.xlabel("Client IDs")
    plt.ylabel("Class IDs")
    plt.xticks(range(len(grouped_data)))
    plt.yticks(range(0, num_classes, ylabel_step))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{data_type}_distribution.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color="skyblue", alpha=0.8)
    plt.xlabel("Class IDs")
    plt.ylabel("Number of Samples")
    plt.title(f"{data_type.capitalize()} Distribution (Total={sum(counts)} samples)")
    plt.xticks(labels)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"{data_type}_distribution_total_classes.png"),
        dpi=300,
    )


def get_granularity_indices(accuracies, granularity=5):
    """
    Returns the indices (epochs) where the accuracies reach each granularity level.
    """
    granularity_levels = list(range(0, 101, granularity))
    granularity_indices = []

    for level in granularity_levels:
        for idx, acc in enumerate(accuracies):
            if acc * 100 >= level:
                granularity_indices.append(idx)
                break
        else:
            granularity_indices.append(None)

    return granularity_indices


def process_heatmap(file_path, case="Fully_connected", t=0, granularity=5):
    path_files = os.path.join(file_path, case, str(t), "results")
    path_coordinator = os.path.join(path_files, "coordinator.csv")
    data = pl.read_csv(path_coordinator).to_dict(as_series=False)
    save_path = os.path.join(path_files, f"{case}_heatmap_{t}.png")
    granularity_df = {
        "coordinator": get_granularity_indices(data["mean_acc"], granularity)
    }
    for file in os.listdir(path_files):
        if "coordinator" in file:
            continue
        client_data = pl.read_csv(os.path.join(path_files, file)).to_dict(
            as_series=False
        )
        client_id = file.split("_")[1].split(".")[0]
        granularity_df[f"client_{client_id}"] = get_granularity_indices(
            client_data["accs"], granularity
        )

    # Save granularity dataframe to CSV
    granularity_df = pl.DataFrame(granularity_df).transpose(
        include_header=True,
        column_names=[str(i) for i in range(0, 101, granularity)],
        header_name="accuracy",
    )
    df = granularity_df.to_pandas()
    df_long = pd.melt(
        df, id_vars=["accuracy"], var_name="thresholds", value_name="epochs"
    )
    df_pivot = df_long.pivot(index="accuracy", columns="thresholds", values="epochs")
    df_pivot = df_pivot[
        sorted(df_pivot.columns, key=lambda x: int(x) if x.isdigit() else x)
    ]

    plt.figure(figsize=(14, 8))
    sns.heatmap(df_pivot, annot=True, fmt=".0f", cmap="viridis", cbar=True)
    plt.title(f"Heatmap of Rounds to Reach Accuracy Thresholds | {case}")
    plt.xlabel("Accuracy Thresholds (%)")
    plt.ylabel("nodes and Coordinator")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # data = pl.read_csv("data.csv").to_dict()
    # print(data)
    # case = 'Fully_connected'
    # t = 0
    # path_files = f'/home/truong/Truong/DFL/runs/'
    # process_heatmap(path_files,case='CenFL',t=3)
    # plot_accuracy_granularity(data)
    pass
