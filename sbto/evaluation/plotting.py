import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_columns_prefix(df, prefix: str):
    """
    Return all columns in the dataframe that represent error metrics.
    """
    return [c for c in df.columns if c.startswith(prefix)]

def get_error_columns(df):
    return get_columns_prefix(df, "err_")

def get_act_acc_columns(df):
    return get_columns_prefix(df, "act_acc")

def plot_histograms_columns(df, columns, bins=50, figsize=(6,4)):
    """
    Plot seaborn histograms for all error columns.
    One figure per error metric.

    Parameters
    ----------
    df : pd.DataFrame
    bins : int
        Number of histogram bins.
    figsize : tuple
        Size of each figure.
    """
    for col in columns:
        plt.figure(figsize=figsize)
        sns.histplot(df[col].dropna(), bins=bins, kde=False)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

def plot_histograms_columns_grid(df, columns, bins=50, cols=3, figsize=(5,4)):
    """
    Plot all histograms in a grid layout (multipanel figure).

    Parameters
    ----------
    df : pd.DataFrame
    bins : int
        Number of bins per histogram.
    cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        Size of each subplot.
    """
    n = len(columns)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize[0], rows * figsize[1]))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), bins=bins, kde=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # turn off leftover axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_error_histograms(df, bins=50, figsize=(6,4)):
    columns = get_error_columns(df)
    return plot_histograms_columns(df, columns, bins=bins, figsize=figsize)

def plot_error_histograms_grid(df, bins=50, cols=3, figsize=(5,4)):
    columns = get_error_columns(df)
    return plot_histograms_columns_grid(df, columns, bins=bins, cols=cols, figsize=figsize)

def plot_act_acc_histograms_grid(df, bins=50, cols=3, figsize=(5,4)):
    columns = get_act_acc_columns(df)
    return plot_histograms_columns_grid(df, columns, bins=bins, cols=cols, figsize=figsize)

def plot_T_vs_duration(df, figsize=(6,4)):
    # plt.figure(figsize=figsize)
    sns.relplot(data=df, x="T", y="opt_duration")
    plt.title("Optimization time (s) vs. opt timesteps T")
    plt.tight_layout()
    plt.show()

def plot_cost_vs_opt_n_it(df, figsize=(6,4)):
    # plt.figure(figsize=figsize)
    sns.relplot(data=df, x="min_cost", y="opt_n_it")
    plt.title("Cost vs. opt iterations")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from sbto.evaluation.load import load_dataset_with_errors
    import multiprocessing as mp

    DATASET_ROOT = "datasets/OmniRetarget"
    mp.set_start_method("spawn", force=True)
    df = load_dataset_with_errors(DATASET_ROOT, num_workers=60)
    
    # single-figure histograms
    # plot_error_histograms(df)
    plot_T_vs_duration(df)
    plot_cost_vs_opt_n_it(df)
    plot_histograms_columns_grid(df, ["T"])
    plot_histograms_columns_grid(df, ["opt_duration"])
    plot_histograms_columns_grid(df, ["total_sim_timesteps"])


    # grid view of all histograms
    plot_act_acc_histograms_grid(df, cols=3)
    plot_error_histograms_grid(df, cols=3)
    