import matplotlib.pyplot as plt
import seaborn as sbn


def clusters_conditional_box_plot(df, cluster_label):
    columns = [col for col in df.columns if col != cluster_label]
    n_rows = (len(columns) + 2 - 1) // 2  

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), sharey=False)

    # Rendi axes un array 1D per un accesso più semplice
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sbn.boxplot(x = cluster_label, y=column, data=df, ax=axes[i])
        axes[i].set_title(f"Distribution of '{column}'", fontweight='bold')
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel(column)

    # Nascondi subplot extra (se presenti)
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()