import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr

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


# it requires a dataframe which is already standardized
def compute_similarity_matrix(standardized_df, cluster_labels):
    # Discard noisy points
    valid_indices = np.where(cluster_labels != -1)[0]
    filtered_df = standardized_df.iloc[valid_indices]  # Dataset filtering
    filtered_labels = cluster_labels[valid_indices]   # Label filtering

    # Determine the pairwise distance matrix (using the Euclidean distance)
    pairwise_distances_ = pairwise_distances(filtered_df, metric="euclidean")

    # Cluster labels
    n = len(filtered_labels)

    # Sorting by labels
    sorted_pairwisedist = pairwise_distances_[np.argsort(filtered_labels)][:, np.argsort(filtered_labels)]
    labels = cluster_labels[np.argsort(cluster_labels)]

    # Keeping the distance values between 0 and 1.
    sorted_pairwisedist = sorted_pairwisedist / np.max(sorted_pairwisedist)
    sorted_similarity = 1- sorted_pairwisedist / np.max(sorted_pairwisedist)

    ideal_similarity_matrix = np.zeros((n, n), dtype=int) #fill all positions with zeros

    for i in range(n):
        for j in range(n):
        #matrix[i, j]==1 if point i and point j belong to the same cluster
            if labels[i] == labels[j]:
                ideal_similarity_matrix[i, j] = 1


    # pearsonr requires two array_like inputs. ravel returns a 1-D array of the inputs
    proximity_vector = sorted_similarity.ravel()
    ideal_similarity_vector = ideal_similarity_matrix.ravel()
    correlation, _ = pearsonr(proximity_vector, ideal_similarity_vector)
    print(f"Pearson correlation between the proximity matrix and the ideal similarity matrix: {correlation:.4f}")

    plt.title("Sorted proximity matrix", fontweight='bold')
    plt.xlabel("Points ordered by cluster")
    plt.ylabel("Points ordered by cluster")
    plt.imshow(sorted_similarity,cmap ='jet')
    plt.colorbar()
    plt.show()
    
    plt.close()


    