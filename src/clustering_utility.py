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
    # Calcola la matrice di prossimità (distanze euclidee tra punti)
    pairwise_distances_ = pairwise_distances(standardized_df, metric="cosine")

    # Etichette dei cluster
    n = len(cluster_labels)


    # sorting by labels
    sorted_pairwisedist = pairwise_distances_[np.argsort(cluster_labels)][:, np.argsort(cluster_labels)]
    labels = cluster_labels[np.argsort(cluster_labels)]

    # keeping the distance values between 0 and 1.
    sorted_pairwisedist = sorted_pairwisedist / np.max(sorted_pairwisedist)
    sorted_similarity = 1- sorted_pairwisedist / np.max(sorted_pairwisedist)

    # Inizializza la matrice di incidenza come matrice di zeri
    incidence_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
        # Verifica se i punti i e j appartengono allo stesso cluster
            if labels[i] == labels[j]:
                incidence_matrix[i, j] = 1

   
    # pearsonr requires two array_like inputs. ravel returns a 1-D array of the inputs
    proximity_vector = sorted_similarity.ravel()
    ideal_similarity_vector = incidence_matrix.ravel()
    correlation, _ = pearsonr(proximity_vector, ideal_similarity_vector)
    print(f"Pearson correlation between the proximity matrix and the ideal similarity matrix: {correlation:.4f}")


    plt.title("Sorted proximity matrix", fontweight='bold')
    plt.xlabel("Points ordered by cluster")
    plt.ylabel("Points ordered by cluster")
    plt.imshow(sorted_similarity,cmap ='jet')
    plt.colorbar()
    plt.show()
    
    plt.close()