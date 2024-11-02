from matplotlib import pyplot as plt
import numpy as np
import seaborn as sbn
        
def stats_and_distribution(races, column_name, attr_type):
    show_stats(races, column_name)
    if attr_type == 'numerical':
        show_hist_and_box_plot(races, column_name)
    elif attr_type == 'binary':
        binary_pie_chart(races, column_name)
    else:
        show_bar_plot(races, column_name)

         
def show_stats(races, column_name):
    column = races[column_name]
    print(f"Description of attribute '{column.name}':")
    display(column.describe())
    print("\nUnique values:")
    print(column.unique())
    mv = column.isna().sum()
    nrec = column.size
    per=mv*100/nrec
    print(f"\nNull values: {mv} over {nrec} records - ({per:.2f}%)")
    print("\nTop 5 common value:" + "\n"+str(column.value_counts().head()))
    
def show_hist_and_box_plot(races, column_name):
    column = races[column_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Istogramma per races_data
    sbn.histplot(column, bins='sturges', ax=axes[0])
    axes[0].set_title(f'Histogram of {column.name} in races_data')
    axes[0].set_xlabel(column.name)
    axes[0].set_ylabel('Frequency')
    
    # Boxplot per races
    boxplot_dict = axes[1].boxplot(column[~np.isnan(column)])  # Rimuove i NaN per evitare errori
    axes[1].set_title(f'Boxplot of {column.name}')
    axes[1].set_xlabel(column.name)
    
    # Recupera gli outlier dal boxplot
    outliers = [flier.get_ydata() for flier in boxplot_dict['fliers']]
    # Ottieni la lista di outlier senza duplicati
    outliers_values = list({value for sublist in outliers for value in sublist})

    plt.tight_layout()  # Per evitare sovrapposizioni
    plt.show()

    # Stampa gli outlier
    print(f"Outliers in {column.name}: {outliers_values}")
    
def show_bar_plot(races, column_name):
    plt.figure(figsize=(14, 10))
    sbn.countplot(data=races, x=column_name)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

def binary_pie_chart(races, binary_feature):
    # Calcola le frequenze di True e False
    counts = races[binary_feature].value_counts()  
    # Imposta le etichette in base ai valori disponibili
    labels = []
    if True in counts.index:
        labels.append('True')
    if False in counts.index:
        labels.append('False')
    # Crea una nuova figura
    plt.figure(figsize=(6, 6))  # Imposta la dimensione della figura
    # Crea il pie chart
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    # Aggiungi il titolo
    plt.title(f'Piechart of {binary_feature}')
    plt.show()