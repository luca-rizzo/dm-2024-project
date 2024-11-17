from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
        
def stats_and_distribution(races, column_name, attr_type):
    show_stats(races, column_name, attr_type)
    if attr_type == 'numerical': 
        show_hist_and_box_plot(races, column_name) #for numeric features we generate the histogram and the boxplot
    elif attr_type == 'binary': 
        binary_pie_chart(races, column_name) #for binary features we generate a piechart
    else:
        show_bar_plot(races, column_name) #for categorical features we generate a barplot

         
def show_stats(races, column_name, attr_type = 'numerical'):
    column = races[column_name].astype('object') if attr_type == 'categorical' else races[column_name]
    #Print some basic statistics like the number of values, mean, std deviation. min/max, quartiles
    print(f"Description of attribute '{column.name}':")
    display(column.describe())

    #Print the unique values
    print("\nUnique values:")
    print(column.unique())

    #Missing values counting
    mv = column.isna().sum()
    nrec = column.size
    per=mv*100/nrec
    print(f"\nNull values: {mv} over {nrec} records - ({per:.2f}%)")

    #Top 5 frequent values
    print("\nTop 5 most frequent values:" + "\n"+str(column.value_counts().head()))
    
def show_hist_and_box_plot(dataframe, column_name):
    column = dataframe[column_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram 
    sbn.histplot(column, bins='sturges', ax=axes[0])
    axes[0].set_title(f'Histogram of {column.name}', fontweight='bold')
    axes[0].set_xlabel(column.name) 
    axes[0].set_ylabel('Frequency')
    
    # Boxplot
    boxplot_dict = axes[1].boxplot(column[~np.isnan(column)])  # Removes the NaNs to avoid problems
    axes[1].set_title(f'Boxplot of {column.name}', fontweight='bold')
    axes[1].set_xlabel(column.name)
    
    # Extraction of the outliers from the boxplot and printing of them
    outliers = [flier.get_ydata() for flier in boxplot_dict['fliers']]
    outliers_values = list({value for sublist in outliers for value in sublist})

    plt.tight_layout()  # Avoid overlapping
    plt.show()

    print(f"Outliers in {column.name}: {outliers_values}")
    
def show_bar_plot(dataframe, column_name):
    values = dataframe[column_name].dropna()
    values_count = values.value_counts()
    plt.figure(figsize=(14, 10))
    sbn.barplot(x= values_count.index, y= values_count.values)
    plt.title(f'Bar plot of {column_name}', fontweight='bold')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()

def binary_pie_chart(dataframe, binary_feature):
    # Determine the True/False frequencies
    counts = dataframe[binary_feature].value_counts()  
    # Imposta le etichette in base ai valori disponibili
    labels = []
    if True in counts.index:
        labels.append('True')
    if False in counts.index:
        labels.append('False')
    plt.figure(figsize=(6, 6))
    # Pie chart creation
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Piechart of {binary_feature}', fontweight='bold')
    plt.show()
    
def calculate_outlier_bounds(dataframe, column_name):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return {"lower": lower_bound, "upper": upper_bound}

def stats_on_missing_values(dataset):
    nrec = dataset.shape[0]
    results = {
        'Feature': [],
        'Null_counter': [],
        'Perc_of_null_(%)': []
    }

    for col in dataset.columns:
        nanCount = dataset[col].isna().sum()
        perc = round((nanCount*100)/nrec,1)
        results['Feature'].append(col)
        results['Null_counter'].append(nanCount)
        results['Perc_of_null_(%)'].append(perc)

    df_results = pd.DataFrame(results)
    display(df_results)