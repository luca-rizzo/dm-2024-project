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
    min_ = dataframe[column_name].min()
    max_ = dataframe[column_name].max()
    lower_bound = max(min_, Q1 - 1.5 * IQR)
    upper_bound = min(max_, Q3 + 1.5 * IQR)
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

def show_heatmap(dataset: pd.DataFrame, columns, correlation_method):
    working_dataset = dataset[columns]
    correlation_matrix = correlations(working_dataset)
    only_selected_type = correlation_matrix[correlation_matrix['correlation_type'] == correlation_method]
    g = sbn.FacetGrid(only_selected_type, col="correlation_type", col_wrap=1, height=9, aspect=5)
    g.map_dataframe(lambda data, color: sbn.heatmap(
        data[data.columns[:-1]], #Ignore the last column of the matrix which contains the correlation type
        cmap="coolwarm", #Colormap 
        annot=True, #Correlation values inside the cell
        square=True, #Square shape for each cell of the matrix
        cbar=True, #Color bar next to the matrix to "visually" interpret the correlation
        linewidths=1,  #Lines between cells
        annot_kws={"size": 8}  #Font size
    ))
    plt.show()
    plt.close()

def correlations(dataset: pd.DataFrame) -> pd.DataFrame:
    correlations_dictionary = {
        correlation_type: dataset.corr(numeric_only=True, method=correlation_type)
        for correlation_type in ("kendall", "pearson", "spearman")
    }
    for i, k in enumerate(correlations_dictionary.keys()):
        correlations_dictionary[k].loc[:, "correlation_type"] = k
    correlations_matrix = pd.concat(correlations_dictionary.values())

    return correlations_matrix



def check_all_outliers(df, columns):

    # dataframe to keep track of outliers for each row
    outlier_flags = pd.DataFrame(index=df.index)

    for column in columns:
        bounds = calculate_outlier_bounds(df, column)
        lower_bound = bounds['lower']
        upper_bound = bounds['upper']
        
        # set rows with outliers for the current column
        outlier_flags[column] = (df[column] < lower_bound) | (df[column] > upper_bound)

    # count the number of column with outliers for each row
    outlier_flags['outlier_count'] = outlier_flags.sum(axis=1)

    # compute the percentage of columns with outliers for each row
    outlier_flags['outlier_percentage'] = (outlier_flags['outlier_count'] / len(columns)) * 100

    # rows with at least one outlier
    rows_with_outliers = outlier_flags[outlier_flags['outlier_count'] > 0]

    display(rows_with_outliers)

    print(f"Total number of rows: {len(df)}")
    print(f"Total rows with outliers: {len(rows_with_outliers)}")
    print(f"Percentage of rows with outliers: {(len(rows_with_outliers) / len(df)) * 100:.2f}%")

    return rows_with_outliers


def compare_distributions(dataframe1, dataframe2, column_name):
    column1 = dataframe1[column_name]
    column2 = dataframe2[column_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
   
    # Histogram1 
    sbn.histplot(column1, bins='sturges', ax=axes[0])
    axes[0].set_title(f'Histogram before outliers cleaning', fontweight='bold')
    axes[0].set_xlabel(column1.name) 
    axes[0].set_ylabel('Frequency')

    #Histogram2
    sbn.histplot(column2, bins='sturges', ax=axes[1])
    axes[1].set_title(f'Histogram after outliers cleaning', fontweight='bold')
    axes[1].set_xlabel(column2.name) 
    axes[1].set_ylabel('Frequency')


    plt.tight_layout() 
    plt.show()


def compare_distributions_consistent(dataframe1, dataframe2, column_name):
    column1 = dataframe1[column_name]
    column2 = dataframe2[column_name]
    
    # Combine the two datasets and calculate shared bin edges
    combined_data = np.concatenate([column1, column2])
    bins = np.histogram_bin_edges(combined_data, bins='sturges')  # Using the same binning method for consistency
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram 1
    sbn.histplot(column1, bins=bins, ax=axes[0])
    axes[0].set_title('Histogram before outliers cleaning', fontweight='bold')
    axes[0].set_xlabel(column1.name)
    axes[0].set_ylabel('Frequency')

    # Histogram 2
    sbn.histplot(column2, bins=bins, ax=axes[1])
    axes[1].set_title('Histogram after outliers cleaning', fontweight='bold')
    axes[1].set_xlabel(column2.name)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def compare_box_plots(dataframe1, dataframe2, column_name):
    column1 = dataframe1[column_name]
    column2 = dataframe2[column_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot1
    boxplot_dict = axes[0].boxplot(column1[~np.isnan(column1)])  # Removes the NaNs to avoid problems
    axes[0].set_title(f'Boxplot before outliers cleaning', fontweight='bold')
    axes[0].set_xlabel(column1.name)

    # Boxplot2
    boxplot_dict = axes[1].boxplot(column2[~np.isnan(column2)])  # Removes the NaNs to avoid problems
    axes[1].set_title(f'Boxplot after outliers cleaning', fontweight='bold')
    axes[1].set_xlabel(column2.name)


    plt.tight_layout() 
    plt.show()
