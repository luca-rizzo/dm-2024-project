from matplotlib import pyplot as plt
import numpy as np

def stats(column, box=True):
    print(f"Description of attribute '{column.name}':")
    display(column.describe())
    print("\nUnique values:")
    print(column.unique())
    print(f"\nNumber of null values: {column.isnull().sum()}")
    print("\nTop 5 common value:" + "\n"+str(column.value_counts().head()))

    if box:
        boxplot_dict = plt.boxplot(column[~np.isnan(column)])
        # Recover Outliers
        outliers = [flier.get_ydata() for flier in boxplot_dict['fliers']]
        # P