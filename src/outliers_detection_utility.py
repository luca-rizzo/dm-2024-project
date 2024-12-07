def analyze_outliers(dataframe, column, lower_bound, upper_bound, num_ranges_up, num_ranges_down, range_increment_up, range_increment_down):
    # Identifica gli outliers
    outliers = dataframe[
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]
    
    outliers_values = outliers[column].round(2)

    outliers_above = outliers[outliers[column] > upper_bound]
    outliers_below = outliers[outliers[column] < lower_bound]

    print(f"Outliers above upper bound ({upper_bound}): {len(outliers_above)}")
    print(f"Outliers below lower bound ({lower_bound}): {len(outliers_below)}")

    total_outliers = len(outliers_values)
    print(f"Total outliers: {total_outliers}")

    # Intervalli sopra l'upper bound (se ci sono outliers sopra)
    if not outliers_above.empty:
        upper_ranges = [upper_bound + i * range_increment_up for i in range(num_ranges_up + 1)]

        upper_ranges_outliers_counts = []
        for i in range(len(upper_ranges) - 1):
            range_min = upper_ranges[i]
            range_max = upper_ranges[i + 1]
            count = len(outliers_values[
                (outliers_values >= range_min) &
                (outliers_values < range_max)
            ])
            upper_ranges_outliers_counts.append((range_min, range_max, count))
            print(f"Number of outliers in the range of [{range_min}, {range_max}]: {count}")
        
        outliers_above_last_range = len(outliers_values[outliers_values >= upper_ranges[-1]])
        print(f"Number of outliers > {upper_ranges[-1]}: {outliers_above_last_range}\n")
    else:
        print("No outliers above the upper bound.")
        upper_ranges_outliers_counts = []
        outliers_above_last_range = 0

    # Intervalli sotto il lower bound (se ci sono outliers sotto)
    if not outliers_below.empty:
        lower_ranges = [lower_bound - i * range_increment_down for i in range(num_ranges_down + 1)]

        lower_ranges_outliers_counts = []
        for i in range(len(lower_ranges) - 1):
            range_min = lower_ranges[i + 1]  # Range più piccolo
            range_max = lower_ranges[i]      # Range più grande
            count = len(outliers_values[
                (outliers_values < range_max) &
                (outliers_values >= range_min)
            ])
            lower_ranges_outliers_counts.append((range_min, range_max, count))
            print(f"Number of outliers in the range of [{range_min}, {range_max}]: {count}")
        
        outliers_below_last_range = len(outliers_values[outliers_values < lower_ranges[-1]])
        print(f"Number of outliers < {lower_ranges[-1]}: {outliers_below_last_range}")
    else:
        print("No outliers below the lower bound.")
        lower_ranges_outliers_counts = []
        outliers_below_last_range = 0

    # return {
    #     "total_outliers": total_outliers,
    #     "upper_ranges_counts": upper_ranges_outliers_counts,
    #     "outliers_above_last_range": outliers_above_last_range,
    #     "lower_ranges_counts": lower_ranges_outliers_counts,
    #     "outliers_below_last_range": outliers_below_last_range
    # }