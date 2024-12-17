import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# All auxiliary code is in ../src
import sys
sys.path.append("../src/")

def custom_std(x):
    return x.std() if len(x) > 1 else 0

def data_pre_processing():
    dataset, cyclists, placements = merge_all_dataset()

    new_stats = create_cyclist_stats_2021(cyclists, placements)

    dataset = drop_previous_and_merge_newstats(dataset, new_stats)

    dataset.drop(['delta', 'median_delta', 'std_delta'], axis=1, inplace=True)
    drop_redundant_columns(dataset)

    dataset['top20'] = (dataset['position'] < 20).astype(int)
    dataset.drop(['position'], axis=1, inplace=True)
    
    return dataset


def drop_redundant_columns(dataset):
    dataset.rename(columns={'year': 'race_year'}, inplace=True)
    dataset.drop(['birth_year'], axis=1, inplace=True)  # birth_year is redundant w.r.t race_year
    dataset.drop(['median position_2021'], axis=1, inplace=True)
    # climb/lenght is redundant
    # cyclist and url are identifiers
    dataset.drop(['climb/length', 'cyclist', '_url'], axis=1, inplace=True)
    # column rendundant w.r.t geo area
    dataset.drop(['nationality', 'continent'], axis=1, inplace=True)


def drop_previous_and_merge_newstats(dataset, new_stats):
    dataset.drop(['AVG position', 'median position', '# cyclist races', 'std_dev position'], axis=1, inplace=True)
    dataset['cyclist'] = dataset['cyclist'].astype(str)
    new_stats['cyclist'] = new_stats['cyclist'].astype(str)
    dataset = pd.merge(dataset, new_stats, how='left', on='cyclist')
    return dataset


def merge_all_dataset():
    races = pd.read_csv('../dataset/preprocessedRaces_without_outliers.csv', sep=",")
    placements = pd.read_csv('../dataset/preprocessedPlacements.csv', sep=",")
    cyclists = pd.read_csv('../dataset/preprocessedCyclists_without_outliers.csv', sep=",")
    cyclists.drop(['AVG weighted position', 'AVG position Autumn', 'AVG position Spring', 'AVG position Summer',
                   'AVG position Winter', 'AVG position Q1', 'AVG position Q2', 'AVG position Q3', 'AVG position Q4'],
                  axis=1, inplace=True)
    races.drop(['uci_points', 'average_temperature', 'is_tarmac', 'is_cobbled', 'is_gravel'], axis=1, inplace=True)
    placements.drop(['cyclist_team'], axis=1, inplace=True)
    dataset = pd.merge(placements, races, how='inner', on='_url')
    dataset.drop(['date_y', 'date_x'], axis=1, inplace=True)
    dataset = pd.merge(dataset, cyclists, how='inner', left_on='cyclist', right_on='name')
    dataset.drop(['name_x', 'name_y'], axis=1, inplace=True)
    return dataset, cyclists, placements


def create_cyclist_stats_2021(cyclists, placements):
    new_stats = pd.merge(placements, cyclists, left_on='cyclist', right_on='name', how='inner')
    new_stats = new_stats[['cyclist', 'position', 'year']]
    new_stats = new_stats[new_stats['year'] < 2022]
    new_stats['median position'] = new_stats.groupby('cyclist')['position'].transform(lambda x: round(x.median()))
    new_stats['AVG position'] = new_stats.groupby('cyclist')['position'].transform(lambda x: round(x.mean()))
    new_stats['# cyclist races'] = new_stats.groupby('cyclist')['cyclist'].transform("count")
    # We had to define this custom function because the default one did not work.
    new_stats['std_dev position'] = new_stats.groupby('cyclist')['position'].transform(custom_std)
    new_stats.drop(['position', 'year'], axis=1, inplace=True)
    new_stats.drop_duplicates(inplace=True)
    new_stats = new_stats.rename(columns={'median position': 'median position_2021',
                                          'AVG position': 'AVG position_2021',
                                          '# cyclist races': '# cyclist races_2021',
                                          'std_dev position': 'std_dev position_2021'
                                          })
    return new_stats


#input: the dataset and the list of variables' names to encode
def categorical_columns_encoding(dataset, variables):
    for variable in variables:
        #get the unique variable's values
        var = sorted(dataset[variable].unique())
        
        #generate a mapping from the variable's values to the number representation  
        mapping = dict(zip(var, range(0, len(var) + 1)))

        print(f"Mapping of {variable}:", mapping)
        #add a new colum with the number representation of the variable
        dataset[variable+'_num'] = dataset[variable].map(mapping).astype(int)
    return dataset

def data_for_non_distance_method():
    dataset = data_pre_processing()
    categorical_columns = ['geo area']
    
    dataset = categorical_columns_encoding(dataset, categorical_columns)
    
    dataset.drop(categorical_columns, axis=1, inplace=True) #column already encoded
    return dataset

def data_for_distance_method():
    dataset = data_pre_processing()
    dataset_copy = dataset.copy()
    dataset.drop(['profile'], axis=1, inplace=True)
    
    scaler = StandardScaler()
    columns_to_be_scaled = ['cyclist_age', 'race_year', 'points', 'length', 'climb_total',
           'startlist_quality', 'BMI', 'AVG position_2021',
           '# cyclist races_2021', 'std_dev position_2021']
    top20 = dataset.pop('top20')
    geo_area = dataset.pop('geo area')
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=columns_to_be_scaled)
    dataset = pd.concat([dataset.reset_index(drop=True), top20.reset_index(drop=True)], axis=1)
    dataset = pd.concat([dataset.reset_index(drop=True), geo_area.reset_index(drop=True)], axis=1)
    dataset = pd.get_dummies(dataset, columns = ['geo area'], prefix_sep='_')
    
    return dataset, dataset_copy

def train_test_for_distance():
    dataset_std, dataset = data_for_distance_method()
    mean = dataset['race_year'].describe()['mean']
    std = dataset['race_year'].describe()['std']
    threshold_for_standardized = (2022-mean)/std
    test_set=dataset_std[dataset_std['race_year']>=threshold_for_standardized]
    train_set=dataset_std[dataset_std['race_year']<threshold_for_standardized]
    train_label=train_set.pop('top20') 
    test_label=test_set.pop('top20')
    return train_set, test_set, train_label, test_label

def train_test_for_non_distance():
    dataset = data_for_non_distance_method()
    test_set=dataset[dataset['race_year']>=2022]
    train_set=dataset[dataset['race_year']<2022]
    train_label=train_set.pop('top20') 
    test_label=test_set.pop('top20')
    return train_set, test_set, train_label, test_label

def split_in_train_and_validation(train_set, train_label):
    return train_test_split(train_set, train_label, stratify = train_label, test_size=0.30, random_state=RANDOM_STATE)
