import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
import seaborn as sbn
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import json
import os
import json


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

    dataset = impute_null_stats_2021(dataset)

    dataset.drop(['delta', 'median_delta', 'std_delta'], axis=1, inplace=True)
    drop_redundant_columns(dataset)

    dataset['top20'] = (dataset['position'] < 20).astype(int)
    dataset.drop(['position'], axis=1, inplace=True)

    return dataset


def impute_null_stats_2021(dataset):
    to_work = dataset.copy()
    # Colonne su cui riempire i valori NaN
    columns_to_fill = ['median position_2021', 'AVG position_2021', '# cyclist races_2021', 'std_dev position_2021']
    for col in columns_to_fill:
        to_work[col] = to_work[col].fillna(to_work[col].mean())
    return to_work


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
    races = pd.read_csv('../../dataset/preprocessedRaces_without_outliers.csv', sep=",")
    placements = pd.read_csv('../../dataset/preprocessedPlacements.csv', sep=",")
    cyclists = pd.read_csv('../../dataset/preprocessedCyclists_without_outliers.csv', sep=",")
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


def perform_best_under_sampling(train_set, train_label):
    rand_undersampler = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy=0.67)
    train_set_us, train_label_us = rand_undersampler.fit_resample(train_set, train_label)
    return train_set_us, train_label_us

def perform_smote_over_sampling(train_set, train_label):
    smote_oversampler = SMOTE(k_neighbors=5, sampling_strategy=0.67, random_state=RANDOM_STATE)
    train_set_smote, train_labels_smote = smote_oversampler.fit_resample(train_set, train_label)
    return train_set_smote, train_labels_smote

def split_in_train_and_validation(train_set, train_label):
    return train_test_split(train_set, train_label, stratify = train_label, test_size=0.30, random_state=RANDOM_STATE)


def get_label_percentages(labels, target_names = ['not top20', 'top_20']):
    label_counts = pd.Series(labels).value_counts()

    label_percentages = (label_counts / len(labels)) * 100

    return pd.DataFrame({
        'Label': target_names,
        'Percentage': label_percentages.values
    }).sort_values(by='Percentage', ascending=False)

def plot_confusion_matrix(classifier, prior_labels, predicted_labels):
    cm = confusion_matrix(prior_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.title("Confusion Matrix", fontweight='bold')
    plt.show()
    plt.close()
    return

def compare_roc_curves(curves, title="ROC Curves"):
    # Determine the curve with the maximum AUC for the ROC curve
    # manages tuples in the form ((fpr, tpr, auc_roc, pr, rec, auc_pr), label)
    best_curve = max(curves, key=lambda x: x[0][2]) 
    best_auc = best_curve[0][2] # x[0][5] is the AUC for the ROC
   
    # Plot all curves, highlighting the best one
    plt.figure(0).clf()
    for curve in curves:
        fpr, tpr, auc_plot, label = curve[0][0], curve[0][1], curve[0][2], curve[1]
        if auc_plot == best_auc:
            plt.plot(fpr, tpr, label=f"{label}, roc_auc={auc_plot:.4f}", linewidth=2.5, color='red')  # Highlight the "best" one
        else:
            plt.plot(fpr, tpr, label=f"{label}, roc_auc={auc_plot:.4f}")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontweight='bold')
    plt.legend(loc=0)
    plt.show()
    plt.close()
    return

def compare_pr_curves(curves, title="Precision-Recall Curves"):
    # Determine the curve with the maximum AUC for Precision-Recall curve (index 5 for auc_pr)
    # manages tuples in the form ((fpr, tpr, auc_roc, pr, rec, auc_pr), label)
    best_curve = max(curves, key=lambda x: x[0][5])  # x[0][5] is the AUC for PR (precision-recall)
    best_auc_pr = best_curve[0][5]
    
    # Plot all curves, highlighting the best one
    plt.figure(0).clf() 
    for curve in curves:
        precision, recall, auc_pr, label = curve[0][3], curve[0][4], curve[0][5], curve[1]
        if auc_pr == best_auc_pr:
            plt.plot(recall, precision, label=f"{label}, PR_auc={auc_pr:.4f}", linewidth=2.5, color='red') # Highlight the best curve
        else:
            plt.plot(recall, precision, label=f"{label}, PR_auc={auc_pr:.4f}") 

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title, fontweight='bold')
    plt.legend(loc=0)
    plt.show()
    plt.close()
    return

def plot_PR_ROC_AUC(classifier, input_data, predicted_labels):
    # Check if the model supports predict_proba
    if isinstance(classifier, tf.keras.Model):
        y_prob = classifier.predict(input_data).ravel()
    elif hasattr(classifier, "predict_proba"):  # For scikit-learn models
        y_prob = classifier.predict_proba(input_data)[:, 1]
    else:
        print("This model does not support probability predictions.")
        return None
    
    precision, recall, _ = precision_recall_curve(predicted_labels, y_prob)
    auc_pr = auc(recall, precision)  # AUC of the precision-recall curve
    print(f"AUC-PR of the classifier: {auc_pr:.4f}")

    fpr, tpr, _ = roc_curve(predicted_labels, y_prob)
    roc_auc = auc(fpr, tpr)  # AUC of the ROC curve
    print(f"AUC-ROC of the classifier: {roc_auc:.4f}")

    # Set up the figure with two subplots (side by side)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall Curve
    ax[0].plot(recall, precision, color='b', linewidth=2, label=f'AUC-PR = {auc_pr:.4f}')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title("Precision-Recall Curve", fontweight='bold')
    ax[0].legend(loc="best")
    ax[0].grid(alpha=0.3)

    # ROC Curve
    ax[1].plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # Diagonal line that represents the random classifier
    ax[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC Curve', fontweight='bold')
    ax[1].legend(loc='best')
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    # Returns the information required for the final plotting of the ROC curves of all the classifiers
    return fpr, tpr, roc_auc, precision, recall, auc_pr

def plot_distribution_and_normal(column, name):
    plt.figure(figsize=(6, 4))
    sbn.histplot(column, kde=True, stat="density", bins=30, label="Histogram")
    #'density' normalizes the graph area to compare it with a normal distrib.
    mean = column.mean()
    std = column.std()
    x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
    plt.plot(x, (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2), 
             color="red", label="Normal Distribution")
    plt.title(f"Histogram of {name} overlaid with normal distribution")
    plt.legend(loc='best')
    plt.show()

def build_model_with_params(model, params):
    model.set_params(**params)
    return model

def plot_confusion_matrix_NN(prior_labels, predicted_labels, class_names):
    cm = confusion_matrix(prior_labels, predicted_labels)
    
    # Crea il display della matrice di confusione
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontweight='bold')
    plt.show()
    return


# input:
#     models -> list of model objects
#     predictions -> list of predicted labels
#     true_labels -> the true labels extracted from the test (or validation) set
# output:
#     returns the best model object based on this metric
#
# The comparison is based on the F1-Score Macro AVG metric
# The code also displays the comparison results
def compare_f1_score(models:list, predictions:list, true_labels):
    macro_f1_scores = []
   
    for y_pred in predictions:
        score = f1_score(true_labels, y_pred, average='macro')
        macro_f1_scores.append(score)
        

    model_names = [model.name if hasattr(model, 'name') else f"Model {i+1}" for i, model in enumerate(models)]

    results = pd.DataFrame({'Model': model_names, 'Macro F1-Score': macro_f1_scores, 'Model Object': models})

   
    results = results.sort_values(by='Macro F1-Score', ascending=False)
    display(results[['Model', 'Macro F1-Score']])
   
    best_model = results.iloc[0]['Model Object']  
    return best_model


def save_classification_res(method_name, params, test_pred_labels, probability_class_res):
    fpr, tpr, roc_auc, precision, recall, auc_pr = probability_class_res

    # Prepara i dati da salvare
    results = {
        "method_name": method_name,
        "params": params,
        "predicted_labels": list(map(bool, test_pred_labels)),
        "roc_curve": {
            "fpr": fpr.tolist(),  # Falsi positivi
            "tpr": tpr.tolist(),  # Veri positivi
            "roc_auc": float(roc_auc)   # Area sotto la curva ROC
        },
        "pr_curve": {
            "precision": precision.tolist(),  # Precisione
            "recall": recall.tolist(),        # Richiamo
            "auc_pr": float(auc_pr)                # Area sotto la curva Precision-Recall
        }
    }
    # Nome del file JSON
    filename = f"classification_results/{method_name}_res.json"

    # Salva i risultati in formato JSON
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)  # indent=4 per renderlo leggibile

    print(f"Risultati salvati in '{filename}'")



def load_classification_res(method_name):

    filename = f"../classification_results/{method_name}_res.json"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Il file '{filename}' non esiste.")

    with open(filename, 'r') as f:
        results = json.load(f)

    method_name = results["method_name"]
    params = results["params"]
    predicted_labels = list(map(bool, results["predicted_labels"]))
    roc_curve = results["roc_curve"]
    pr_curve = results["pr_curve"]

    fpr = roc_curve["fpr"]
    tpr = roc_curve["tpr"]
    roc_auc = roc_curve["roc_auc"]
    precision = pr_curve["precision"]
    recall = pr_curve["recall"]
    auc_pr = pr_curve["auc_pr"]

    probability_class_res = (fpr, tpr, roc_auc, precision, recall, auc_pr)

    return method_name, params, predicted_labels, probability_class_res


def train_test_for_non_distance_comp():
    dataset = data_for_non_distance_method_comp()
    test_set=dataset[dataset['race_year']>=2022]
    train_set=dataset[dataset['race_year']<2022]
    train_label=train_set.pop('top20')
    test_label=test_set.pop('top20')
    return train_set, test_set, train_label, test_label


def data_for_non_distance_method_comp():
    dataset = data_pre_processing_comp()
    categorical_columns = ['geo area']

    dataset = categorical_columns_encoding(dataset, categorical_columns)

    dataset.drop(categorical_columns, axis=1, inplace=True) #column already encoded
    return dataset

def data_pre_processing_comp():
    dataset, cyclists, placements = merge_all_dataset_comp()

    new_stats = create_cyclist_stats_2021(cyclists, placements)

    dataset = drop_previous_and_merge_newstats(dataset, new_stats)

    dataset = impute_null_stats_2021(dataset)

    dataset.drop(['delta', 'median_delta', 'std_delta'], axis=1, inplace=True)
    drop_redundant_columns(dataset)

    dataset['top20'] = (dataset['position'] < 20).astype(int)
    dataset.drop(['position'], axis=1, inplace=True)

    return dataset

def merge_all_dataset_comp():
    races = pd.read_csv('../../dataset/preprocessedRaces_without_outliers.csv', sep=",")
    placements = pd.read_csv('../../dataset/preprocessedPlacements.csv', sep=",")
    cyclists = pd.read_csv('../../dataset/preprocessedCyclists_without_outliers.csv', sep=",")
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