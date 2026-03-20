# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Alexander Del Brocco
# STUDENT EMAIL: sc22ajdb@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
# Task 1 [8 marks]: ----------------------------------------------------------- 
def load_data(file_path, delimiter=','):
## Loads Data from CSV to numpy array, with supporting variables
    num_rows, data, header_list=None, None, None

    ### Filepath verification
    if not os.path.isfile(file_path):
        warnings.warn(
            f"Task 1: Warning - CSV file '{file_path}' does not exist."
        )
        return None, None, None

    ### Open CSV, and extract data
    csv = pd.read_csv(file_path)
    header_list = pd.read_csv(file_path, nrows=0).columns.tolist()
    data = csv.to_numpy()
    num_rows = int(len(csv))
    print(header_list)

    return num_rows, data, header_list

# Task 2 [8 marks]: -----------------------------------------------------------
def filter_data(data):
## Removes rows with missing values
    filtered_data=[None]*1
    
    ### I don't know what you expect here, but I'm not touching your code!
    filtered_data.clear()

    ### Append every complete row to new array
    for row in data:
        if -99 in row:
            continue
        else:
            filtered_data.append(np.array(row))

    print(len(filtered_data))

    return np.array(filtered_data)

# Task 3 [8 marks]: -----------------------------------------------------------
def statistics_data(data):
## Returns the coefficient of variation for each column in data
    coefficient_of_variation=None

    feature_datasets = []
    coefficient_of_variation = []

    ### Use filter_data() to clear incomplete rows
    filtered_data = filter_data(data)

    ### Colect datasets for each feature
    for i in range(len(filtered_data[0])):
        dataset = []
        for row in filtered_data:
            dataset.append(row[i])
        feature_datasets.append(dataset)

    ### Use datasets to calculate stdev and means, to produces the coef of var
    for feature in feature_datasets:
        coefficient_of_variation.append(np.std(feature) / np.mean(feature))

    return np.array(coefficient_of_variation)

# Task 4 [8 marks]: -----------------------------------------------------------
def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test=None, None, None, None
    np.random.seed(1)

    ### Seperate label column from data (the end column)
    x = data[:, :-1] 
    label = data[:, -1] 

    ### Split training and testing data according to label ratio
    x_train, x_test, y_train, y_test = train_test_split(
        x, label, test_size=test_size, random_state=random_state, stratify=label
    )

    return x_train, x_test, y_train, y_test

# Task 5 [8 marks]: -----------------------------------------------------------
def train_decision_tree(x_train, y_train, ccp_alpha=0):
## Create a Decision Tree from provided data
    model=None
    
    ### Create and train decision tree from parameters
    model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)

    return model

# Task 6 [8 marks]: -----------------------------------------------------------
def make_predictions(model, X_test):
## Makes predictions on model based on parameter
    y_test_predicted=None
    
    y_test_predicted = model.predict(X_test)

    return y_test_predicted

# Task 7 [8 marks]: -----------------------------------------------------------
def evaluate_model(model, x, y):
## Compares model's predictions to real results
    accuracy, recall=None,None
    
    ### Make the prediction
    y_pred = make_predictions(model, x)

    ### Compare accuracy to real results
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, average="binary")

    return accuracy, recall

# Task 8 [8 marks]: -----------------------------------------------------------
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
## Calculates the optimal ccp alpha for a training and testing dataset
    optimal_ccp=None
    optimal_ccp, i = 0, 0

    ### Create and evaluation an initial model
    unpruned_model = train_decision_tree(x_train, y_train)
    unpruned_accuracy, ignore = evaluate_model(unpruned_model, x_test, y_test)

    ### Use previous evaluations for benchmarking accuracy
    low_acc_bound = unpruned_accuracy - 0.01
    upp_acc_bound = unpruned_accuracy + 0.01
    accuracy = unpruned_accuracy

    ### Increment through ccp alphas, pruning until break in accuaracy
    while accuracy > low_acc_bound and accuracy < upp_acc_bound:
        if optimal_ccp == 1 or i == 1000: break

        pruned_model = train_decision_tree(x_train, y_train, optimal_ccp)
        accuracy, ignore = evaluate_model(pruned_model, x_test, y_test)

        optimal_ccp += 0.001
        i += 1

    return optimal_ccp

# Task 9 [8 marks]: -----------------------------------------------------------
def tree_depths(model):
## Calculates the depth of a decision tree
    depth=None

    depth = model.get_depth()

    return depth

 # Task 10 [8 marks]: ---------------------------------------------------------
def important_feature(x_train, y_train, header_list):
## Extracts feature remaining at a pruned tree depth of 1
    best_feature=None
    ccp_alpha = 0

    ### Increment ccp_alpha until depth of 1, extract remaining feature
    while True:
        model = train_decision_tree(x_train, y_train, ccp_alpha)
        if tree_depths(model) <= 1 or ccp_alpha >= 1:
            break
        else:
            ccp_alpha += 0.01
    best_feature = header_list[model.tree_.feature[0]]

    return best_feature
    
# Task 11 [10 marks]: ---------------------------------------------------------
def optimal_ccp_alpha_single_feature(
        x_train, y_train, x_test, y_test, header_list):
## Returns optimal cpp_alpha for a reduced (important feature) dataset
    optimal_ccp=None
    new_x_train, new_x_test, new_y_train, new_y_test = [], [], [], []

    ### Deduce most important feature
    focus = important_feature(x_train, y_train, header_list)
    col = header_list.index(focus)

    ### Extract feature columns from datasets
    for xtr_row, xte_row, ytr_row, yte_row in zip(
        x_train, x_test, y_train, y_test
    ):
        new_x_train.append([xtr_row[col]])
        new_x_test.append([xte_row[col]])
        new_y_train.append([ytr_row])
        new_y_test.append([yte_row])

    ### Deduce optimal_ccp_alpha
    optimal_ccp = optimal_ccp_alpha(
        new_x_train, new_y_train, new_x_test, new_y_test
    )

    return optimal_ccp

# Task 12 [10 marks]: ---------------------------------------------------------
def optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list):
## Returns optimal ccp_alpha for a slightly reduces (two feature) dataset
    optimal_depth=None
    new_x_train, new_x_test, new_y_train, new_y_test = [], [], [], []

    ### Deduce most important feature
    focus_1 = important_feature(x_train, y_train, header_list)
    col_1 = header_list.index(focus_1)

    ### Remove the most important feature and deduce second most important feature
    remaining_header_list = [header for header in header_list if header != focus_1]
    remaining_x_train = [xtr_row[:col_1] + xtr_row[col_1 + 1:] for xtr_row in x_train]
    remaining_x_test = [xte_row[:col_1] + xte_row[col_1 + 1:] for xte_row in x_test]
    
    focus_2 = important_feature(remaining_x_train, y_train, remaining_header_list)
    col_2 = remaining_header_list.index(focus_2)

    ### Extract the two most important feature columns from datasets
    for xtr_row, xte_row, ytr_row, yte_row in zip(
        remaining_x_train, remaining_x_test, y_train, y_test
    ):
        new_x_train.append([xtr_row[col_2]])
        new_x_test.append([xte_row[col_2]])
        new_y_train.append([ytr_row])
        new_y_test.append([yte_row])

    # Add the first important feature back to the dataset
    for i in range(len(new_x_train)):
        new_x_train[i].insert(0, x_train[i][col_1])
        new_x_test[i].insert(0, x_test[i][col_1])

    ### Deduce optimal ccp_alpha using the two most important features
    optimal_ccp = optimal_ccp_alpha(
        new_x_train, new_y_train, new_x_test, new_y_test
    )

    ### Train new tree and deduce depth
    reduced_tree = train_decision_tree(new_x_train, new_y_train, optimal_ccp)
    optimal_depth = tree_depths(reduced_tree)

    return optimal_depth    

# Example usage (Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print(header_list)
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train, header_list)  # Insert your code here for task 1ader_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
    
    # Test optimal ccp_alpha with single feature
    optimal_alpha_single = optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal ccp_alpha using single most important feature: {optimal_alpha_single:.4f}")
    print("-" * 50)
    
    # Test optimal depth with two features
    optimal_depth_two = optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal tree depth using two most important features: {optimal_depth_two}")
    print("-" * 50)        
# References: 
# Here please provide recognition to any source if you have used or got code snippets from
# Please tell the lines that are relavant to that reference.
# For example: 
# Line 80-87 is inspired by a code at https://stackoverflow.com/questions/48414212/how-to-calculate-accuracy-from-decision-trees


