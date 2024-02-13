from numba import jit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import joblib
import networkx as nx
import dgl
import torch
import statistics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    auc, accuracy_score, balanced_accuracy_score, roc_auc_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
#import lightgbm as lgb
import xgboost as xgb
from dgl import DGLGraph
from dgl.nn import SAGEConv
from node2vec import Node2Vec
from Bio import SeqIO
import argparse
import subprocess
import sys
import os.path
from subprocess import Popen
from multiprocessing import Manager
import polars as pl
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.exceptions import FitFailedWarning
import shap
from interpretability_report import Report, REPORT_MAIN_TITLE_BINARY, REPORT_SHAP_PREAMBLE_BINARY, REPORT_SHAP_BAR_BINARY, \
    REPORT_SHAP_BEESWARM_BINARY, REPORT_SHAP_WATERFALL_BINARY

from utility_graphs import REPORT_USABILITY_TITLE_BINARY, REPORT_USABILITY_DESCRIPITION, \
    REPORT_1, REPORT_2, REPORT_3

from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from numpy.random import default_rng

from bio_extrac import extrac_math_features, graph_ext, make_graph, make_graph_all, make_graph_N
from utility_graphs import final_predictions, precision_graph, coverage_graph, make_fig_graph
from shap_graphs import type_model, shap_waterf, shap_bar, shap_beeswarm
from make_model import metrics, better_model, Score_table, objective_rf, tuning_rf_bayesian, objective_cb, tuning_catboost_bayesian, objective_gb, tuning_xgb_bayesian

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss 
from imblearn.under_sampling import EditedNearestNeighbours 
from imblearn.under_sampling import CondensedNearestNeighbour 
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek 
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/repDNA/repDNA/')
from nac import *
from psenac import *
from ac import *


scaler = StandardScaler()
warnings.filterwarnings('ignore')
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=40)

def imbalanced_techniques(model, tech, train, train_labels):

    """Testing imbalanced data techniques"""

    sm = tech
    pipe = Pipeline([('tech', sm), ('classifier', model)])
    #  train_new, train_labels_new = sm.fit_sample(train, train_labels)
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    acc = cross_val_score(pipe,
                          train,
                          train_labels,
                          cv=kfold,
                          scoring=make_scorer(balanced_accuracy_score),
                          n_jobs=n_cpu).mean()
    return acc


def imbalanced_function(clf, train, train_labels):

    """Preprocessing: Imbalanced datasets"""

    print('Checking for imbalanced labels...')
    df = pd.DataFrame(train_labels)
    n_labels = pd.value_counts(df.values.flatten())
    if all(x == n_labels[0] for x in n_labels) is False:
        print('There are imbalanced labels...')
        print('Checking the best technique...')
        performance = []
        smote = imbalanced_techniques(clf, SMOTE(random_state=42), train, train_labels)
        random = imbalanced_techniques(clf, RandomUnderSampler(random_state=42), train, train_labels)
        cluster = imbalanced_techniques(clf, ClusterCentroids(random_state=42), train, train_labels)
        near = imbalanced_techniques(clf, EditedNearestNeighbours(), train, train_labels)
        near_miss = imbalanced_techniques(clf, NearMiss(), train, train_labels)
        performance.append(smote)
        performance.append(random)
        performance.append(cluster)
        performance.append(near)
        performance.append(near_miss)
        max_pos = performance.index(max(performance))
        if max_pos == 0:
            print('Applying Smote - Oversampling...')
            sm = SMOTE(random_state=42)
            train, train_labels = sm.fit_resample(train, train_labels)
        elif max_pos == 1:
            print('Applying Random - Undersampling...')
            sm = RandomUnderSampler(random_state=42)
            train, train_labels = sm.fit_resample(train, train_labels)
        elif max_pos == 2:
            print('Applying ClusterCentroids - Undersampling...')
            sm = ClusterCentroids(random_state=42)
            train, train_labels = sm.fit_resample(train, train_labels)
        elif max_pos == 3:
            print('Applying EditedNearestNeighbours - Undersampling...')
            sm = EditedNearestNeighbours()
            train, train_labels = sm.fit_resample(train, train_labels)
        else:
            print('Applying NearMiss - Undersampling...')
            sm = NearMiss()
            train, train_labels = sm.fit_resample(train, train_labels)
    else:
        print('There are no imbalanced labels...')
    return train, train_labels, sm

def fitting(test_data, score_name, clf_fit, X_test, y_test, output_table, output_metrics=None):
    """
    Fit a classifier, generate predictions and scores, and create a score table. Optionally, calculate metrics.

    Parameters:
        test_data (DataFrame): DataFrame containing test data, including sequence names.
        score_name (str): Name of the score column.
        clf_fit (Classifier): A trained classifier.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        output_table (str): File path where the score table will be saved.
        output_metrics (str, optional): File path where metrics will be saved. Default is None.

    Returns:
        model (DataFrame): The score table DataFrame.
    """
    # Generate predictions using the trained classifier
    #print(X_test.values)
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    #print(X_test)
    predictions = clf_fit.predict(X_test.values)
    preds_proba = clf_fit.predict_proba(X_test.values)[:, 1]
    # Calculate and save metrics if output_metrics is provided
    if output_metrics is not None:
        metrics(predictions, y_test, preds_proba, output_metrics)

    # Generate probability scores for the positive class
    scores = clf_fit.predict_proba(X_test.values)[:, 1]

    # Create a score table with sequence names, predicted labels, and scores
    model = Score_table(test_data, predictions, scores, score_name, output_table)

    return model


def make_dataset(test_edges, carac, carac2):
    """
    Create a dataset for testing by merging test_edges with carac (characteristics) data.

    Parameters:
        test_edges (DataFrame): DataFrame containing test edges, including source and target nodes.
        carac (DataFrame): DataFrame containing characteristics data, indexed by node identifiers.

    Returns:
        X_test (DataFrame): Features of the test dataset.
        y_test (Series): Labels of the test dataset.
        test_data (DataFrame): Merged dataset for testing.
    """
    columns = test_edges.columns

    # Ensure that the index of carac is of string type
    carac.index = carac.index.astype(str)
    carac2.index = carac2.index.astype(str)
    
    #carac = carac.drop_duplicates()
    #carac2 = carac2.drop_duplicates()
    test_edges.iloc[:, :2] = test_edges.iloc[:, :2].astype(str)

    
    
    # Merge test_edges with carac based on source and target nodes
    test_data = test_edges.merge(carac2, left_on=columns[1], right_index=True).merge(carac, left_on=columns[0], right_index=True)

    
    print('test_d', test_data)
    
    test_data = test_data.drop_duplicates()
    
    print('test_d2', test_data)
    # Remove duplicate rows from the merged dataset
    
    # Separate features (X_test) and labels (y_test)
    X_test = test_data.drop(columns, axis=1)
    
    X_test = X_test.select_dtypes(include='number')
    X_test = X_test.astype(float)
    
    y_test = test_data[columns[2]]
    y_test = y_test.astype(float)
    
    
   
    
    

    return X_test, y_test, test_data

def make_trains(train, test, final_test, carac, carac2):
    """
    Create training and testing datasets by merging edge data with characteristics data.

    Parameters:
        train (str): Path to the training data file (edges).
        test (str): Path to the testing data file (edges).
        final_test (str): Path to the final testing data file (edges).
        carac (DataFrame): DataFrame containing characteristics data, indexed by node identifiers.

    Returns:
        X_train (DataFrame): Features of the training dataset.
        y_train (Series): Labels of the training dataset.
        X_test (DataFrame): Features of the testing dataset.
        y_test (Series): Labels of the testing dataset.
        test_data (DataFrame): Merged dataset for testing.
        X_final (DataFrame): Features of the final testing dataset.
        y_final (Series): Labels of the final testing dataset.
        test_data_final (DataFrame): Merged dataset for final testing.
    """
    # Read training, testing, and final testing data from CSV files
    train_edges = pd.read_csv(train, sep=',')
    test_edges = pd.read_csv(test, sep=',')
    final_test = pd.read_csv(final_test, sep=',')
    
        # Drop duplicate rows in the characteristics data
    
    #carac = carac.drop_duplicates()
    #carac2 = carac2.drop_duplicates()

    # Create datasets for training, testing, and final testing
    X_train, y_train, test_data = make_dataset(train_edges, carac, carac2)
    X_test, y_test, test_data = make_dataset(test_edges, carac, carac2)
    X_final, y_final, test_data_final = make_dataset(final_test, carac, carac2)



    return X_train, y_train, X_test, y_test, test_data, X_final, y_final, test_data_final


def partial_models(index, datasets, names, train_edges, test_edges, final_test, partial_folds):
    """
    Create and save partial models and associated data for a specific dataset.

    Parameters:
        index (str): Name of the index column in datasets.
        datasets (str): Path to the dataset file containing characteristics data.
        names (str): Name or identifier for the dataset.
        train_edges (str): Path to the training data file (edges).
        test_edges (str): Path to the testing data file (edges).
        final_test (str): Path to the final testing data file (edges).
        partial_folds (str): Directory where partial models and data will be saved.

    Returns:
        None
    """
    # Read characteristics data from CSV file and set index
    carac = pd.read_csv(datasets, sep=",")    
    carac.set_index(index, inplace=True)

    # Check if 'label' column exists in carac and drop it if present
    if 'label' in carac.columns:
        carac = carac.drop(columns='label')

    # Create training, testing, and final testing datasets
    X_train, y_train, X_test, y_test, test_data, X_final, y_final, test_data_final = \
        make_trains(train_edges, test_edges, final_test, carac, carac)
    
    #print(X_train)
    
    #print('treino', y_train)
    '''
    duplicated_indices = X_train.duplicated(keep='first')
    X_train = X_train[~duplicated_indices]
    y_train = y_train[~duplicated_indices]

    duplicated_indices = X_test.duplicated(keep='first')
    X_test = X_test[~duplicated_indices]
    y_test = y_test[~duplicated_indices]
    test_data = test_data[~duplicated_indices]

    duplicated_indices = X_final.duplicated(keep='first')
    X_final = X_final[~duplicated_indices]
    y_final = y_final[~duplicated_indices]
    test_data_final = test_data_final[~duplicated_indices]
    '''
    #print('duplicata', y_train)
    
    # Train a classifier, save the model, and generate score tables and metrics
    clf = better_model(X_train, y_train, X_test, y_test, partial_folds+'/model_'+names+'.sav', tuning=False)      
    fitting(test_data, 'Score_'+names, clf, X_test, y_test, 
            partial_folds+'/data_train_'+names+'.csv',
            partial_folds+'/metrics_'+names+'.csv')        
    fitting(test_data_final, 'Score_'+names, clf, X_final, y_final,
            partial_folds+'/data_test_'+names+'.csv')

def partial_models2(index, datasets, names, datasets2, names2, train_edges, test_edges, final_test, partial_folds):
    """
    Create and save partial models and associated data for a specific dataset.

    Parameters:
        index (str): Name of the index column in datasets.
        datasets (str): Path to the dataset file containing characteristics data.
        names (str): Name or identifier for the dataset.
        train_edges (str): Path to the training data file (edges).
        test_edges (str): Path to the testing data file (edges).
        final_test (str): Path to the final testing data file (edges).
        partial_folds (str): Directory where partial models and data will be saved.

    Returns:
        None
    """
    n_cpu = 1
    # Read characteristics data from CSV file and set index
    #print('data', names, names2)
    carac = pd.read_csv(datasets, sep=",")
    carac.set_index(index, inplace=True)
    carac2 = pd.read_csv(datasets2, sep=",")
    carac2.set_index(index, inplace=True)
    
    #print('aq', carac, carac2)

    #print(carac)
    # Check if 'label' column exists in carac and drop it if present
    if 'label' in carac.columns:
        carac = carac.drop(columns='label')
    if 'label' in carac2.columns:
        carac2 = carac2.drop(columns='label')
      

    # Create training, testing, and final testing datasets
    X_train, y_train, X_test, y_test, test_data, X_final, y_final, test_data_final = \
        make_trains(train_edges, test_edges, final_test, carac, carac2)
    '''
    duplicated_indices = X_train.duplicated(keep='first')
    X_train = X_train[~duplicated_indices]
    y_train = y_train[~duplicated_indices]

    duplicated_indices = X_test.duplicated(keep='first')
    X_test = X_test[~duplicated_indices]
    y_test = y_test[~duplicated_indices]
    test_data = test_data[~duplicated_indices]

    duplicated_indices = X_final.duplicated(keep='first')
    X_final = X_final[~duplicated_indices]
    y_final = y_final[~duplicated_indices]
    test_data_final = test_data_final[~duplicated_indices]
    '''
    #print(X_train)
    # Train a classifier, save the model, and generate score tables and metrics
    clf = better_model(X_train, y_train, X_test, y_test, partial_folds+'/model_'+names+'_'+names2+'.sav')      
    fitting(test_data, 'Score_'+names+'_'+names2, clf, X_test, y_test, 
            partial_folds+'/data_train_'+names+'_'+names2+'.csv',
            partial_folds+'/metrics_'+names+'_'+names2+'.csv')        
    fitting(test_data_final, 'Score_'+names+'_'+names2, clf, X_final, y_final,
            partial_folds+'/data_test_'+names+'_'+names2+'.csv')
    
    
def concat_data_models(datas, output, name):
    """
    Concatenate data from multiple models and save the final training dataset.

    Parameters:
        datas (list): List of paths to CSV files containing data from multiple models.
        output (str): Directory where the final training data will be saved.
        name (str): Name or identifier for the concatenated dataset.

    Returns:
        final_X_train (DataFrame): Final training data features.
        final_y_train (DataFrame): Final training data labels.
    """
    # Read the first dataset file
    merged_train = pd.read_csv(datas[0], sep=',')
    merged_trainB = pd.read_csv(datas[1], sep=',')
    
    # Merge the first two datasets on the 'nameseq' column and drop the redundant 'Label_x'
    merged_train = merged_train.merge(merged_trainB, on='nameseq')
    #print(merged_train)
    #f(merged_train.columns)
    merged_train = merged_train.drop(columns=['Label_x'], axis=1)

    # Iterate through the remaining dataset files and merge them, dropping the 'Label' column each time
    for k in range(2, len(datas)):
        merged_trainB = pd.read_csv(datas[k], sep=',')
        #print(merged_trainB.columns, merged_train.columns)
        merged_trainB = merged_trainB.drop(columns=['Label'], axis=1)
        #print(merged_train, merged_trainB)
        merged_train = merged_train.merge(merged_trainB, on='nameseq')
        #merged_train = merged_train.drop(columns=['Label'], axis=1)

    # Extract the 'nameseq' and 'Label_y' columns as final training data
    merged_nameseq = merged_train[['nameseq']].copy()
    final_y_train = merged_train[['Label_y']].copy()
    final_X_train = merged_train.drop(columns=['nameseq', 'Label_y'])

    # Save the final training data as CSV files
    final_X_train.to_csv(output+'/final_X_'+name+'.csv', index=False) 
    final_y_train.to_csv(output+'/final_y_'+name+'.csv', index=False) 
    merged_nameseq.to_csv(output+'/final_nameseq_'+name+'.csv', index=False)
    
    return final_X_train, final_y_train, merged_nameseq

def interp_shap(model, X_test, X_label, output, path='explanations'):
    """
    Generate various types of SHAP interpretation graphs for a given model and dataset.
    
    Parameters:
    - model: The machine learning model to be explained.
    - X_test: The test data for which SHAP values will be calculated.
    - X_label: The labels or target values associated with the test data.
    - output: The output directory where the explanation graphs will be saved.
    - path: The subdirectory within 'output' where the explanation graphs will be saved (default is 'explanations').

    Returns:
    - generated_plt: A dictionary containing the file paths of the generated explanation graphs.
    """
    path = os.path.join(output, path)
    generated_plt = {}
    
    # Create a SHAP explainer for the model with tree-based feature perturbation.
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    
    # Calculate SHAP values and potentially transform labels based on model type.
    shap_values, X_label = type_model(explainer, model, X_test.values, X_label)
    
    if not os.path.exists(path):
        print(f"Creating explanations directory: {path}...")
        os.mkdir(path)
    else:
        print(f"Directory {path} already exists. Will proceed using it...")

    generated_plt['bar_graph'] = [shap_bar(shap_values, path, fig_name='bar_graph')]
    generated_plt['beeswarm_graph'] = [shap_beeswarm(shap_values, path, fig_name='beeswarm_graph')]
    generated_plt['waterfall_graph'] = shap_waterf(explainer, model, X_test.values, X_label, path)
    return generated_plt


   
def build_interpretability_report(generated_plt=[], report_name="interpretability.pdf", directory="."):
    """
    Build an interpretability report by combining generated SHAP interpretation graphs into a PDF report.
    
    Parameters:
    - generated_plt: A dictionary containing file paths of generated explanation graphs.
    - report_name: The name of the PDF report to be generated (default is "interpretability.pdf").
    - directory: The directory where the PDF report will be saved (default is the current directory).
    """
    report = Report(report_name, directory=directory)
    root_dir = os.path.abspath(os.path.join(__file__, os.pardir))

    report.insert_doc_header(REPORT_MAIN_TITLE_BINARY, logo_fig=os.path.join(root_dir, "img/BioAutoML.png"))
    report.insert_text_on_doc(REPORT_SHAP_PREAMBLE_BINARY, font_size=14)

    report.insert_figure_on_doc(generated_plt['bar_graph'])
    report.insert_text_on_doc(REPORT_SHAP_BAR_BINARY, font_size=14)

    report.insert_figure_on_doc(generated_plt['beeswarm_graph'])
    report.insert_text_on_doc(REPORT_SHAP_BEESWARM_BINARY, font_size=12)

    report.insert_figure_on_doc(generated_plt['waterfall_graph'])
    report.insert_text_on_doc(REPORT_SHAP_WATERFALL_BINARY, font_size=12)

    report.build()
    
def utility(output_data, output):
    generated_plt = {}
    output = output + '/utility'
    make_fold(output)
    
    generated_plt['precision'] = precision_graph(output_data, output)
    generated_plt['coverage'] = coverage_graph(output_data, output)
    make_fig_graph(output_data, output)
    return generated_plt

def build_usability_report(generated_plt=[], report_name="usability.pdf", directory="."):
    """
    Build an interpretability report by combining generated SHAP interpretation graphs into a PDF report.
    
    Parameters:
    - generated_plt: A dictionary containing file paths of generated explanation graphs.
    - report_name: The name of the PDF report to be generated (default is "interpretability.pdf").
    - directory: The directory where the PDF report will be saved (default is the current directory).
    """
    report = Report(report_name, directory=directory)
    root_dir = os.path.abspath(os.path.join(__file__, os.pardir))

    report.insert_doc_header(REPORT_USABILITY_TITLE_BINARY, logo_fig=os.path.join(root_dir, "img/BioAutoML.png"))
    report.insert_text_on_doc(REPORT_USABILITY_DESCRIPITION, font_size=14)

    #report.insert_figure_on_doc(generated_plt['precision'])
    report.insert_text_on_doc(REPORT_1, font_size=14)
    
    #report.insert_figure_on_doc(generated_plt['coverage'])
    report.insert_text_on_doc(REPORT_1, font_size=14)
    report.build()

def real_part(complex_num):
    return complex_num.real

def extrac_features_topology(train_edges, edges, output):
    """
    Extract various graph topology features and save them to CSV files.

    Parameters:
        train_edges (pd.DataFrame): DataFrame containing training edge data.
        edges (pd.DataFrame): DataFrame containing edge information.
        output (str): Output directory for saving feature files.
    """

    feat_output = output+'/feat_topology_P.csv'
    
    # Create a graph from training edges (considering positive edges only)
    G_trP = make_graph(train_edges, edges)    
    data_nodes_trainP = graph_ext(G_trP)    
    graph_feat_trainP = pd.DataFrame(data_nodes_trainP)
    graph_feat_trainP.set_index('Node', inplace=True)
    
    # Extract real parts of complex numbers (if any) in the DataFrame
    graph_feat_trainP = graph_feat_trainP.applymap(real_part) 
    
    # Save the graph features to a CSV file
    feat_output = output+'/feat_topology_P.csv'
    graph_feat_trainP.to_csv(feat_output, index=True)

    # Create a graph from training edges (considering all edges, including negative)
    G_trN = make_graph_all(train_edges, edges)
    data_nodes_trainN = graph_ext(G_trN)
    graph_feat_trainN = pd.DataFrame(data_nodes_trainN)
    graph_feat_trainN.set_index('Node', inplace=True)
    
    # Extract real parts of complex numbers (if any) in the DataFrame
    graph_feat_trainN = graph_feat_trainN.applymap(real_part)
    
    # Save the graph features to a CSV file
    feat_output = output+'/feat_topology_N.csv'
    graph_feat_trainN.to_csv(feat_output, index=True)
    
    
    
    '''
    G_trN = make_graph_N(train_edges, edges)
    data_nodes_trainN = graph_ext(G_trN)
    graph_feat_trainN = pd.DataFrame(data_nodes_trainN)
    graph_feat_trainN.set_index('Node', inplace=True)
    
    # Extract real parts of complex numbers (if any) in the DataFrame
    graph_feat_trainN = graph_feat_trainN.applymap(real_part)
    
    # Save the graph features to a CSV file
    feat_output = output+'/feat_topology_N.csv'
    graph_feat_trainN.to_csv(feat_output, index=True)
    ''' 

def check_path(paths, type_path='This path'):
    for subpath in paths:
        if os.path.exists(subpath):
            print(f'{type_path} - {subpath}: Found File')
        else:
            print(f'{type_path} - {subpath}: File not exists')
            sys.exit()            

def debug_path(path_input):
    path_output = []
    for i in range(len(path_input)):
        path_output.append(os.path.join(path_input[i]))
    return path_output

def make_fold(path_results):
    if not os.path.exists(path_results):
        os.mkdir(path_results)
        
def feat_eng(input_interactions_train, sequences_dictionary, stype, n_cpu, foutput, extrac_topo_features = False):
    input_interactions_train = pd.read_csv(input_interactions_train, sep=',')
    
    global train_edges_output,  test_edges_output, final_edges_output, partial_folds, output_folds, output_folds_number, features_amino
    
    #extrac_topo_features = False
    extrac_math_featuresB = False
    output_folds = foutput+'/folds_and_topology_feats'
    make_fold(output_folds)

    feat_path = foutput + '/extructural_feats'
    make_fold(feat_path)

    print('Make the folds')

    edges = input_interactions_train[input_interactions_train.columns]
    #fprint(input_interactions_train.columns)
    
    train_edges_output = []
    test_edges_output = []
    final_edges_output = []
    output_folds_number = []
    partial_folds = []

    for fold, (train_index, test_index) in enumerate(kf.split(input_interactions_train)):
        train_edges = input_interactions_train.iloc[train_index]
        final_test = input_interactions_train.iloc[test_index]
        train_edges, test_edges = train_test_split(train_edges, test_size=1/8, random_state=42)
        output_folds_number.append(output_folds +'/fold'+str(fold+1))
        make_fold(output_folds_number[fold])

        partial_folds.append(output_folds_number[fold]+'/partial_models')
        make_fold(partial_folds[fold])

        train_edges_output.append(output_folds_number[fold]+'/edges_train.csv')
        train_edges.to_csv(train_edges_output[fold], index=False)
        
        test_edges_output.append(output_folds_number[fold]+'/edges_test.csv')
        test_edges.to_csv(test_edges_output[fold], index=False)

        final_edges_output.append(output_folds_number[fold]+'/edges_final_test.csv')
        final_test.to_csv(final_edges_output[fold], index=False)

        if extrac_topo_features:
            print('Topology features extraction fold'+str(fold+1))
            extrac_features_topology(train_edges, edges, output_folds_number[fold])

        
    names_topo = ['topology_P', 'topology_N']
    datasets_topo=[]
    for p in range(num_folds):            
            datasets_topo.append([foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+names_topo[0]+'.csv',
             foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+names_topo[1]+'.csv'])

    features_amino = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,25]
    if extrac_math_featuresB:  
        datasets_extr, names_math = extrac_math_features(features_amino, sequences_dictionary, stype, feat_path)
        if stype == 0:
         
    # Selecionar a primeira coluna como índice para os conjuntos de dados restantes
            df_first = pd.read_csv(datasets_extr[0], sep=',')
            index_column = df_first.columns[0]
            #print(index_column)
            # Iterar pelos conjuntos de dados restantes
            for i in range(5, 6+6):
                # Carregar o conjunto de dados
                #print(i, datasets_extr)
                print('aq2',datasets_extr[i])
                df = pd.read_csv(datasets_extr[i], sep=',')
                df['nameseq'] =  df_first[index_column]

                # Definir a primeira coluna como índice
                df.set_index(index_column, inplace=True)
                print('novas feats com nome', df)
                # Salvar o conjunto de dados com o mesmo nome (substituindo o original)
                df.to_csv(datasets_extr[i], index=True)
        

    #datasets_extr = [] ####################### AQUI ###########################
    #names_math = []
    #names_math_feat=['Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40',
    #                 'ComplexNetworks', 'kGap_di', 'AAC', 'DPC']

    #for i in features_amino:
    #    names_math.append(names_math_feat[i-1])
    #    datasets_extr.append(foutput+'/extructural_feats/'+names_math_feat[i-1]+'.csv')    
        

def fit_mod(input_interactions_train, sequences_dictionary, n_cpu, foutput):
    parcial_models_cond = True
    final_model = True
    
    
    datasets_extr = []
    names_math = []    
    
    df = []
    datasets_conj = []
    
    names_math_feat=['AAC_protein', 'DPC_protein', 'Mean_feat']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    #print(df)
    datasets_conj = pd.concat([df[0], df[1], df[2]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_1_protein'+'.csv')    
    names_math.append('partial_1_protein')
    datasets_extr.append(foutput+'/extructural_feats/'+'partial_1_protein'+'.csv')
    
    df = []
    datasets_conj = []
    names_math_feat=['feat_H1', 'feat_P1', 'feat_V']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    datasets_conj = pd.concat([df[0], df[1], df[2]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_2_protein'+'.csv')
    
    names_math.append('partial_2_protein')
    datasets_extr.append(foutput+'/extructural_feats/'+'partial_2_protein'+'.csv')
    
    df = []
    datasets_conj = []
    names_math_feat=['feat_H2', 'feat_P2', 'feat_NCI', 'feat_SASA']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    datasets_conj = pd.concat([df[0], df[1], df[2], df[3]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_3_protein'+'.csv')
    
    names_math.append('partial_3_protein')
    datasets_extr.append(foutput+'/extructural_feats/'+'partial_3_protein'+'.csv')
    
    
    df = []
    datasets_conj = []
    names_math_feat=['fq_grups']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    datasets_conj = pd.concat([df[0]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_4_protein'+'.csv')
    
    names_math.append('partial_4_protein')
    datasets_extr.append(foutput+'/extructural_feats/'+'partial_4_protein'+'.csv')
    
    

       

    datasets_extr2 = []
    names_math2 = []
    df = []
    datasets_conj = []
    
    names_math_feat=['NAC_dna', 'DNC_dna', 'TNC_dna']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    datasets_conj = pd.concat([df[0], df[1], df[2]], axis=1)
    #print('datasss', datasets_conj)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_1_dna'+'.csv')
    
    names_math2.append('partial_1_dna')
    datasets_extr2.append(foutput+'/extructural_feats/'+'partial_1_dna'+'.csv')
    
    
    datasets_conj = []
    df = []
    names_math_feat=['Rev_kmer_dna', 'Pse_dnc_dna', 'Pse_knc_dna']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    datasets_conj = pd.concat([df[0], df[1], df[2]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_2_dna'+'.csv')
    names_math2.append('partial_2_dna')
    datasets_extr2.append(foutput+'/extructural_feats/'+'partial_2_dna'+'.csv')
    
    
    datasets_conj = []
    df = []
    names_math_feat=['SCPseDNC_dna', 'SCPseTNC_dna']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    datasets_conj = pd.concat([df[0], df[1]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_3_dna'+'.csv')
    names_math2.append('partial_3_dna')
    datasets_extr2.append(foutput+'/extructural_feats/'+'partial_3_dna'+'.csv')
    
    
    datasets_conj = []
    df = []
    names_math_feat=['QNC_dna']
    for i in range(len(names_math_feat)):
        datasets_conj.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        
        df.append(pd.read_csv(datasets_conj[i], index_col='nameseq'))
    for i in range(len(df)):
        df[i] = df[i].drop(columns=['label'])
    datasets_conj = pd.concat([df[0]], axis=1)
    datasets_conj.to_csv(foutput+'/extructural_feats/'+'partial_4_dna'+'.csv')
    names_math2.append('partial_4_dna')
    datasets_extr2.append(foutput+'/extructural_feats/'+'partial_4_dna'+'.csv')
    
    
    

    
    datasets_topo = []
    names_topo = []
    for p in range(num_folds):            
        df1 = foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+'topology_P'+'.csv'
        df2 = foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+'topology_N'+'.csv'
        #df3 = foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+'topology_A'+'.csv'
        
        df1 = pd.read_csv(df1)
        df2 = pd.read_csv(df2)
        #df3 = pd.read_csv(df3)
        
        
        dfs = []
        dfs.append(df1)
        dfs.append(df2)
        #dfs.append(df3)
        

        result_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
        result_df = result_df.drop(columns=['Node_y'])
        result_df = result_df.rename(columns={'Node_x': 'Node'})
        '''
        result_df = pd.merge(result_df, dfs[2], left_index=True, right_index=True)
        result_df = result_df.drop(columns=['Node_y'])
        result_df = result_df.rename(columns={'Node_x': 'Node'})
        '''
        
        #print('topo_df', result_df)
        result_df.to_csv(foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+'topology'+'.csv')
        datasets_topo.append(foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+'topology'+'.csv')
        names_topo.append('topology')
     
            


    if parcial_models_cond:                 
        for p in range(num_folds): 
            #print(datasets_topo[p])
            carac = pd.read_csv(datasets_topo[p], sep=",")

            carac.set_index('Node', inplace=True)
            carac2 = pd.read_csv(datasets_topo[p], sep=",")
            carac2.set_index('Node', inplace=True)
            #print(carac2)

            # Check if 'label' column exists in carac and drop it if present
            if 'label' in carac.columns:
                carac = carac.drop(columns='label')
            if 'label' in carac2.columns:
                carac2 = carac2.drop(columns='label')

            X_tr, y_tr, X_te, y_te, te_data, X_fi, y_fi, te_data_final = \
                make_trains(train_edges_output[p], test_edges_output[p], final_edges_output[p], carac, carac)

            #X_tr, y_tr, sm = imbalanced_function(xgb.XGBClassifier(random_state=43), X_tr, y_tr)
            print('aaaaaaaaa')


            '''
            # Suponha que train_edges_output[p] seja o caminho para o arquivo CSV
            caminho_do_arquivo = train_edges_output[p]

            # Carregue o dataset original
            dataset_original = pd.read_csv(caminho_do_arquivo)

            # Separando as features (X) e os rótulos (y)
            X = dataset_original.drop('Label', axis=1)
            y = dataset_original['Label']

            # Aplicando o RandomUnderSampler
            sm = RandomUnderSampler(random_state=42)
            #print(X, y)
            X_resampled, y_resampled = sm.fit_resample(X, y)

            # Criando um novo DataFrame com os dados resampleados
            dataset_resampleado = pd.DataFrame(X_resampled, columns=X.columns)
            dataset_resampleado['Label'] = y_resampled

            # Salvar o DataFrame resampleado de volta no lugar original
            dataset_resampleado.to_csv(caminho_do_arquivo, index=False)


            #print(dataset_resampleado)
            '''

            #train_edges_output[p], train_edges_output[p] = sm.fit_resample(train_edges_output[p], train_edges_output[p])

            trains = []
            tests = []  

            partial_models('Node', datasets_topo[p], names_topo[0], train_edges_output[p], test_edges_output[p], final_edges_output[p], partial_folds[p])
            trains.append(partial_folds[p]+'/data_train_'+names_topo[0]+'.csv')
            tests.append(partial_folds[p]+'/data_test_'+names_topo[0]+'.csv')

            for j in range(len(datasets_extr)):
                #print(j, datasets_extr, datasets_extr2)
                partial_models2('nameseq', datasets_extr[j], names_math[j], datasets_extr2[j], names_math2[j], train_edges_output[p], 
                               test_edges_output[p], final_edges_output[p], partial_folds[p])
                trains.append(partial_folds[p]+'/data_train_'+names_math[j]+'_'+names_math2[j]+'.csv')
                tests.append(partial_folds[p]+'/data_test_'+names_math[j]+'_'+names_math2[j]+'.csv')                
                
            concat_output = output_folds +'/fold'+str(p+1)
            final_X_train, final_y_train, nameseq = concat_data_models(trains, concat_output, 'train')
            final_X_test, final_y_test, nameseq = concat_data_models(tests, concat_output, 'test')

            global X_train, y_train

            X_f = final_X_train
            y_f = final_y_train
            '''
            X_train = X_f
            y_train = y_f

            # Aplicando o RandomUnderSampler
            sm = RandomUnderSampler(random_state=42)
            X_f, y_f = sm.fit_resample(X_f, y_f)
            X_train = X_f
            y_train = y_f
            '''
            #print('final data', X_f, y_f)
            clf = better_model(X_f, y_f, final_X_test, final_y_test, 
                               output_folds_number[p]+'/model_final.sav', tuning=True)


            #generated_plt = interp_shap(clf, final_X_train, final_y_train, output_folds_number[p]) 
            #build_interpretability_report(generated_plt=generated_plt, directory=output_folds_number[p])

            output_data = final_predictions(final_X_test, final_y_test, nameseq, clf, output_folds_number[p])
            #generated_plt = utility(output_data, output_folds_number[p])
            #build_usability_report(generated_plt, report_name="usability.pdf", directory=output_folds_number[p])

            predictions = clf.predict(final_X_test.values)
            preds_proba = clf.predict_proba(final_X_test.values)[:, 1]
            metrics(predictions, final_y_test, preds_proba, output_folds_number[p]+'/metrics_model_final.csv')
            
    metrics_output = []
    for p in range(num_folds):
        metrics_output.append(foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/metrics_model_final.csv')

    if final_model:
        print('Calculating the average of metrics')          
        metrics1 = pd.read_csv(metrics_output[0], sep=',')
        for i in range(0,len(metrics_output)):
            metrics2 = pd.read_csv(metrics_output[i], sep=',')
            metrics1 = pd.concat([metrics1 , metrics2])
        new_dataframe = pd.DataFrame(columns=['Metric', 'Mean', 'Standard deviation'])
        metric_name = ['Precision_1', 'Recall_1', 'F1_1', 'Specificity_1', 'Precision_0', 
                            'Recall_0', 'F1_0', 'Accuracy', 'AUC', 'AUPR', 
                       'Balanced_Accuracy', 'MCC']
        for i in metric_name:
            metricA = metrics1[metrics1['Metrics'] == i]
            mean = metricA['Values'].mean()
            std = metricA['Values'].std()
            new_line = {'Metric': i, 'Mean': mean, 'Standard deviation': std}    
            new_dataframe = new_dataframe.append(new_line, ignore_index=True)

        new_dataframe.to_csv(output_folds+'\cross_validation_metrics.csv', index=False)
        print('Final perform:')
        print()
        print(new_dataframe)          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_interactions_train', '--input_interactions_train',  
                        help='path of table with tree columns (firts protein A, secund protein B \
                        and third interaction label) e.g., interations_train.txt')
    
    parser.add_argument('-input_interactions_test', '--input_interactions_test', 
                        help='path of table with tree columns (firts protein A, secund protein B \
                        and third interaction label) e.g., interations_test.txt')
    
    parser.add_argument('-sequences_dictionary_protein', '--sequences_dictionary_protein', help='all sequences in \
                        the problem in fasta format, e.g., dictionary.fasta')
    
    parser.add_argument('-sequences_dictionary_rna', '--sequences_dictionary_rna', help='all sequences in \
                        the problem in fasta format, e.g., dictionary.fasta')
    
    parser.add_argument('-output', '--output', help='resutls directory, e.g., result/')
    parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
    
    args = parser.parse_args() 
    input_interactions_train = args.input_interactions_train
    input_interactions_test = args.input_interactions_test
    sequences_dictionary_protein = args.sequences_dictionary_protein
    sequences_dictionary_rna = args.sequences_dictionary_rna
    foutput = args.output
    n_cpu = args.n_cpu
    
    check_path([input_interactions_train], 'input_interactions_train')  
    #input_interactions_train = pd.read_csv(input_interactions_train, sep=',')
        
    if None != input_interactions_test:
        check_path([input_interactions_test], 'input_interactions_test')
        
    check_path([sequences_dictionary_protein], 'sequences_dictionary_protein') 
    check_path([sequences_dictionary_rna], 'sequences_dictionary_rna') 
    make_fold(foutput)
    
    #print(input_interactions_train)
    
    stype = [0,1]
    feat_eng(input_interactions_train, sequences_dictionary_rna, stype[0], n_cpu, foutput, extrac_topo_features=True)
    feat_eng(input_interactions_train, sequences_dictionary_protein, stype[1], n_cpu, foutput)
    fit_mod(input_interactions_train, sequences_dictionary_protein, n_cpu, foutput)
    
    #python BioPrediction.py -input_interactions_train data/dataset_1/LPI_new.csv -sequences_dictionary_protein data/dataset_1/protein_Sequence.txt -sequences_dictionary_rna data/dataset_1/lncRNA_Sequence.txt -output test_rna_new
    
    #python BioPrediction.py -input_interactions_train data/dataset_4/int.csv -sequences_dictionary_protein data/dataset_4/output_proteins.fasta -sequences_dictionary_rna data/dataset_4/output_dna.fasta -output test_rna_new4
    
    #python BioPrediction.py -input_interactions_train data/NPInter_pairs.csv -sequences_dictionary_protein data/NPInter_proteins_seq.fa -sequences_dictionary_rna data/NPInter_dna_seq.fa -output test_rna_tipo2
    
    #python BioPrediction.py -input_interactions_train data/RPI2241_pairs.csv -sequences_dictionary_protein data/RPI2241_protein_seq.fa -sequences_dictionary_rna data/RPI2241_dna_seq.fa -output test_rna_2241
    
        #python BioPrediction.py -input_interactions_train data/RPI2241_pairs.csv -sequences_dictionary_protein data/RPI2241_protein_seq.fa -sequences_dictionary_rna data/RPI2241_dna_seq.fa -output test_rna_2241_2
    
    #python BioPrediction.py -input_interactions_train data/RPI369/RPI369_pairs.csv -sequences_dictionary_protein data/RPI369/RPI369_protein_seq.fa -sequences_dictionary_rna data/RPI369/RPI369_dna_seq.fa -output test_rna_369
    
        #python BioPrediction.py -input_interactions_train data/RPI1807/RPI1807_pairs.csv -sequences_dictionary_protein data/RPI1807/RPI1807_protein_seq.fa -sequences_dictionary_rna data/RPI1807/RPI1807_dna_seq.fa -output test_rna_RPI1807_final
    
    #python BioPrediction.py -input_interactions_train data/NPInter/NPInter_pairs.csv -sequences_dictionary_protein data/NPInter/NPInter_protein_seq.fa -sequences_dictionary_rna data/NPInter/NPInter_dna_seq.fa -output test_rna_NPI
    
    
    #python BioPrediction.py -input_interactions_train data/RPI1807/RPI1807_pairs.csv -sequences_dictionary_protein data/RPI1807/RPI1807_protein_seq.fa -sequences_dictionary_rna data/RPI1807/RPI1807_dna_seq.fa -output test_rna_RPI1807_CERTO