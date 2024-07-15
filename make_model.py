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
from sklearn.metrics import  matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

def tune_model(model, X_train, y_train, param_grid, cv=5):
    """
    Tune a machine learning model using Grid Search Cross-Validation.

    Parameters:
        model: An instance of a scikit-learn model.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        param_grid (dict): Dictionary of hyperparameter values to search.
        cv (int): Number of cross-validation folds.

    Returns:
        best_params (dict): Best hyperparameters found during tuning.
    """
    # Define the F1 scorer for class 1
    f1_scorer = make_scorer(f1_score, pos_label=1)

    # Create Grid Search Cross-Validation object with F1 scoring
    #grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1)    
    grid_search = RandomizedSearchCV(
        model, param_grid, cv=cv, scoring=f1_scorer, n_iter=75, n_jobs=-1, random_state=42
    )

    # Fit the model to the data
    best_model = grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    #best_params = grid_search.best_params_

    #print("Best Hyperparameters:")
    #print(best_params)

    return best_model.best_estimator_





def metrics(preds, test_labels, preds_proba, output):
    """
    Compute various classification metrics and save them to a CSV file.

    Parameters:
        preds (array-like): Predicted labels by the model.
        test_labels (array-like): True labels from the test dataset.
        output (str): File path where the metrics will be saved.

    Returns:
        None
    """
    confusion = confusion_matrix(test_labels, preds)
    true_negatives, false_positives, true_positives, false_negatives = confusion.ravel()
    

    # Calculate precision, recall, and F1-score for class 1
    precision_1 = precision_score(test_labels, preds, average='binary', pos_label=1)
    recall_1 = recall_score(test_labels, preds, average='binary', pos_label=1)
    f1_1 = f1_score(test_labels, preds, average='binary', pos_label=1)
    specificity = true_negatives / (true_negatives + false_positives)

    # Calculate precision, recall, and F1-score for class 0
    precision_0 = precision_score(test_labels, preds, average='binary', pos_label=0)
    recall_0 = recall_score(test_labels, preds, average='binary', pos_label=0)
    f1_0 = f1_score(test_labels, preds, average='binary', pos_label=0)


    # Calculate accuracy
    accuracy = accuracy_score(test_labels, preds)

    # Calculate precision and recall for various thresholds (used for AUPR calculation)
    precision, recall, _ = precision_recall_curve(test_labels, preds)

    auc_roc = roc_auc_score(test_labels, preds_proba)
    
    # Calculate AUPR (Area Under Precision-Recall Curve)
    aupr = auc(recall, precision)

    # Calculate balanced accuracy
    balanced = balanced_accuracy_score(test_labels, preds)

    # Calculate MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(test_labels, preds)

    #Create a dictionary to store the computed metrics
    metrics_dict = {
        'Metrics': ['Precision_1', 'Recall_1', 'F1_1', 'Specificity_1', 'Precision_0',
                    'Recall_0', 'F1_0', 'Accuracy', 'AUC', 'AUPR', 
                      'Balanced_Accuracy', 'MCC'],
        'Values': [precision_1, recall_1, f1_1, specificity, precision_0, recall_0,
                   f1_0, accuracy, auc_roc, aupr, balanced, mcc]
    }

    #metrics_dict = {
    #    'Metrics': ['Precision_1', 'Recall_1', 'F1_1', 'Precision_0',
    #                'Recall_0', 'F1_0', 'Accuracy', 'AUPR', 'Balanced_Accuracy', 'MCC'],
    #    'Values': [precision_1, recall_1, f1_1, precision_0, recall_0,
    #               f1_0, accuracy, aupr, balanced, mcc]
    #}
    
    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    
    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(output, index=False)


def better_model(X_train, y_train, X_test, y_test, output, tuning=False, metric=f1_score):
    """
    Train multiple classification models, select the best-performing model based on a specified metric,
    and save it to a file.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        output (str): File path where the best-performing model will be saved.
        tuning (bool, optional): If True, hyperparameter tuning will be performed for the best model.
        metric (function, optional): The evaluation metric to use for model selection.
            Default is balanced_accuracy_score.

    Returns:
        best_model: The best-performing classifier.
    """
    valor_teto = 1e9  # Substitua pelo valor desejado
    print(X_train)
    # Substitua valores infinitos ou muito grandes pelo valor teto
    X_train.replace([np.inf, -np.inf, np.nan], valor_teto, inplace=True)

    # Converta os valores para 'float32'
    X_train = X_train.astype('float32')
    
    
    # Lists to store performance scores and model instances
    perf = []
    model = []
    
    #X_train = X_train.drop_duplicates()
    #y_train = y_train.drop_duplicates()
    
    
    # Train and evaluate a RandomForestClassifier
    clf = RandomForestClassifier(random_state=43)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    perf.append(metric(y_test, preds))
    model.append(RandomForestClassifier(random_state=43))
        
    # Train and evaluate a CatBoostClassifier
    clf = CatBoostClassifier(random_state=43, verbose=False)
    clf.fit(X_train.values, y_train)
    preds = clf.predict(X_test.values)
    perf.append(metric(y_test, preds))
    model.append(CatBoostClassifier(random_state=43, verbose=False))
                
    # Train and evaluate an XGBClassifier
    clf = xgb.XGBClassifier(random_state=43)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    perf.append(metric(y_test, preds))
    model.append(xgb.XGBClassifier(random_state=43))
    
    # Train and evaluate a DecisionTreeClassifier
    
    clf = DecisionTreeClassifier(random_state=43)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    perf.append(metric(y_test, preds))
    model.append(DecisionTreeClassifier(random_state=43))
    


    # Select the best-performing model based on the specified metric
    best_model = model[perf.index(max(perf))]
    
    
    
    
    # If tuning is True, perform hyperparameter tuning for the best model
    if tuning:
        dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 2, 4, 6, 8],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [5, 10, 20],
        'max_features': [None, 'sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }
        
        rf_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 2, 5, 8, 10],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [5, 10, 20],
        #'learning_rate': [0.01, 0.05, 0.1]
    }
        catboost_param_grid = {
        #'iterations': [100, 200, 300],  # Número de iterações
        'learning_rate': [0.01, 0.05, 0.1],  # Taxa de aprendizado
        'depth': [None, 1, 2, 4, 6, 8],  # Profundidade da árvore
        'l2_leaf_reg': [1, 3, 5],  # Regularização L2
        #'bagging_temperature': [0.5, 1.0, 1.5],  # Temperatura de bagging
        #'border_count': [32, 64, 128],  # Número de pontos de divisão na borda
        'scale_pos_weight': [1, 2, 3],  # Peso positivo para balanceamento de classes
        'eval_metric': ['F1'],  # Métrica de avaliação
    }        
        xgb_param_grid = {
        #'criterion': ['gini', 'entropy'],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [None, 1, 2, 4, 6, 8],
        'min_child_weight': [1, 2, 3],
        #'colsample_bytree': [0.8, 0.9, 1.0],
        #'gamma': [0, 0.1, 0.2],
        #'min_samples_split': [5, 10, 20],  # Adding min_samples_split here
        #'min_samples_leaf': [5, 10, 20], 
        #'reg_alpha': [0, 0.1, 0.3, 0.5],
        #'reg_lambda': [0, 0.1, 0.5],
        'scale_pos_weight': [1, 2, 3]
        #'eval_metric': ['logloss'],
    }
        
        if perf.index(max(perf)) == 0:
            best_model = tune_model(best_model, X_train, y_train, rf_param_grid)
        elif  perf.index(max(perf)) == 1:
            best_model = tune_model(best_model, X_train, y_train, catboost_param_grid)       
        elif  perf.index(max(perf)) == 2:
            best_model = tune_model(best_model, X_train, y_train, xgb_param_grid)
        elif  perf.index(max(perf)) == 3:
            best_model = tune_model(best_model, X_train, y_train, dt_param_grid)

    else:
        best_model.fit(X_train.values, y_train)    
        
    # Save the best-performing model to a file
    joblib.dump(best_model, output)

    return best_model


def Score_table(test_data, predictions, scores, scores_name, output):
    """
    Create a score table with sequence names, predicted labels, and scores, and save it to a CSV file.

    Parameters:
        test_data (DataFrame): DataFrame containing test data, including sequence names.
        predictions (array-like): Predicted labels.
        scores (array-like): Scores associated with predictions.
        scores_name (str): Name of the score column.
        output (str): File path where the score table will be saved.

    Returns:
        model_result (DataFrame): The score table DataFrame.
    """
    columns = test_data.columns
    model_result = test_data.copy()

    # Create a 'nameseq' column by concatenating the first and second columns as strings
    model_result['nameseq'] = model_result[columns[0]].astype(str) + '_' + model_result[columns[1]].astype(str)
    model_result = model_result.drop(columns=[columns[0], columns[1]])
    #print(columns[0], columns[1], scores_name, scores)
    # Add columns for predicted labels and the specified scores
    #model_result['PredictedLabel'] = predictions
    model_result[scores_name] = scores
    #print(model_result.columns)
    # Reorder the columns in the desired order

    #print(columns, 'the len', len(columns))
    if 'Label' in test_data.columns:
        #print(columns, 'the len', len(columns))
        model_result = model_result[['nameseq', scores_name, columns[2]]]
    else:
        model_result = model_result[['nameseq', scores_name]]
    model_result = model_result.sort_index()

    # Save the score table DataFrame to a CSV file
    model_result.to_csv(output, index=False)

    return model_result


def objective_rf(space):
    """Tuning of classifier: Objective Function - Random Forest - Bayesian Optimization"""
    n_cpu = 1
    model = RandomForestClassifier(
        criterion=space['criterion'],
        max_depth=int(space['max_depth']), 
        max_features=space['max_features'],
        random_state=63,
        bootstrap=space['bootstrap'],
        n_jobs=n_cpu)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FitFailedWarning)
            balanced_accuracy = cross_val_score(
                model,
                X_train,  
                y_train,  
                cv=kfold,
                scoring=make_scorer(balanced_accuracy_score),
                n_jobs=n_cpu).mean()
    except Exception as e:
        print(f"An error occurred during cross-validation: {str(e)}")
        balanced_accuracy = None

    if balanced_accuracy is None:
        return {'loss': 1.0, 'status': STATUS_FAIL} 
    else:
        return {'loss': -balanced_accuracy, 'status': STATUS_OK}



def tuning_rf_bayesian():

    """Tuning of classifier: Random Forest - Bayesian Optimization"""
    n_cpu = 1
    param = {'criterion': ['entropy', 'gini'], 'max_depth': ['max_depth',1, 10, 2],
             'max_features': ['auto', 'sqrt', 'log2', None], 'min_samples_leaf': ['min_samples_leaf', 1, 10, 2],
             'min_samples_split': ['min_samples_split', 2, 10, 2],
             'bootstrap': [True, False]}
    
    
    space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
             'max_depth': hp.quniform('max_depth', 1, 10, 2),
             'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
             'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 2),
             'min_samples_split': hp.quniform('min_samples_split', 2, 10, 2),
             'bootstrap': hp.choice('bootstrap', [True, False])}

    trials = Trials()
    best_tuning = fmin(fn=objective_rf,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    best_rf = RandomForestClassifier(
                                     criterion=param['criterion'][best_tuning['criterion']],
                                     max_depth=int(best_tuning['max_depth']),
                                     max_features=param['max_features'][best_tuning['max_features']],
                                     min_samples_leaf=int(best_tuning['min_samples_leaf']),
                                     min_samples_split=int(best_tuning['min_samples_split']),
                                     random_state=63,
                                     bootstrap=param['bootstrap'][best_tuning['bootstrap']],
                                     n_jobs=n_cpu)
    return best_tuning, best_rf


def objective_cb(space):

    """Tuning of classifier: Objective Function - CatBoost - Bayesian Optimization"""

    model = CatBoostClassifier(
                               max_depth=int(space['max_depth']),
                               learning_rate=space['learning_rate'],
                               thread_count=n_cpu, nan_mode='Max', logging_level='Silent',
                               random_state=63)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FitFailedWarning)
            balanced_accuracy = cross_val_score(
                model,
                X_train,  
                y_train,  
                cv=kfold,
                scoring=make_scorer(balanced_accuracy_score),
                n_jobs=n_cpu).mean()
    except Exception as e:
        print(f"An error occurred during cross-validation: {str(e)}")
        balanced_accuracy = None

    if balanced_accuracy is None:
        return {'loss': 1.0, 'status': STATUS_FAIL}  # Indicar que houve uma falha
    else:
        return {'loss': -balanced_accuracy, 'status': STATUS_OK}


def tuning_catboost_bayesian():

    """Tuning of classifier: CatBoost - Bayesian Optimization"""
    n_cpu = 1
    space = {
             'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
             'max_depth': hp.quniform('max_depth', 1, 5, 1),
             #  'random_strength': hp.loguniform('random_strength', 1e-9, 10),
             #  'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 1.0),
             #  'border_count': hp.quniform('border_count', 1, 255, 1),
             #  'l2_leaf_reg': hp.quniform('l2_leaf_reg', 2, 30, 1),
             #  'scale_pos_weight': hp.uniform('scale_pos_weight', 0.01, 1.0),
             #  'bootstrap_type' =  hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
    }

    trials = Trials()
    best_tuning = fmin(fn=objective_cb,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=100,
                       trials=trials)

    best_cb = CatBoostClassifier(
                                 max_depth=int(best_tuning['max_depth']),
                                 learning_rate=best_tuning['learning_rate'],
                                 thread_count=n_cpu, nan_mode='Max', logging_level='Silent',
                                 random_state=63)

    return best_tuning, best_cb

"""
def objective_lightgbm(space):

    #Tuning of classifier: Objective Function - Lightgbm - Bayesian Optimization

    model = lgb.LGBMClassifier(
                               max_depth=int(space['max_depth']),
                               num_leaves=int(space['num_leaves']),
                               learning_rate=space['learning_rate'],
                               subsample=space['subsample'],
                               n_jobs=n_cpu,
                               random_state=63)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FitFailedWarning)
            balanced_accuracy = cross_val_score(
                model,
                X_train,  
                y_train,  
                cv=kfold,
                scoring=make_scorer(balanced_accuracy_score),
                n_jobs=n_cpu).mean()
    except Exception as e:
        print(f"An error occurred during cross-validation: {str(e)}")
        balanced_accuracy = None

    if balanced_accuracy is None:
        return {'loss': 1.0, 'status': STATUS_FAIL}  # Indicar que houve uma falha
    else:
        return {'loss': -balanced_accuracy, 'status': STATUS_OK}


def tuning_lightgbm_bayesian():

    #Tuning of classifier: Lightgbm - Bayesian Optimization

    space = {
             'max_depth': hp.quniform('max_depth', 1, 10, 1),
             'num_leaves': hp.quniform('num_leaves', 10, 200, 10),
             'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
             'subsample': hp.uniform('subsample', 0.1, 1.0)}

    trials = Trials()
    best_tuning = fmin(fn=objective_lightgbm,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=100,
                       trials=trials)

    best_cb = lgb.LGBMClassifier(
                                 max_depth=int(best_tuning['max_depth']),
                                 num_leaves=int(best_tuning['num_leaves']),
                                 learning_rate=best_tuning['learning_rate'],
                                 subsample=best_tuning['subsample'],
                                 n_jobs=n_cpu,
                                 random_state=63)

    return best_tuning, best_cb
"""


def objective_gb(space):
    """Tuning of classifier: Objective Function - XGBoost - Bayesian Optimization"""

    model = XGBClassifier(
        max_depth=int(space['max_depth']),
        learning_rate=space['learning_rate'],
        subsample=space['subsample'],
        n_jobs=n_cpu,
        random_state=63
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FitFailedWarning)
            balanced_accuracy = cross_val_score(
                model,
                X_train,  
                y_train,  
                cv=kfold,
                scoring=make_scorer(balanced_accuracy_score),
                n_jobs=n_cpu
            ).mean()
    except Exception as e:
        print(f"An error occurred during cross-validation: {str(e)}")
        balanced_accuracy = None

    if balanced_accuracy is None:
        return {'loss': 1.0, 'status': STATUS_FAIL}  # Indicar que houve uma falha
    else:
        return {'loss': -balanced_accuracy, 'status': STATUS_OK}

def tuning_xgb_bayesian():

    """Tuning of classifier: XGBoost - Bayesian Optimization"""

    space = {
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
        'subsample': hp.uniform('subsample', 0.1, 1.0)
    }

    trials = Trials()
    best_tuning = fmin(fn=objective_gb,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=100,
                       trials=trials)

    best_cb = objective_gb(
        max_depth=int(best_tuning['max_depth']),
        learning_rate=best_tuning['learning_rate'],
        subsample=best_tuning['subsample'],
        n_jobs=n_cpu,
        random_state=63
    )

    return best_tuning, best_cb
