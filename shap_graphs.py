import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import networkx as nx
import torch
import statistics
from Bio import SeqIO
import argparse
import subprocess
import sys
import os.path
from subprocess import Popen
from multiprocessing import Manager
from sklearn.exceptions import FitFailedWarning
import shap
from numpy.random import default_rng



def type_model(explainer, model, data, labels):
    """
    This function checks the type of the 'model' and modifies the 'shap' structure accordingly in the next function.
    
    Parameters:
    - explainer: A SHAP explainer object used to explain the model's predictions.
    - model: The machine learning model for which explanations are generated.
    - data: The input data used for generating explanations.
    - labels: The labels or target values associated with the data.

    Returns:
    - shap_values: The modified SHAP values based on the model type.
    - labels: The transformed labels (only if the model is an XGBoost classifier).
    """
    
    # Generate SHAP values for the given data using the provided explainer.
    shap_values = explainer(data)
    
    # Define string representations of expected model types.
    xgbtype = "<class 'xgboost.sklearn.XGBClassifier'>"
    cattype = "<class 'catboost.core.CatBoostClassifier'>"
    lgbmtype = "<class 'lightgbm.sklearn.LGBMClassifier'>"
    randtype = "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"
    decitype = "<class 'sklearn.tree._classes.DecisionTreeClassifier'>"

    # Check if the 'model' type matches one of the expected types, otherwise raise an error.
    assert lgbmtype == str(type(model)) or randtype == str(type(model)) or xgbtype == str(type(model)) \
            or cattype == str(type(model)) or decitype == str(type(model)), f"Error: Model type not as expected {str(type(model))}"

    # If the model type is LGBM or RandomForest, modify SHAP values to only use the first dimension.
    if lgbmtype == str(type(model)) or randtype == str(type(model)) or decitype == str(type(model)):
        shap_values = shap_values[:, :, 0]

    # If the model type is XGBoost, transform the labels using label encoding.
    #if xgbtype == str(type(model)):
        #labels = le.fit_transform(labels)

    return shap_values, labels


def shap_waterf(explainer, model, X_test, X_label, path):
    """
    This function generates two waterfall graphs for each class in the problem.
    
    Parameters:
    - explainer: A SHAP explainer object used to explain the model's predictions.
    - model: The machine learning model for which explanations are generated.
    - X_test: The test data for which explanations are generated.
    - X_label: The labels or target values associated with the test data.
    - path: The directory path where the generated waterfall graphs will be saved.

    Returns:
    - graphs_path: A list of file paths to the generated waterfall graph images.
    """

    # Initialize an empty list to store file paths of generated waterfall graphs.
    graphs_path = []

    # Create a DataFrame containing a single column 'label' from X_label['Label_y'].
    X_label = pd.DataFrame(data={'label': X_label['Label_y']})
       

    # Get the unique classes from the 'label' column.
    classes = X_label.iloc[:, 0].unique()

    # Check if there are exactly two unique classes, otherwise raise an error.
    assert len(classes) == 2, \
        f"Error: Classes generated by the explainer of 'model' don't match the distinct \
    number of classes in 'targets'. [Explainer=2, Target={len(classes)}]"

    # Iterate over the two unique classes.
    for i in range(1):
        # Create a subset of the test data for the current class.
        subset = X_test #[X_label.label == classes[i]]

        # Generate SHAP values for the subset using the type_model function.
        shap_values, classes = type_model(explainer, model, subset, classes)

        # Choose two samples from the current class.
        #print('shape', subset.shape[0], X_label.label)
        numbers = default_rng().choice(range(1, subset.shape[0]-1), size=(6), replace=False)

        # Generate waterfall graphs for the selected samples.
        for j in numbers:
            waterfall_name = 'class_' + str(X_label.label.iloc[j]) + '_sample_' + str(j)
            local_name = os.path.join(path, f"{waterfall_name}.png")

            # Set the title for the waterfall graph.
            plt.title(waterfall_name, fontsize=16)

            # Generate the waterfall plot and save it as an image.
            sp = shap.plots.waterfall(shap_values[j], show=False)
            plt.savefig(local_name, dpi=300, bbox_inches='tight')

            # Close the plot to release resources.
            plt.close()

            # Append the file path of the generated waterfall graph to the list.
            graphs_path.append(local_name)

    # Return the list of graph file paths.
    return graphs_path


def shap_bar(shap_values, path, fig_name):
    """
    Generate and save a SHAP bar plot.

    Parameters:
    - shap_values: The SHAP values to be visualized in the bar plot.
    - path: The directory path where the generated bar plot image will be saved.
    - fig_name: The name of the generated bar plot image.

    Returns:
    - local_name: The file path to the saved bar plot image.
    """
    local_name = os.path.join(path, f"{fig_name}.png")
    plt.title(fig_name, fontsize=16)
    sp = shap.plots.bar(shap_values, show=False)
    plt.savefig(local_name, dpi=300, bbox_inches='tight')
    plt.close()
    return local_name


def shap_beeswarm(shap_values, path, fig_name):
    """
    Generate and save a SHAP beeswarm plot.

    Parameters:
    - shap_values: The SHAP values to be visualized in the beeswarm plot.
    - path: The directory path where the generated beeswarm plot image will be saved.
    - fig_name: The name of the generated beeswarm plot image.

    Returns:
    - local_name: The file path to the saved beeswarm plot image.
    """
    local_name = os.path.join(path, f"{fig_name}.png")
    plt.title(fig_name, fontsize=16)
    sp = shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(local_name, dpi=300, bbox_inches='tight')
    plt.close()
    return local_name
