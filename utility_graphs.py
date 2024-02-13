import logging
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from itertools import zip_longest
from os.path import join, basename, exists
from sys import stdout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import networkx as nx
import statistics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    auc, accuracy_score, balanced_accuracy_score, roc_auc_score
)

report_handler = logging.StreamHandler(stream=stdout)
report_handler.setLevel(logging.WARNING)
report_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

report_logger = logging.getLogger(basename(__file__))
report_logger.addHandler(report_handler)



REPORT_USABILITY_TITLE_BINARY = "Model Interpretability Report (BioPrediction)"
REPORT_USABILITY_DESCRIPITION = "This a usability report for explain some models aspects that can be util for your aplication."

REPORT_1 = """
Type one ...
"""

REPORT_2 = """
Type two ...
"""

REPORT_3 = """
Type tree ...
"""


def final_predictions(X_test, y_test, y_pares, model, output):  
    """
    Generate final predictions using a machine learning model, append them to a DataFrame, and save the results to a CSV file.

    Parameters:
    - X_test: The test data for which predictions are generated.
    - y_test: The true labels or target values associated with the test data.
    - y_pares: A DataFrame with additional information (e.g., identifiers) related to the predictions.
    - model: The machine learning model used for making predictions.
    - output: The directory where the CSV file with final predictions will be saved.

    Returns:
    - output_data: A DataFrame containing the final predictions and related information.
    """
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)

    output_data = y_pares
    output_data['Class_pred'] = y_pred
    output_data['Class_real'] = y_test
    output_data[['Probability of class 0', 'Probability of class 1']] = pd.DataFrame(y_proba)
    output_data = output_data.drop('Probability of class 0', axis=1)
    output_data = output_data.sort_values(by='Probability of class 1', ascending=False)
    output_data.to_csv(output+'/list_final_predictions.csv', index=False)
    return output_data
    
def precision_graph(output_data, output):
    """
    Generate and save a precision vs. threshold graph using filtered data.

    Parameters:
    - output_data: A DataFrame containing the final predictions and related information.
    - output: The directory where the graph image will be saved.

    Returns:
    - output: The file path to the saved precision vs. threshold graph image.
    """
    generated_plt = []
 
    # Set the initial threshold value for filtering data.
    threshold = 0.5
    filtered_data = output_data[output_data['Probability of class 1'] > threshold]
    y_true_filtered = filtered_data['Class_real']
    y_pred_filtered = filtered_data['Class_pred']
    filtered_data.sort_values(by='Probability of class 1', ascending=False, inplace=True)

    # Define a range of thresholds for analysis.
    thresholds = np.arange(0.99, 0.45, -0.05)
    accuracies = []

    for threshold in thresholds:
        threshold_data = filtered_data[filtered_data['Probability of class 1'] >= threshold]
        accuracy = precision_score(threshold_data['Class_real'], threshold_data['Class_pred'])
        accuracies.append(accuracy)
    thresholds = thresholds[::-1]
    accuracies = accuracies[::-1]

    plt.plot(thresholds * 100, accuracies, marker='o', linestyle='-')
    plt.title('Precision vs. Threshold (Threshold > 50%)')
    plt.xlabel('Threshold (%)')
    plt.ylabel('Precision')
    plt.gca().invert_xaxis()  # Invert the x-axis for increasing thresholds.
    plt.grid(True)

    output = output + '/graph_precision.png'
    plt.savefig(output, format='png')  # dpi=300, bbox_inches='tight')
    plt.clf()
    
    # Return the file path to the saved precision vs. threshold graph.
    return output


def coverage_graph(output_data, output):
    """
    Generate and save a coverage vs. percentage of the dataset graph.

    Parameters:
    - output_data: A DataFrame containing the final predictions and related information.
    - output: The directory where the graph image will be saved.

    Returns:
    - output: The file path to the saved coverage vs. percentage of the dataset graph.
    """

    y_true = output_data['Class_real']
    y_pred = output_data['Class_pred']
    y_score = output_data['Probability of class 1']

    max_radius = y_true.value_counts()[1]
    total_size = y_true.shape[0]
    predicted_values = y_score.unique()
    score_list = y_score.values.tolist()
    points = []
    
    for i in range(1, len(predicted_values)):
        index = score_list.index(predicted_values[i])
        points.append(len(score_list[:index]))

    division = 0
    limit = 0.5

    # Find the division point where predicted scores are less than 0.5.
    for i in range(1, len(predicted_values)):
        if predicted_values[i] < 0.5:
            limit = predicted_values[i]
            division = score_list.index(predicted_values[i])
            break
    true_list = y_true.values.tolist()
    radius = []

    # Calculate coverage percentage for each point.
    for i in range(len(points)):
        radius.append(sum(true_list[:points[i]]) / max_radius)
        if radius[len(radius) - 1] > 0.985:
            break
            
    total_samples = len(y_true)
    x = points[:len(radius)]
    x = [element / total_samples for element in x]
    y = radius

    # Plot the coverage vs. percentage of the dataset graph.
    plt.plot(x, y, label='')
    plt.axvline(x=division / total_samples, color='red', label='Conventional division')
    plt.xlabel('Percentage of the dataset')
    plt.ylabel('Coverage percentage')
    plt.title('Coverage of true positives on the test data')
    plt.tick_params(axis='x', which='major', direction='inout', length=10, width=1, colors='black')
    plt.legend()
    output = output + '/graph_coverage.png'
    plt.savefig(output, format='png')

    # Return the file path to the saved coverage vs. percentage of the dataset graph.
    return output


def make_fig_graph(output_data, output):
    """
    Create and save graphs based on node connections using NetworkX.

    Parameters:
    - output_data: A DataFrame containing information, including 'nameseq', 'Class_pred', etc.
    - output: The directory where the generated graphs will be saved.

    Returns:
    - None
    """
    # Extract 'nameseq' values for sequences predicted as class 1.
    names = output_data.loc[output_data['Class_pred'] == 1, 'nameseq'].tolist()

    aminoA, aminoB = [], []
    for i in names:
        nu, am = i.split("_", 1)
        aminoA.append(nu)
        aminoB.append(am)

    # Create an undirected graph using NetworkX.
    G = nx.Graph()

    # Sort the lists by length to ensure consistent node order.
    list1 = sorted(aminoA, key=len)
    list2 = sorted(aminoB, key=len)

    # Add nodes to the graph.
    G.add_nodes_from(list1)
    G.add_nodes_from(list2)

    # Add edges to connect nodes.
    for i in range(len(list1)):
        G.add_edge(list1[i], list2[i])

    # Calculate degrees for each node.
    d = dict(G.degree())

    # Define threshold degrees for selecting nodes.
    threshold_degrees = [0.8, 0.15, 0.05]
    deg = []

    for k in threshold_degrees:
        sorted_degrees = sorted(d.values(), reverse=True)
        num_top_nodes = int(len(sorted_degrees) * k)
        grau_10_percent = sorted_degrees[num_top_nodes - 1]
        deg.append(grau_10_percent)

    for j in range(3):
        d = dict(G.degree())

        # Filter nodes based on degree threshold.
        filtered_nodes = [node for node, degree in d.items() if degree >= deg[j]]
        G_ = G.subgraph(filtered_nodes)
        d = dict(G_.degree())

        # Create a graph visualization.
        pos = nx.spring_layout(G_)
        fig, ax = plt.subplots(figsize=(36 * 3, 30 * 3))
        nx.draw(G_, node_size=[v * 150 for v in d.values()], pos=pos, node_color='skyblue', ax=ax)
        nx.draw_networkx_edges(G_, pos, width=2, edge_color='gray', ax=ax)
        labels = {node: node for node in G_.nodes()}
        nx.draw_networkx_labels(G_, pos, labels, font_size=12, font_color='black', font_family='sans-serif', ax=ax)
        ax.set_axis_off()
        plt.tight_layout()

        # Save the graph image.
        output_name = output + '/graph_degree_' + str(deg[j]) + 'or_more.png'
        plt.savefig(output_name, format='png')
        plt.clf()