
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mpld3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def processInitialFile(fname, header, filesep):
    df = pd.DataFrame()
    if filesep == 'TXT' and header == 'No':
        df = pd.read_csv(fname, sep='\t', header=None)
    elif filesep == 'CSV' and header == 'No':
        df = pd.read_csv(fname, header=None)
    elif filesep == 'Excel' and header == 'No':
        df = pd.read_excel(fname, header=None)
    elif filesep == 'TXT' and header == 'Yes':
        df = pd.read_csv(fname, sep='\t', header=0)
    elif filesep == 'CSV' and header == 'Yes':
        df = pd.read_csv(fname, header=0)
    elif filesep == 'Excel' and header == 'Yes':
        df = pd.read_excel(fname, header=0)
    return df


def get_knn_plot(x, y, k):
    # Split data and fit model
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Create plot and save it
    fig, ax = plt.subplots()
    for i in range(len(set(pred))):
        val = list(set(pred))[i]
        ax.scatter(X_test.iloc[pred == val, 0], X_test.iloc[pred == val, 1],
                   c=list(np.random.choice(range(256), size=3)/255), label=val)
    plt.xlabel(X_test.columns[0])
    plt.ylabel(X_test.columns[1])
    axhandles, axlabels = ax.get_legend_handles_labels()
    mpld3.plugins.connect(fig, mpld3.plugins.InteractiveLegendPlugin(axhandles, axlabels))
    fig.subplots_adjust(right=0.7)
    mpld3.save_html(fig, 'templates/knn_fig.html')

    return {'accuracy': model.score(X_test, y_test)}


def get_tree_plot(x, y, pred_type, depth):
    # Split data and fit model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    if pred_type == 'Classification':
        dtree = DecisionTreeClassifier(random_state=0, max_depth=depth)
    else:
        dtree = DecisionTreeRegressor(random_state=0, max_depth=depth)
    dtree.fit(x_train, y_train)
    pred = dtree.predict(x_test)

    # Create plot and save
    fig, ax = plt.subplots()
    ax.plot(x_test.index, pred, label='Prediction', c='blue')
    ax.plot(x_test.index, y_test, label='Actual', c='black')
    axhandles, axlabels = ax.get_legend_handles_labels()
    mpld3.plugins.connect(fig, mpld3.plugins.InteractiveLegendPlugin(axhandles, axlabels))
    fig.subplots_adjust(right=0.7)
    mpld3.save_html(fig, 'templates/tree_fig.html')

