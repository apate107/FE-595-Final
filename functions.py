import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image, HTML, display
import mpld3
import matplotlib.pyplot as plt



def processInitialFile(fname, header, filesep):

    df = pd.DataFrame()
    if filesep == 'TXT' and header=='No':
        df = pd.read_csv(fname, sep='\t', header=None)
    elif filesep == 'CSV' and header=='No':
        df = pd.read_csv(fname, header=None)
    elif filesep == 'Excel' and header=='No':
        df = pd.read_excel(fname, header=None)
    elif filesep == 'TXT' and header=='Yes':
        df = pd.read_csv(fname, sep='\t', header=0)
    elif filesep == 'CSV' and header=='Yes':
        df = pd.read_csv(fname, header=0)
    elif filesep == 'Excel' and header=='Yes':
        df = pd.read_excel(fname, header=0)

    return df


def get_tree_plot(x,y, pred_type, depth):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    if pred_type == 'Classification':
        dtree = DecisionTreeClassifier(random_state=0,max_depth=depth)
    else:
        dtree = DecisionTreeRegressor(random_state=0,max_depth=depth)

    dtree.fit(x_train, y_train)

    pred = dtree.predict(x_test)

    fig,ax = plt.subplots()
    ax.plot(x_test.index, pred, label='Prediction', c='blue')
    ax.plot(x_test.index, y_test, label ='Actual', c='black')
    mpld3.save_html(fig,'templates/tree_fig.html')

