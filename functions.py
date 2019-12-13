
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier,plot_tree,export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
import mpld3
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def processInitialFile(fstream, header, filesep):
    df = pd.DataFrame()
    if filesep == 'TXT' and header == 'No':
        df = pd.read_csv(fstream, sep='\t', header=None)
    elif filesep == 'CSV' and header == 'No':
        df = pd.read_csv(fstream, header=None)
    elif filesep == 'Excel' and header == 'No':
        df = pd.read_excel(fstream, header=None)
    elif filesep == 'TXT' and header == 'Yes':
        df = pd.read_csv(fstream, sep='\t', header=0)
    elif filesep == 'CSV' and header == 'Yes':
        df = pd.read_csv(fstream, header=0)
    elif filesep == 'Excel' and header == 'Yes':
        df = pd.read_excel(fstream, header=0)
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
        dtree.fit(x_train, y_train)
        is_pred = dtree.predict(x_train)
        pred = dtree.predict(x_test)
        cm = confusion_matrix(y_test,pred)
        fig,ax = plt.subplots()
        im = ax.imshow(cm,interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im,ax=ax)
        #fig.suptitle('This is suptitle')
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               title = 'Confusion Matrix', xlabel = "Predicted", ylabel='True')
        ax.set_title('Confusion Matrix',fontsize=20,pad=20,fontweight='bold')

        ax.set_ylabel('True',fontsize=15,labelpad=10)
        ax.set_xlabel('Predicted', fontsize=15, labelpad=10)
        #ax.set_yticks(np.arange(cm.shape[0]), [i for i in np.arange(cm.shape[0])])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i,format(cm[i,j],'d'), ha='center',
                        va='center', color='black',fontsize=20)
                #ax.axvline(x=i, linewidth=2,color='black', ymax=5)

        #for c in range(len(np.unique(y))):
        #    if c > 0:
                #ax.axvline(x=c, linewidth=2, color='black')
                #ax.axhline(y=c - 0.05, linewidth=2, color='black', xmin=-1, xmax=cm.shape[0])

        fig.set_size_inches(10,5)
        fig.tight_layout()
        axhandles, axlabels = ax.get_legend_handles_labels()
        mpld3.plugins.connect(fig, mpld3.plugins.InteractiveLegendPlugin(axhandles, axlabels))
        fig.subplots_adjust(right=0.7)
        mpld3.save_html(fig, 'templates/tree_fig.html')
        #axhandles, axlabels = ax.get_legend_handles_labels()
        #mpld3.plugins.connect(fig,mpld3.plugins.InteractiveLegendPlugin(axhandles,axlabels))
        #mpld3.save_html(fig,'templates/tree_fig.html')


        m1 = round(np.mean(y_train == is_pred),3)
        m2 = round(np.mean(y_test == pred),3)

    else:
        dtree = DecisionTreeRegressor(random_state=0, max_depth=depth)
        dtree.fit(x_train, y_train)
        pred = dtree.predict(x_test)
        is_pred = dtree.predict(x_train)
        yerr = pred-y_test
        m1 = round(mean_squared_error(y_train,is_pred),3)
        m2 = round(mean_squared_error(y_test,pred),3)

        # Create plot and save
        #fig, ax = plt.subplots()
        #ax.errorbar(x, y, yerr, solid_capstyle='projecting', capsize=5)
        #ax.grid(alpha=0.5, linestyle=':')
        #plt.show()



        fig, ax = plt.subplots()
        #ax.plot(x_test.index, pred, label='Prediction', c='blue')
        #ax.scatter(x_test.index, y_test, label='Actual', c='black')
        ax.errorbar(x_test.index,pred,yerr,fmt='bo',c='red',ecolor='red')
        ax.grid(alpha=0.5, linestyle=':')
        ax.set_ylabel('Values',fontdict={'fontsize':15,'fontweight':'bold'},
                      labelpad=10)
        ax.set_xlabel('Index', fontdict={'fontsize': 15, 'fontweight': 'bold'})
        ax.set_title('Plot of OOS Absolute Error',
                     fontdict={'fontsize':20,'fontweight':'bold'})
        axhandles, axlabels = ax.get_legend_handles_labels()
        mpld3.plugins.connect(fig, mpld3.plugins.InteractiveLegendPlugin(axhandles, axlabels))
        fig.subplots_adjust(right=0.7)
        mpld3.save_html(fig, 'templates/tree_fig.html')


    #export_graphviz(dtree, out_file='templates/dtree.dot',
    #                feature_names=x_train.columns.tolist())

    #os.system('dot -Tpng templates/dtree.dot -o templates/ddtree.png')



    return [m1,m2]


