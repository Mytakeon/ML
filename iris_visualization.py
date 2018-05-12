"""
Different methods to look at the iris dataset. Uses pandas and seaborn for data visualization, and sklearn for PCA and
Decision tree.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas
import pydotplus
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix, parallel_coordinates
from sklearn import model_selection, tree
from sklearn.decomposition import PCA

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pandas.read_csv(url, names=names)
y = np.array(data['class'])  # Label/output: type of flower
x = np.array(data.drop(['class'], 1))  # All the features


def info():
    print(data.head())
    print(data.corr())


def scatter_plot_panda():
    data.hist()
    scatter_matrix(data, color={'Iris-setosa': '#377eb8',
                                'Iris-versicolor': '#4eae4b',
                                'Iris-virginica': '#e41a1c'})
    plt.show()


def scatter_plot_seaborn():
    sns.set(style="ticks")
    sns.pairplot(data, hue='class')
    plt.show()


def parallel_coord():
    parallel_coordinates(data, 'class', color=['#377eb8', '#4eae4b', '#e41a1c'])
    plt.show()


def decision_tree():
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
    clf.fit(x_train, y_train)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                    class_names=['Setosa', 'Versicolor', 'Virginica'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("irisColoured.pdf")


def principal_component_analysis():
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions

    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(x)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


if __name__ == '__main__':
    principal_component_analysis()
