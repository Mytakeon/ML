import pandas
from sklearn import model_selection, linear_model

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

array = data.values
# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
# scaler = sk.preprocessing.StandardScaler().fit(X) # Set mean to 0 and standard deviation of 1
# X = scaler.transform(X)

kfold = model_selection.KFold(n_splits=10, random_state=7)
model = linear_model.LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results)

# sb.pairplot(data, hue='class')
# plt.show()
