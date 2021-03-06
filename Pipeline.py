# import a dataset
from sklearn import datasets
iris = datasets.load_iris()
# making data between features and labels
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
#splitting half of data between training and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .5)

from sklearn import tree
# loading DecisionTreeClassifier
my_clasifier = tree.DecisionTreeClassifier()
my_clasifier.fit(X_train,y_train)
my_predictions = my_clasifier.predict(X_test)

print(my_predictions)

from sklearn.metrics import accuracy_score
# testing accuracy of the algorithm  by comparing actual labels and prediction labels
print(accuracy_score(y_test,my_predictions))