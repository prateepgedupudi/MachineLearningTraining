from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_distance = euc(row,X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            distance = euc(row, self.X_train[i])
            if distance < best_distance:
                best_distance = distance
                best_index = i

        return self.y_train[best_index]



# import a data set
from sklearn import datasets

iris = datasets.load_iris()
# making data between features and labels
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

# splitting half of data between training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# from sklearn.neighbors import KNeighborsClassifier

# loading DecisionTreeClassifier
my_clasifier = ScrappyKNN()
my_clasifier.fit(X_train, y_train)

my_predictions = my_clasifier.predict(X_test)

print(my_predictions)

from sklearn.metrics import accuracy_score

# testing accuracy of the algorithm  by comparing actual labels and prediction labels
print(accuracy_score(y_test, my_predictions))
