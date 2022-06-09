import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from collections import Counter
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle



mnist = fetch_openml('mnist_784')


X, y = mnist['data'], mnist['target']
data = np.array(X, 'int16')
target = np.array(y, 'int')


list_hog = []
for feature in data:
 fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14,14),cells_per_block=(1,1))
 list_hog.append(fd)
hog_features = np.array(list_hog)


X_train, X_test, y_train, y_test = train_test_split(hog_features,target , test_size=0.25,random_state = 50)



model =  svm.SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("accuracy=%.2f%%" % (score * 100))


filename = 'finalized_SVM_model.pkl'
pickle.dump(model, open(filename, 'wb'))