from itertools import groupby

import cv2
import numpy as np
#Load the training image
img = cv2.imread("a.png")
#Convert this Image in gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train data and test data.
train = x[:,:50].reshape(-1,400).astype(np.float32)
print(train.shape)

test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]

test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=5
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
knn.save('knn.yml')