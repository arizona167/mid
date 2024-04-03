#import breast cancer winsconsin, perform 3 fold cross validation which s and implement 80-20 train/test split
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load the breast cancer Wisconsin dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Create an SVM model
model = SVC(kernel='linear')  

# Perform 3-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=3)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Print the mean cross-validation score
print("Mean cross-validation score:", np.mean(cv_scores))
