
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# Load the breast cancer Wisconsin dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Create an SVM model
model = SVC(kernel='linear', probability=True)  # Set probability to True for ROC curve

# Perform 3-fold cross-validation
predicted_labels_cv = cross_val_predict(model, X, y, cv=3, method='predict_proba')
cv_fpr, cv_tpr, _ = roc_curve(y, predicted_labels_cv[:, 1])
cv_auc = auc(cv_fpr, cv_tpr)

# 80-20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the test set
predicted_labels_test = model.predict_proba(X_test)
test_fpr, test_tpr, _ = roc_curve(y_test, predicted_labels_test[:, 1])
test_auc = auc(test_fpr, test_tpr)

# Plot ROC curve
plt.figure(figsize=(5, 5))
plt.plot(cv_fpr, cv_tpr, label=f'Cross-Validated ROC')
plt.plot(test_fpr, test_tpr, label=f'Test Set ROC')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
