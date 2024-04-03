import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Admission.xlsx')

# Separate features (X) and target variable (Y)
X = df[['X1', 'X2']]
Y = df['Y']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, Y_train)

# Get the coefficients θ0, θ1, and θ2
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

# Display the coefficients
print(f'θ0 (intercept): {theta0}')
print(f'θ1 (coefficient for X1): {theta1}')
print(f'θ2 (coefficient for X2): {theta2}')
