import numpy as np

# Assuming you have a dataset in the form of numpy arrays X1, X2, and Y
# Make sure to normalize/standardize your features if needed

# Add a column of ones to X for θ0
X = np.column_stack((np.ones_like(X1), X1, X2))
Y = Y.reshape(-1, 1)  # Reshape Y to be a column vector

# Initialize parameters
theta = np.zeros((3, 1))
alpha = 0.01  # Learning rate
iterations = 1000

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Perform gradient descent
for _ in range(iterations):
    # Calculate predictions
    predictions = sigmoid(np.dot(X, theta))

    # Calculate errors
    errors = predictions - Y

    # Update parameters using gradient descent
    gradient = np.dot(X.T, errors) / len(Y)
    theta -= alpha * gradient

# Extract coefficients
theta0, theta1, theta2 = theta.flatten()

print("Theta0 (Intercept):", theta0)
print("Theta1 (Coefficient for X1):", theta1)
print("Theta2 (Coefficient for X2):", theta2)
