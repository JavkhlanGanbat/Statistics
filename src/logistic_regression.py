import numpy as np
from scipy.sparse import issparse

class LogisticRegressionCustom:
    """
    Custom Logistic Regression implementation using gradient descent.
    Shows the core mathematical concepts: sigmoid, log loss, and gradient updates.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, verbose=True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
        Maps any value to (0, 1) for probability interpretation
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss (log loss):
        L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        This measures how well our predictions match the true labels
        """
        m = len(y_true)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -1/m * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute gradients of loss with respect to weights and bias:
        ∂L/∂w = 1/m * X^T * (ŷ - y)
        ∂L/∂b = 1/m * Σ(ŷ - y)
        """
        m = len(y_true)
        error = y_pred - y_true
        
        # Handle sparse matrices
        if issparse(X):
            dw = (1/m) * X.T.dot(error)
            dw = np.asarray(dw).flatten()
        else:
            dw = (1/m) * np.dot(X.T, error)
        
        db = (1/m) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Algorithm:
        1. Initialize weights randomly
        2. For each iteration:
           a. Compute predictions: ŷ = σ(Xw + b)
           b. Compute loss
           c. Compute gradients
           d. Update weights: w = w - α * ∂L/∂w
           e. Update bias: b = b - α * ∂L/∂b
        3. Stop when converged or max iterations reached
        """
        # Convert sparse matrix to dense if needed for shape
        if issparse(X):
            m, n = X.shape
            X_array = X  # Keep as sparse for operations
        else:
            X = np.array(X)
            m, n = X.shape
            X_array = X
        
        y = np.array(y).reshape(-1, 1)
        
        # Initialize weights and bias
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        # Gradient descent
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Forward pass: compute predictions
            # z = Xw + b
            if issparse(X_array):
                z = X_array.dot(self.weights) + self.bias
                z = np.asarray(z).flatten().reshape(-1, 1)
            else:
                z = np.dot(X_array, self.weights) + self.bias
            
            # ŷ = σ(z)
            y_pred = self.sigmoid(z)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Compute gradients
            dw, db = self.compute_gradients(X_array, y, y_pred)
            
            # Update parameters (gradient descent step)
            self.weights -= self.learning_rate * dw.reshape(-1, 1)
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_loss = loss
            
            # Print progress
            if self.verbose and (iteration % 100 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability estimates.
        Returns probabilities for both classes [P(y=0), P(y=1)]
        """
        # Handle sparse matrices
        if issparse(X):
            z = X.dot(self.weights) + self.bias
            z = np.asarray(z).flatten()
        else:
            X = np.array(X)
            z = np.dot(X, self.weights) + self.bias
            z = z.flatten()
        
        prob_class_1 = self.sigmoid(z)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X):
        """
        Predict binary class labels (0 or 1).
        Uses 0.5 as decision threshold
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
