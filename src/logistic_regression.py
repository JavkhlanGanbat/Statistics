"""
Custom Logistic Regression Implementation from Scratch

This module implements binary logistic regression using gradient descent,
showing all the key mathematical operations explicitly rather than hiding
them behind library calls.

Mathematical Background:
-----------------------
Logistic regression models the probability of a binary outcome (0 or 1) using:
    P(y=1|x) = σ(w^T x + b)
    
where σ is the sigmoid function and w, b are learned parameters.
"""

import numpy as np
from scipy.sparse import issparse


class LogisticRegressionCustom:
    """
    Custom Logistic Regression using Batch Gradient Descent.
    
    This implementation shows the complete learning algorithm:
    1. Forward pass: compute predictions using current weights
    2. Loss calculation: measure prediction error
    3. Backward pass: compute gradients
    4. Parameter update: adjust weights to reduce loss
    
    Parameters:
    -----------
    learning_rate : float
        Step size for gradient descent. Controls how much we adjust weights
        in each iteration. Too large = unstable, too small = slow convergence.
        
    max_iter : int
        Maximum number of training iterations. We stop if we hit this limit
        even if the model hasn't fully converged.
        
    tol : float
        Convergence tolerance. If loss changes by less than this between
        iterations, we consider the model converged.
        
    verbose : bool
        Whether to print training progress to console.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, verbose=True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.weights = None  # Will be initialized to zeros during fit()
        self.bias = None     # Intercept term
        self.losses = []     # Track loss at each iteration for convergence analysis
    
    def sigmoid(self, z):
        """
        Sigmoid (Logistic) Function: σ(z) = 1 / (1 + e^(-z))
        
        Maps any real number to (0, 1), making it perfect for probabilities.
        
        Properties:
        - σ(0) = 0.5 (decision boundary)
        - σ(+∞) → 1
        - σ(-∞) → 0
        - Smooth and differentiable everywhere
        
        Parameters:
        -----------
        z : array-like
            Linear combination of features: z = w^T x + b
            
        Returns:
        --------
        Probability estimates between 0 and 1
        
        Note: We clip z to prevent numerical overflow with large values.
        """
        # Prevent overflow: e^(-500) ≈ 0, e^(500) overflows
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross-Entropy Loss (Log Loss)
        
        Mathematical formula:
        L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Why this loss?
        - Penalizes confident wrong predictions heavily
        - Smooth gradient for optimization
        - Convex (has single global minimum)
        - Derived from maximum likelihood estimation
        
        Interpretation:
        - When y=1: loss = -log(ŷ), small when ŷ→1, large when ŷ→0
        - When y=0: loss = -log(1-ŷ), small when ŷ→0, large when ŷ→1
        
        Parameters:
        -----------
        y_true : array
            Actual labels (0 or 1)
        y_pred : array
            Predicted probabilities (between 0 and 1)
            
        Returns:
        --------
        Average loss across all samples
        """
        m = len(y_true)
        
        # Add epsilon to prevent log(0) which would give -infinity
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute cross-entropy: negative log-likelihood
        loss = -1/m * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute Gradients for Gradient Descent
        
        Derivation (using chain rule):
        ∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂w = (ŷ - y) * x
        ∂L/∂b = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂b = (ŷ - y)
        
        Key insight: The gradient is simply the error (ŷ - y) weighted by the features.
        - Large errors → large gradient → bigger weight update
        - Features with high values contribute more to the gradient
        
        Parameters:
        -----------
        X : array-like, shape (m, n)
            Feature matrix (m samples, n features)
        y_true : array, shape (m,)
            True labels
        y_pred : array, shape (m,)
            Predicted probabilities
            
        Returns:
        --------
        dw : array, shape (n,)
            Gradient with respect to weights
        db : float
            Gradient with respect to bias
        """
        m = len(y_true)
        
        # Error: how far off our predictions are
        error = y_pred - y_true
        
        # Handle sparse matrices from one-hot encoding efficiently
        if issparse(X):
            # Sparse matrix multiplication: X^T @ error
            dw = (1/m) * X.T.dot(error)
            dw = np.asarray(dw).flatten()  # Convert to dense array
        else:
            # Standard matrix multiplication
            dw = (1/m) * np.dot(X.T, error)
        
        # Bias gradient: average error across all samples
        db = (1/m) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        """
        Train the Logistic Regression Model using Gradient Descent
        
        Algorithm Overview:
        -------------------
        1. Initialize weights to zero (unbiased starting point)
        2. Repeat until convergence or max iterations:
           a. Forward pass: compute predictions
           b. Compute loss: measure how wrong we are
           c. Backward pass: compute gradients
           d. Update parameters: w = w - α * ∇w
        3. Stop when loss stops improving (converged)
        
        Gradient Descent Update Rule:
        w_new = w_old - learning_rate * gradient
        
        The learning rate controls how big our steps are:
        - Too large: might overshoot minimum, oscillate
        - Too small: slow convergence, many iterations needed
        
        Parameters:
        -----------
        X : array-like or sparse matrix, shape (m, n)
            Training data (m samples, n features)
        y : array-like, shape (m,)
            Target labels (0 or 1)
            
        Returns:
        --------
        self : fitted model
        """
        # Handle both dense and sparse input matrices
        if issparse(X):
            m, n = X.shape
            X_array = X  # Keep sparse for efficient operations
        else:
            X = np.array(X)
            m, n = X.shape
            X_array = X
        
        y = np.array(y).reshape(-1, 1)
        
        # Initialize parameters to zero (common practice)
        # Starting with zeros means no initial bias toward any class
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        prev_loss = float('inf')
        
        # Training loop: iterate until convergence or max iterations
        for iteration in range(self.max_iter):
            # --- FORWARD PASS ---
            # Compute linear combination: z = Xw + b
            if issparse(X_array):
                z = X_array.dot(self.weights) + self.bias
                z = np.asarray(z).flatten().reshape(-1, 1)
            else:
                z = np.dot(X_array, self.weights) + self.bias
            
            # Apply sigmoid to get probabilities: ŷ = σ(z)
            y_pred = self.sigmoid(z)
            
            # --- COMPUTE LOSS ---
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # --- BACKWARD PASS ---
            # Compute how much to adjust each parameter
            dw, db = self.compute_gradients(X_array, y, y_pred)
            
            # --- PARAMETER UPDATE ---
            # Move in the opposite direction of the gradient (descent)
            self.weights -= self.learning_rate * dw.reshape(-1, 1)
            self.bias -= self.learning_rate * db
            
            # --- CHECK CONVERGENCE ---
            # If loss isn't improving much, we've converged
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"✓ Converged at iteration {iteration}")
                break
            
            prev_loss = loss
            
            # Print progress periodically
            if self.verbose and (iteration % 100 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration:4d}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict Class Probabilities
        
        Returns probability estimates for both classes:
        - P(y=0|x) = 1 - σ(w^T x + b)
        - P(y=1|x) = σ(w^T x + b)
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Features to predict
            
        Returns:
        --------
        probabilities : array, shape (m, 2)
            [P(y=0), P(y=1)] for each sample
        """
        # Compute linear combination and apply sigmoid
        if issparse(X):
            z = X.dot(self.weights) + self.bias
            z = np.asarray(z).flatten()
        else:
            X = np.array(X)
            z = np.dot(X, self.weights) + self.bias
            z = z.flatten()
        
        # Get probability of positive class
        prob_class_1 = self.sigmoid(z)
        # Probability of negative class (must sum to 1)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X):
        """
        Predict Binary Class Labels
        
        Decision rule: 
        - If P(y=1|x) >= 0.5, predict 1
        - Otherwise, predict 0
        
        The 0.5 threshold means we predict the more likely class.
        This can be adjusted for imbalanced datasets.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Features to predict
            
        Returns:
        --------
        predictions : array
            Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        # Apply 0.5 decision threshold
        return (probas[:, 1] >= 0.5).astype(int)
