"""
Logistic regression

Models binary outcome probability: P(y=1|x) = σ(w^T x + b)
where σ is sigmoid and w, b are learned parameters.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import f1_score, precision_score, recall_score


class LogisticRegression:
    """
    Logistic regression via batch gradient descent.
    
    Algorithm: forward pass → loss → gradients → parameter update
    
    Parameters:
    -----------
    learning_rate : float
        Gradient descent step size. Too large = unstable, too small = slow.
    max_iter : int
        Maximum training iterations.
    tol : float
        Convergence tolerance for loss change.
    reg_lambda : float
        L2 regularization strength to prevent overfitting.
    lr_decay : float
        Learning rate decay: new_lr = old_lr / (1 + lr_decay * epoch)
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 reg_lambda=0.01, lr_decay=0.001, class_weight='balanced',
                 threshold=0.5):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.reg_lambda = reg_lambda
        self.lr_decay = lr_decay
        self.class_weight = class_weight
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.losses = []
        self.class_weights_ = None
    
    def sigmoid(self, z):
        """
        Sigmoid: σ(z) = 1 / (1 + e^(-z))
        Maps reals to (0,1). σ(0)=0.5, σ(+∞)→1, σ(-∞)→0
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def compute_class_weights(self, y):
        """Compute weights to handle imbalanced datasets."""
        if self.class_weight == 'balanced':
            classes = np.unique(y)
            n_samples = len(y)
            n_classes = len(classes)
            
            weights = {}
            for cls in classes:
                n_samples_cls = np.sum(y == cls)
                weights[cls] = n_samples / (n_classes * n_samples_cls)
            
            return weights
        else:
            return {0: 1.0, 1: 1.0}
    
    def compute_loss(self, y_true, y_pred, sample_weights=None):
        """
        Binary cross-entropy with L2 regularization:
        L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + λ/(2m) * ||w||²
        
        Penalizes confident wrong predictions. Convex, smooth gradient.
        Regularization prevents overfitting.
        """
        m = len(y_true)
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if sample_weights is not None:
            sample_losses = sample_losses * sample_weights
        
        cross_entropy = np.mean(sample_losses)
        l2_penalty = (self.reg_lambda / (2 * m)) * np.sum(self.weights ** 2)
        
        return cross_entropy + l2_penalty
    
    def compute_gradients(self, X, y_true, y_pred, sample_weights=None):
        """
        Gradients with L2 regularization:
        ∂L/∂w = 1/m * X^T(ŷ - y) + λ/m * w
        ∂L/∂b = 1/m * Σ(ŷ - y)
        
        Error weighted by features, plus regularization term.
        """
        m = len(y_true)
        
        error = y_pred - y_true
        
        if sample_weights is not None:
            error = error * sample_weights
        
        if issparse(X):
            dw = (1/m) * X.T.dot(error)
            dw = np.asarray(dw).flatten()
        else:
            dw = (1/m) * np.dot(X.T, error)
        
        dw += (self.reg_lambda / m) * self.weights.flatten()
        db = (1/m) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train model with optional early stopping.
        
        Steps: initialize weights (Xavier) → iterate until convergence:
        forward pass → loss → gradients → update → decay lr
        
        Parameters:
        -----------
        X : array-like or sparse, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Labels (0 or 1)
        X_val, y_val : optional validation set for early stopping
        """
        if issparse(X):
            m, n = X.shape
            X_array = X
        else:
            X = np.array(X)
            m, n = X.shape
            X_array = X
        
        y = np.array(y).reshape(-1, 1)
        
        self.class_weights_ = self.compute_class_weights(y.flatten())
        sample_weights = np.array([self.class_weights_[cls] for cls in y.flatten()]).reshape(-1, 1)
        
        # Xavier initialization: scale by sqrt(1/n)
        self.weights = np.random.randn(n, 1) * np.sqrt(1.0 / n)
        self.bias = 0
        
        prev_loss = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for iteration in range(self.max_iter):
            # Decay learning rate
            self.learning_rate = self.initial_lr / (1 + self.lr_decay * iteration)
            
            # Forward pass: z = Xw + b, ŷ = σ(z)
            if issparse(X_array):
                z = X_array.dot(self.weights) + self.bias
                z = np.asarray(z).flatten().reshape(-1, 1)
            else:
                z = np.dot(X_array, self.weights) + self.bias
            
            y_pred = self.sigmoid(z)
            
            # Loss and gradients
            loss = self.compute_loss(y, y_pred, sample_weights)
            self.losses.append(loss)
            
            dw, db = self.compute_gradients(X_array, y, y_pred, sample_weights)
            
            # Update: w = w - α * ∇w
            self.weights -= self.learning_rate * dw.reshape(-1, 1)
            self.bias -= self.learning_rate * db
            
            # Early stopping
            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {iteration}")
                        break
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break
            
            prev_loss = loss
        
        return self
    
    def _compute_val_loss(self, X_val, y_val):
        """Compute validation loss."""
        if issparse(X_val):
            z = X_val.dot(self.weights) + self.bias
            z = np.asarray(z).flatten().reshape(-1, 1)
        else:
            z = np.dot(X_val, self.weights) + self.bias
        
        y_pred = self.sigmoid(z)
        y_val = np.array(y_val).reshape(-1, 1)
        return self.compute_loss(y_val, y_pred)
    
    def predict_proba(self, X):
        """
        Return probability estimates: [P(y=0), P(y=1)] for each sample.
        """
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
        Predict binary labels. Returns 1 if P(y=1|x) >= threshold, else 0.
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)
    
    def optimize_threshold(self, X, y_true, metric='f1'):
        """
        Find optimal decision threshold to maximize metric.
        
        Parameters:
        -----------
        X : array-like
            Validation features
        y_true : array-like
            True labels
        metric : str
            'f1', 'precision', or 'recall'
        
        Returns best_threshold : float
        """
        probas = self.predict_proba(X)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (probas >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        return best_threshold