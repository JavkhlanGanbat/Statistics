"""
Custom implementation of binary logistic regression using gradient descent.

This version performs:
- Forward pass (sigmoid activation)
- Binary cross-entropy loss with L2 regularization
- Class-imbalance weighting
- Batch gradient descent with learning rate decay
- Optional early stopping on validation loss
- Threshold optimization on F1/precision/recall

All mathematical formulas are included in the docstrings of each method.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import f1_score, precision_score, recall_score


class LogisticRegression:
    """
    Logistic regression classifier trained with batch gradient descent.

    Logistic Regression models the probability of a positive class as:

        P(y=1 | x) = σ(wᵀx + b)

    where σ is the sigmoid (inverse-logit) function:

        σ(z) = 1 / (1 + e^(−z))

    Training minimizes the **regularized binary cross-entropy loss**:

        L = -(1/m) Σ [ y log(ŷ) + (1 - y) log(1 - ŷ) ]  +  λ/(2m) ||w||²

    Parameters
    ----------
    learning_rate : float
        Initial gradient descent step size. Controls update magnitude.
    max_iter : int
        Maximum number of passes through the full training batch.
    tol : float
        Stop early if the change in loss between iterations drops below this.
    reg_lambda : float
        L2 regularization strength (prevents overfitting by shrinking weights).
    lr_decay : float
        Learning rate decay factor. New LR = initial_lr / (1 + decay * t)
    class_weight : str or dict
        If 'balanced', applies inverse-frequency weighting to handle imbalance.
    threshold : float
        Default decision boundary for converting probabilities → labels.
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
        Compute the logistic sigmoid function.

        Mathematical definition:
            σ(z) = 1 / (1 + e^(−z))

        The sigmoid maps any real number into a probability in (0, 1).

        - z >> 0 → σ(z) ≈ 1  
        - z = 0 → σ(z) = 0.5  
        - z << 0 → σ(z) ≈ 0  

        Clipping is applied for numerical stability to avoid overflow
        when computing e^(−z).
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_class_weights(self, y):
        """
        Compute per-class sample weights for imbalanced datasets.

        Balanced weighting uses the rule:

            w_c = N_total / (N_classes * N_c)

        where N_c is the count of samples in class c.

        This increases the influence of the minority class so that
        the model does not learn a trivial "predict majority every time"
        solution.

        Returns
        -------
        dict
            Mapping {class_value: weight}
        """
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
        Compute the regularized binary cross-entropy loss.

        Binary cross-entropy:

            L_ce = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]

        L2 regularization term:

            L_reg = (λ / (2m)) ||w||²

        Total loss:

            L = L_ce + L_reg

        This loss is convex for logistic regression, ensuring a single global
        minimum (good for gradient descent).

        Parameters
        ----------
        y_true : array of shape (m,1)
            Ground truth labels (0 or 1).
        y_pred : array of shape (m,1)
            Model-predicted probabilities.
        sample_weights : array, optional
            Per-sample weights for class balancing.

        Returns
        -------
        float
            Total loss value.
        """
        m = len(y_true)

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        sample_losses = -(y_true * np.log(y_pred) +
                          (1 - y_true) * np.log(1 - y_pred))

        if sample_weights is not None:
            sample_losses = sample_losses * sample_weights

        cross_entropy = np.mean(sample_losses)
        l2_penalty = (self.reg_lambda / (2 * m)) * np.sum(self.weights ** 2)

        return cross_entropy + l2_penalty

    def compute_gradients(self, X, y_true, y_pred, sample_weights=None):
        """
        Compute the gradients of the loss w.r.t weights and bias.

        For logistic regression, the gradient is:

            ∂L/∂w = (1/m) Xᵀ(ŷ - y) + (λ/m)w
            ∂L/∂b = (1/m) Σ(ŷ - y)

        Explanation:
        - (ŷ - y) is the "error signal"
        - Xᵀ(ŷ - y) accumulates how each feature contributes to the error
        - Regularization nudges weights toward zero to prevent overfitting

        Parameters
        ----------
        X : array-like, shape (m,n)
        y_true : array-like, shape (m,1)
        y_pred : array-like, shape (m,1)
        sample_weights : array-like, optional

        Returns
        -------
        (dw, db) : tuple
            Gradients for weights and bias.
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
        Train the logistic regression model using batch gradient descent.

        Training loop per iteration t:

        1. Compute learning rate decay:
               α_t = α₀ / (1 + decay · t)

        2. Forward pass:
               z = Xw + b
               ŷ = σ(z)

        3. Compute loss L

        4. Compute gradients (dw, db)

        5. Parameter update:
               w ← w − α_t · dw
               b ← b − α_t · db

        6. Early stopping:
               stop if validation loss stops improving

        Parameters
        ----------
        X : array-like or sparse
            Training features.
        y : array-like
            Binary labels.
        X_val, y_val : array-like, optional
            Validation data for early stopping.

        Returns
        -------
        self
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
        sample_weights = np.array([self.class_weights_[cls]
                                   for cls in y.flatten()]).reshape(-1, 1)

        # Xavier initialization: helps avoid exploding/vanishing initial values
        self.weights = np.random.randn(n, 1) * np.sqrt(1.0 / n)
        self.bias = 0

        prev_loss = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for iteration in range(self.max_iter):
            # Learning rate decay
            self.learning_rate = self.initial_lr / (1 + self.lr_decay * iteration)

            # Forward pass
            if issparse(X_array):
                z = X_array.dot(self.weights) + self.bias
                z = np.asarray(z).flatten().reshape(-1, 1)
            else:
                z = np.dot(X_array, self.weights) + self.bias

            y_pred = self.sigmoid(z)

            # Loss and gradient
            loss = self.compute_loss(y, y_pred, sample_weights)
            self.losses.append(loss)

            dw, db = self.compute_gradients(X_array, y, y_pred, sample_weights)

            # Parameter update
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
                        break  # stop training

            # Convergence check
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def _compute_val_loss(self, X_val, y_val):
        """
        Compute validation loss using the same cross-entropy function.

        This is used only for early stopping and does not update parameters.
        """
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
        Predict class probabilities for each sample.

        Output format:

            [ P(y=0 | x),  P(y=1 | x) ]

        P(y=1 | x) is computed using σ(wᵀx + b).
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
        Convert probabilities into class predictions using a threshold.

        Default rule:
            predict 1 if P(y=1 | x) ≥ threshold
            else predict 0

        The threshold can be optimized for F1/precision/recall using
        optimize_threshold().
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)

    def optimize_threshold(self, X, y_true, metric='f1'):
        """
        Search for the optimal probability threshold that maximizes a metric.

        Evaluated thresholds span the range [0.1, 0.9].

        For each threshold τ:

            y_pred = 1 if P(y=1 | x) ≥ τ else 0  
            compute metric(y_true, y_pred)

        Metrics available:
            - 'f1'        → balances precision/recall
            - 'precision' → reduces false positives
            - 'recall'    → reduces false negatives

        The selected threshold is stored in self.threshold.

        Returns
        -------
        float
            Best-performing threshold.
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
