"""
Логистик регресс

Хоёртын үр дүн (binary outcome)-г загварчлах: P(y=1|x) = σ(w^T x + b)
σ: Сигмойд функц
w, b: Модель өөрөө сурах параметрүүд.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import f1_score, precision_score, recall_score


class LogisticRegression:
    """
    Logistic regression & batch gradient descent.
    
    forward pass → loss → gradients → parameter update.
    
    Энэ модел нь оролтын x дээр үндэслэн хоёртын үр дүнгийн магадлал
    P(y=1 | x)-ийг таамагладаг. Оролт болон параметрийн шугаман нийлбэрийг
    (w^T x + b) сигмоид функц нь хэрэглээд магадлал руу хувиргана.
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
        
        Сигмоид функц нь бодит тоог (0,1) интервалд орших магадлал руу хувиргана..
        
        σ(z) нь:
        - z их эерэг тоо бол ≈ 1
        - z их сөрөг тоо бол ≈ 0
        - z = 0 үед 0.5
        
        z-г clip() хийх нь хөвөгч цэгийн overflow-г багасгана.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_class_weights(self, y):
        """
        Классуудын жинг тооцоолно (Тэнцвэргүй байдлыг харгалзан үзсэн)
        
        'balanced' үед:
            weight(c) = n_samples / (n_classes * n_samples_in_class_c)
        
        Энэ нь тэнцвэргүй өгөгдлийн үед бага тоотой ангиллыг илүү "жинтэй"
        болгож, үргэлж олонхын прогнозчилох хандлагыг бууруулна.
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
        L2 тогтворжуулалттай (regularization) хоёртын cross-entropy алдаа
        
        Хоёртын cross-entropy:

            L_ce = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]

        L2 тогтворжуулалт

            L_reg = (λ / (2m)) ||w||²

        Нийт алдаа:

            L = L_ce + L_reg

        Cross-entropy нь буруу таамаглалуудыг итгэлтэйгээр гаргахад илүү шийтгэдэг
        
        L2 регуляц нь том жинuүүдийг багасгаж overfitting-ээс сэргийлдэг.

        Энэ алдааны функц L(w,b) нь w, b параметрүүдийн хувьд хотгор, ганцхан глобал минимум утгатай.
        Gradient descent аргаар энэ минимум утгыг гаргаж авахад харьцангүй амар.
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
        Алдааны функцын gradient-г тооцоолно.
        Таамагласан магадлал:
            ŷ = σ(z) = σ(Xw + b)

            X = онцлогийн (feature) матриц (m түүвэр × n онцлог)
            w = жингийн вектор (n × 1)
            b = хазайлтын скаляр
            ŷ = таамагласан магадлал
            y = жинхэнэ утга (m × 1)

        Алдааны функц L-г түүний параметр болох w болон b-н хувьд тухайн уламжлалыг авч
        gradient-уудыг тооцоолно.

        ∂L/∂w = 1/m * X^T(ŷ - y) + λ/m * w
        ∂L/∂b = 1/m * Σ(ŷ - y)
        
        Энэ градиент нь параметрүүдийг бууруулах чиглэлийг зааж өгнө.
        λ/m * w хэсэг нь жингүүдийг хэт өсөхөөс сэргийлнэ.
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
        Train the model using batch gradient descent.
        
        Сургалтын алхмууд:
        1. Xavier initialization (жингүүдийг √(1/n)-р үржүүлэн эхлүүлэх)
        2. Forward pass: ŷ = σ(Xw + b)
        3. Алдаагаа тооцох
        4. Gradient descent => w, b оновчлох
        5. Learning rate decay = lr / (1 + decay * epoch)
        6. Early stopping (Үнэлгээний өгөгдөл дээр алдаа нь өсөхөд)
        
        tol параметр нь loss-ийн өөрчлөлт бага болсон үед сургалтыг дуусгана.
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
        
        self.weights = np.random.randn(n, 1) * np.sqrt(1.0 / n)
        self.bias = 0
        
        prev_loss = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for iteration in range(self.max_iter):
            self.learning_rate = self.initial_lr / (1 + self.lr_decay * iteration)
            
            if issparse(X_array):
                z = X_array.dot(self.weights) + self.bias
                z = np.asarray(z).flatten().reshape(-1, 1)
            else:
                z = np.dot(X_array, self.weights) + self.bias
            
            y_pred = self.sigmoid(z)
            
            loss = self.compute_loss(y, y_pred, sample_weights)
            self.losses.append(loss)
            
            dw, db = self.compute_gradients(X_array, y, y_pred, sample_weights)
            
            self.weights -= self.learning_rate * dw.reshape(-1, 1)
            self.bias -= self.learning_rate * db
            
            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        
        return self
    
    def _compute_val_loss(self, X_val, y_val):
        """
        Үнэлгээний өгөгдөл дээр алдааг тооцоолно.
        
        Энийг ашигласнаар
        - overfitting илрүүлнэ
        - early stopping хийнэ
        
        Уг алгоритм нь сургалтын алдаанаас гадна модель өмнө нь хараагүй өгөгдөл дээр
        хэрхэн ажиллаж байгааг шалгахад тусална.
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
        Түүвэр буюу мөр болгонд классуудын магадлалыг буцаана: [P(y=0), P(y=1)].
        
        - P(y=1) = σ(w^T x + b)
        - P(y=0) = 1 - P(y=1)
        
        Энэ функц нь [0,1] интервалд орших жинхэнэ утгыг буцаана. Энэ утгуудыг 
        threshold утгатайгаа жишиж 0 эсвэл 1 гэсэн дискрет утгуудыг буцаана.
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
        P(y=1|x) >= threshold   =>   1
        P(y=1|x) < threshold    =>   0
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)
    
    def optimize_threshold(self, X, y_true, metric='f1'):
        """
        Аль нэг үзүүлэлтийг дээд зэргээр ихэсгэх заагийн утгыг олох.
        
        Үнэлгээний түүвэр дээр энэ утгыг [0.1, 0.9] интервалаас хайж,
        F1, precision, эсвэл recall хамгийн их болдог утгыг сонгоно.
        
        Тэнцвэргүй өгөгдлийн үед маш чухал бөгөөд
        model-ийн бодит хэрэглээний зорилгод
        хамгийн сайн тохирох шийдвэрийн цэгийг олно.
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
