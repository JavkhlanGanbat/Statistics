"""
Visualization module for model results.
Supports terminal, matplotlib, and Jupyter notebook outputs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os

from utils import load_data, load_model
from config import RESULTS_DIR, TEST_SIZE, RANDOM_STATE, FIGURE_SIZE, DPI


class ModelEvaluator:
    """Class to evaluate model and compute metrics."""
    
    def __init__(self):
        """Initialize evaluator with model and data."""
        self.model = load_model()
        self.X_train, self.X_val, self.y_train, self.y_val = self._load_split_data()
        self.y_pred = self.model.predict(self.X_val)
        
    def _load_split_data(self):
        """Load and split the training data."""
        df = load_data(split="train")
        if "fnlwgt" in df.columns:
            df = df.drop(columns=["fnlwgt"])
        
        X = df.drop(columns=["income_>50K"])
        y = df["income_>50K"]
        
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    def get_metrics(self):
        """Calculate and return all metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "Accuracy": accuracy_score(self.y_val, self.y_pred),
            "Precision": precision_score(self.y_val, self.y_pred),
            "Recall": recall_score(self.y_val, self.y_pred),
            "F1 Score": f1_score(self.y_val, self.y_pred)
        }
    
    def get_confusion_matrix(self):
        """Return confusion matrix."""
        return confusion_matrix(self.y_val, self.y_pred)
    
    def get_feature_importance(self, top_n=20):
        """Extract feature importance from the logistic regression model."""
        # Get the preprocessor and logistic regression
        preprocessor = self.model.named_steps['preprocess']
        logreg = self.model.named_steps['logreg']
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Numeric features
        numeric_features = preprocessor.transformers_[0][2]
        feature_names.extend(numeric_features)
        
        # Categorical features (one-hot encoded)
        cat_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(
            preprocessor.transformers_[1][2]
        )
        feature_names.extend(cat_feature_names)
        
        # Get coefficients
        coefficients = logreg.coef_[0]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df.head(top_n)


class ResultsVisualizer:
    """Class to create visualizations for model results."""
    
    def __init__(self):
        """Initialize visualizer with model and data."""
        self.evaluator = ModelEvaluator()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.style.use('default')
    
    def print_terminal_report(self):
        """Print comprehensive evaluation report."""
        metrics = self.evaluator.get_metrics()
        cm = self.evaluator.get_confusion_matrix()
        
        print("\n" + "="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        
        print("\nPerformance Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric:12s}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print("-" * 30)
        print(f"True Negatives:  {cm[0][0]:5d}")
        print(f"False Positives: {cm[0][1]:5d}")
        print(f"False Negatives: {cm[1][0]:5d}")
        print(f"True Positives:  {cm[1][1]:5d}")
        
        print("\n" + "="*50 + "\n")
    
    def plot_confusion_matrix(self, save=True):
        """Plot confusion matrix."""
        cm = self.evaluator.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['<=50K', '>50K']
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=DPI)
            print(f"Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")
        
        return fig
    
    def plot_feature_importance(self, top_n=15, save=True):
        """Plot top N feature importances."""
        importance_df = self.evaluator.get_feature_importance(top_n=top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
        
        colors = ['green' if x > 0 else 'red' for x in importance_df['coefficient']]
        
        ax.barh(range(len(importance_df)), importance_df['coefficient'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=DPI)
            print(f"Feature importance plot saved to {RESULTS_DIR}/feature_importance.png")
        
        return fig
    
    def plot_metrics_comparison(self, save=True):
        """Plot comparison of different metrics."""
        metrics = self.evaluator.get_metrics()
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'), dpi=DPI)
            print(f"Metrics comparison plot saved to {RESULTS_DIR}/metrics_comparison.png")
        
        return fig
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        print(f"\nGenerating visualizations and saving to: {RESULTS_DIR}\n")
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_metrics_comparison()
        print("\n✓ All visualizations saved!")


def main():
    """Generate all reports and visualizations."""
    viz = ResultsVisualizer()
    
    # Terminal report
    viz.print_terminal_report()
    
    # Plots
    viz.generate_all_plots()


if __name__ == "__main__":
    main()
