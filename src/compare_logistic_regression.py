# %%
"""
Comparison script: Logistic Regression vs scGPT model
Creates a pipeline with StandardScaler and Multinomial Logistic Regression
to compare performance with scGPT model on age classification task.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set scanpy figure parameters
sc.set_figure_params(figsize=(6, 6))

# Data paths
AGEANNO_ADATA_TRAIN = "src/data/subset_edited_genes_skin_ageanno_train_divided_by_donor.h5ad.gz"
AGEANNO_ADATA_TEST = "src/data/subset_edited_genes_skin_ageanno_test_divided_by_donor.h5ad.gz"

# Output directory
save_dir = Path("save/logistic_regression_comparison")
save_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load data
print("Loading data...")
adata_train = sc.read_h5ad(AGEANNO_ADATA_TRAIN)
adata_test = sc.read_h5ad(AGEANNO_ADATA_TEST)

# %%
# Prepare data similar to scGPT preprocessing
adata_train.var["gene_name"] = adata_train.var_names
adata_test.var["gene_name"] = adata_test.var_names

adata_train.obs["age_category"] = [x[:3] for x in adata_train.obs['orig.ident']]
adata_test.obs["age_category"] = [x[:3] for x in adata_test.obs['orig.ident']]

adata_train.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_train.obs['age_category']]
adata_test.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_test.obs['age_category']]

# %%
# Extract features (gene expression) and labels
print("Extracting features and labels...")

# Use raw expression data (X) - convert sparse to dense if needed
from scipy.sparse import issparse

if issparse(adata_train.X):
    X_train = adata_train.X.toarray()
else:
    X_train = adata_train.X

if issparse(adata_test.X):
    X_test = adata_test.X.toarray()
else:
    X_test = adata_test.X

y_train = adata_train.obs["age_id"].values
y_test = adata_test.obs["age_id"].values

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels - Class 0 (young): {np.sum(y_train == 0)}, Class 1 (old): {np.sum(y_train == 1)}")
print(f"Test labels - Class 0 (young): {np.sum(y_test == 0)}, Class 1 (old): {np.sum(y_test == 1)}")

# %%
# Create pipeline with StandardScaler and LogisticRegression
print("\nCreating pipeline with StandardScaler and LogisticRegression...")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs'  # lbfgs works well with multinomial
    ))
])

# %%
# Train the model
print("Training logistic regression model...")
pipeline.fit(X_train, y_train)

# %%
# Make predictions on test set
print("Making predictions on test set...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# For binary classification, use probability of positive class (class 1)
y_pred_proba_positive = y_pred_proba[:, 1]

# %%
# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
print(f"\n{'='*60}")
print(f"Logistic Regression ROC-AUC Score: {roc_auc:.4f}")
print(f"{'='*60}")

# %%
# Calculate other metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Young', 'Old']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
# Plot ROC curve
def plot_roc_curve(y_true, y_scores, roc_auc_score, title='ROC Curve', 
                   save_path=None, show_plot=True, figsize=(10, 8)):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores (probabilities of positive class)
        roc_auc_score: ROC-AUC score
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size tuple
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'ROC curve (AUC = {roc_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add text annotation with AUC score
    plt.text(0.6, 0.2, f'AUC = {roc_auc_score:.4f}', 
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

# %%
# Plot ROC curve
plot_roc_curve(y_test, y_pred_proba_positive, roc_auc,
              title='ROC Curve: Logistic Regression (StandardScaler + Multinomial LR)',
              save_path=save_dir / "roc_curve_logistic_regression.png",
              show_plot=True)

# %%
# Save results to file
results = {
    'model': 'Logistic Regression (StandardScaler + Multinomial LR)',
    'roc_auc': roc_auc,
    'n_features': X_train.shape[1],
    'n_train_samples': X_train.shape[0],
    'n_test_samples': X_test.shape[0],
    'train_class_distribution': {
        'young (0)': int(np.sum(y_train == 0)),
        'old (1)': int(np.sum(y_train == 1))
    },
    'test_class_distribution': {
        'young (0)': int(np.sum(y_test == 0)),
        'old (1)': int(np.sum(y_test == 1))
    }
}

import json
with open(save_dir / "results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {save_dir / 'results.json'}")
print(f"\nSummary:")
print(f"  Model: {results['model']}")
print(f"  ROC-AUC: {results['roc_auc']:.4f}")
print(f"  Features: {results['n_features']}")
print(f"  Training samples: {results['n_train_samples']}")
print(f"  Test samples: {results['n_test_samples']}")

# %%
# Save predictions for comparison
predictions_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'probability_class_0': y_pred_proba[:, 0],
    'probability_class_1': y_pred_proba[:, 1]
})

predictions_df.to_csv(save_dir / "predictions.csv", index=False)
print(f"\nPredictions saved to {save_dir / 'predictions.csv'}")

# %%
print("\n" + "="*60)
print("Logistic Regression model training and evaluation complete!")
print("="*60)
print(f"\nCompare this ROC-AUC ({roc_auc:.4f}) with the scGPT model ROC-AUC")
print(f"Results saved in: {save_dir}")

