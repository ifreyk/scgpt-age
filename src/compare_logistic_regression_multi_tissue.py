# %%
"""
Multi-tissue Logistic Regression Comparison Script
Performs logistic regression on multiple tissues and compares results.
For each tissue, loads data from: donor_divided/subset_edited_genes_{tissue}_ageanno_{train/test}.h5ad.gz
"""
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
import json
from scipy.sparse import issparse
from typing import List, Dict, Tuple
warnings.filterwarnings('ignore')

# Set scanpy figure parameters
sc.set_figure_params(figsize=(6, 6))

# Configuration: List of tissues to process
# Example: TISSUES = ["skin", "bladder", "lung", "heart"]
TISSUES = ["skin","bladder","blood","bone-marrow","brain","liver","lung","pancreas","skeletal-muscle","stomach","testis"]  # Modify this list to include multiple tissues

# Data directory
DATA_DIR = Path("src/data/donor_divided")

# Output directory
save_dir = Path("save/logistic_regression_multi_tissue_regul")
save_dir.mkdir(parents=True, exist_ok=True)

# %%
def load_tissue_data(tissue: str) -> Tuple:
    """
    Load train and test data for a specific tissue.
    
    Args:
        tissue: Name of the tissue (e.g., "skin", "bladder")
    
    Returns:
        Tuple of (adata_train, adata_test)
    """
    train_path = DATA_DIR / f"subset_edited_genes_{tissue}_ageanno_train.h5ad.gz"
    test_path = DATA_DIR / f"subset_edited_genes_{tissue}_ageanno_test.h5ad.gz"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    print(f"Loading {tissue} data...")
    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)
    
    return adata_train, adata_test

# %%
def prepare_data(adata_train, adata_test):
    """
    Prepare data similar to scGPT preprocessing.
    
    Args:
        adata_train: Training AnnData object
        adata_test: Test AnnData object
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Prepare data similar to scGPT preprocessing
    adata_train.var["gene_name"] = adata_train.var_names
    adata_test.var["gene_name"] = adata_test.var_names
    
    adata_train.obs["age_category"] = [x[:3] for x in adata_train.obs['orig.ident']]
    adata_test.obs["age_category"] = [x[:3] for x in adata_test.obs['orig.ident']]
    
    adata_train.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_train.obs['age_category']]
    adata_test.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_test.obs['age_category']]
    
    # Extract features (gene expression) and labels
    # Use raw expression data (X) - convert sparse to dense if needed
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
    
    return X_train, X_test, y_train, y_test

# %%
def train_logistic_regression(X_train, y_train, X_test, y_test, tissue: str, gene_names: List[str], C: float = 1.0):
    """
    Train logistic regression model with L1 regularization and make predictions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        tissue: Tissue name for logging
        gene_names: List of gene names corresponding to features
        C: Inverse of regularization strength (smaller values specify stronger regularization)
    
    Returns:
        Dictionary with predictions, metrics, and coefficients
    """
    print(f"\n{'='*60}")
    print(f"Processing {tissue.upper()} tissue")
    print(f"{'='*60}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels - Class 0 (mid): {np.sum(y_train == 0)}, Class 1 (old): {np.sum(y_train == 1)}")
    print(f"Test labels - Class 0 (mid): {np.sum(y_test == 0)}, Class 1 (old): {np.sum(y_test == 1)}")
    
    # Create pipeline with StandardScaler and LogisticRegression with L1 regularization
    print(f"\nCreating pipeline with StandardScaler and LogisticRegression (L1 regularization, C={C})...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            penalty='l1',
            solver='liblinear',
            C=C,
            multi_class='ovr'  # Use 'ovr' for binary classification with liblinear
        ))
    ])
    
    # Train the model
    print("Training logistic regression model with L1 regularization...")
    pipeline.fit(X_train, y_train)
    
    # Extract coefficients
    classifier = pipeline.named_steps['classifier']
    coefficients = classifier.coef_[0]  # For binary classification, coef_ is shape (1, n_features)
    intercept = classifier.intercept_[0]
    
    # Count non-zero coefficients (non-nulled genes)
    n_nonzero_coef = np.sum(coefficients != 0)
    n_zero_coef = len(coefficients) - n_nonzero_coef
    
    print(f"\nRegularization results:")
    print(f"  Total genes: {len(coefficients)}")
    print(f"  Non-zero coefficients (selected genes): {n_nonzero_coef}")
    print(f"  Zero coefficients (nulled genes): {n_zero_coef}")
    print(f"  Sparsity: {n_zero_coef/len(coefficients)*100:.2f}%")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # For binary classification, use probability of positive class (class 1)
    y_pred_proba_positive = y_pred_proba[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"{tissue.upper()} Results:")
    print(f"{'='*60}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'='*60}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Mid', 'Old']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Create coefficients dataframe
    coefficients_df = pd.DataFrame({
        'gene': gene_names,
        'coefficient': coefficients
    })
    coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
    
    return {
        'pipeline': pipeline,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_pred_proba_positive': y_pred_proba_positive,
        'y_test': y_test,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Mid', 'Old'], output_dict=True),
        'coefficients': coefficients,
        'coefficients_df': coefficients_df,
        'intercept': intercept,
        'n_nonzero_coef': int(n_nonzero_coef),
        'n_zero_coef': int(n_zero_coef),
        'regularization_C': C,
        'regularization_type': 'L1'
    }

# %%
def plot_roc_curve_single(y_true, y_scores, roc_auc_score, title='ROC Curve', 
                          save_path=None, show_plot=False, figsize=(10, 8)):
    """
    Plot ROC curve with AUC score for a single tissue.
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
def plot_roc_curve_multi_tissue(tissue_results: Dict[str, Dict], save_path=None, show_plot=True, figsize=(12, 10)):
    """
    Plot ROC curves for multiple tissues on the same plot.
    
    Args:
        tissue_results: Dictionary mapping tissue names to their results dictionaries
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Color palette for different tissues
    colors = plt.cm.tab10(np.linspace(0, 1, len(tissue_results)))
    
    for idx, (tissue, results) in enumerate(tissue_results.items()):
        y_test = results['y_test']
        y_pred_proba_positive = results['y_pred_proba_positive']
        roc_auc = results['roc_auc']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_positive)
        plt.plot(fpr, tpr, color=colors[idx], lw=2.5, 
                label=f'{tissue.upper()} (AUC = {roc_auc:.4f})', alpha=0.8)
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier (AUC = 0.50)', alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title('ROC Curves: Logistic Regression (L1 Regularized) - Multiple Tissues', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined ROC curve saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

# %%
def save_tissue_results(tissue: str, results: Dict, X_train, X_test, y_train, y_test, tissue_save_dir: Path):
    """
    Save results for a single tissue.
    
    Args:
        tissue: Tissue name
        results: Results dictionary from train_logistic_regression
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        tissue_save_dir: Directory to save tissue-specific results
    """
    # Save results JSON with regularization info
    results_dict = {
        'tissue': tissue,
        'model': 'Logistic Regression (StandardScaler + L1 Regularized LR)',
        'regularization': {
            'type': results['regularization_type'],
            'C': float(results['regularization_C']),
            'n_total_genes': int(X_train.shape[1]),
            'n_selected_genes': results['n_nonzero_coef'],
            'n_nulled_genes': results['n_zero_coef'],
            'sparsity_percentage': float(results['n_zero_coef'] / X_train.shape[1] * 100)
        },
        'roc_auc': float(results['roc_auc']),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'n_features': int(X_train.shape[1]),
        'n_train_samples': int(X_train.shape[0]),
        'n_test_samples': int(X_test.shape[0]),
        'train_class_distribution': {
            'mid (0)': int(np.sum(y_train == 0)),
            'old (1)': int(np.sum(y_train == 1))
        },
        'test_class_distribution': {
            'mid (0)': int(np.sum(y_test == 0)),
            'old (1)': int(np.sum(y_test == 1))
        },
        'confusion_matrix': results['confusion_matrix'],
        'classification_report': results['classification_report'],
        'intercept': float(results['intercept'])
    }
    
    with open(tissue_save_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': results['y_pred'],
        'probability_class_0': results['y_pred_proba'][:, 0],
        'probability_class_1': results['y_pred_proba'][:, 1]
    })
    predictions_df.to_csv(tissue_save_dir / "predictions.csv", index=False)
    
    # Save coefficients CSV
    results['coefficients_df'].to_csv(tissue_save_dir / "coefficients.csv", index=False)
    print(f"Coefficients saved to {tissue_save_dir / 'coefficients.csv'}")
    
    # Plot and save ROC curve
    plot_roc_curve_single(
        y_test, 
        results['y_pred_proba_positive'], 
        results['roc_auc'],
        title=f'ROC Curve: Logistic Regression (L1) - {tissue.upper()}',
        save_path=tissue_save_dir / "roc_curve.png",
        show_plot=False
    )
    
    print(f"\n{tissue.upper()} results saved to {tissue_save_dir}")

# %%
def print_summary_table(tissue_results: Dict[str, Dict]):
    """
    Print a summary table with all metrics for each tissue.
    
    Args:
        tissue_results: Dictionary mapping tissue names to their results dictionaries
    """
    print("\n" + "="*80)
    print("SUMMARY: All Metrics for Each Tissue")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    for tissue, results in tissue_results.items():
        summary_data.append({
            'Tissue': tissue.upper(),
            'ROC-AUC': f"{results['roc_auc']:.4f}",
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1']:.4f}",
            'Selected_Genes': results['n_nonzero_coef'],
            'Nulled_Genes': results['n_zero_coef']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Save summary to CSV
    summary_df.to_csv(save_dir / "summary_metrics.csv", index=False)
    print(f"\nSummary metrics saved to {save_dir / 'summary_metrics.csv'}")

# %%
# Main execution
if __name__ == "__main__":
    print("="*80)
    print("Multi-Tissue Logistic Regression Analysis (L1 Regularized)")
    print("="*80)
    print(f"Tissues to process: {', '.join([t.upper() for t in TISSUES])}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {save_dir}")
    print("="*80)
    
    # Store results for all tissues
    all_tissue_results = {}
    
    # Load and filter aging-related DEGs for all tissues
    ageanno_genes = pd.read_csv("src/data/Aging-related DEGs.txt", encoding='iso-8859-1')
    ageanno_genes = ageanno_genes[ageanno_genes['group']=='old vs mid']
    unique_genes = ageanno_genes['gene'].unique().tolist()
    
    # Process each tissue
    for tissue in TISSUES:
        try:
            # Create tissue-specific save directory
            tissue_save_dir = save_dir / tissue
            tissue_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data
            adata_train, adata_test = load_tissue_data(tissue)
            
            # Prepare data
            X_train, X_test, y_train, y_test = prepare_data(adata_train, adata_test)
            
            # Filter to aging-related DEGs for all tissues
            # Get gene names from AnnData objects
            gene_names = adata_train.var_names.tolist()
            # Find indices of genes that are in unique_genes
            gene_indices = [i for i, gene in enumerate(gene_names) if gene in unique_genes]
            # Filter X_train and X_test to only include these genes
            X_train = X_train[:, gene_indices]
            X_test = X_test[:, gene_indices]
            # Get filtered gene names
            filtered_gene_names = [gene_names[i] for i in gene_indices]
            print(f"\nFiltered to {len(gene_indices)} aging-related DEGs for {tissue} tissue")
            
            # Train model with L1 regularization and get results
            # Using C=1.0 for L1 regularization (can be adjusted)
            results = train_logistic_regression(X_train, y_train, X_test, y_test, tissue, filtered_gene_names, C=1.0)
            
            # Store results
            all_tissue_results[tissue] = results
            
            # Save tissue-specific results
            save_tissue_results(tissue, results, X_train, X_test, y_train, y_test, tissue_save_dir)
            
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print(f"Skipping {tissue} tissue...")
            continue
        except Exception as e:
            print(f"\nERROR processing {tissue} tissue: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # If we have multiple tissues, create combined visualizations
    if len(all_tissue_results) > 1:
        print("\n" + "="*80)
        print("Creating combined visualizations for multiple tissues...")
        print("="*80)
        
        # Plot combined ROC curves
        plot_roc_curve_multi_tissue(
            all_tissue_results,
            save_path=save_dir / "roc_curve_multi_tissue.png",
            show_plot=True
        )
    
    # Print summary table
    if all_tissue_results:
        print_summary_table(all_tissue_results)
        
        # Save combined results JSON
        combined_results = {
            'tissues': list(all_tissue_results.keys()),
            'results': {
                tissue: {
                    'roc_auc': float(results['roc_auc']),
                    'accuracy': float(results['accuracy']),
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'f1': float(results['f1'])
                }
                for tissue, results in all_tissue_results.items()
            }
        }
        
        with open(save_dir / "combined_results.json", "w") as f:
            json.dump(combined_results, f, indent=4)
        
        print(f"\nCombined results saved to {save_dir / 'combined_results.json'}")
    
    print("\n" + "="*80)
    print("Multi-Tissue Logistic Regression Analysis Complete!")
    print("="*80)
    print(f"\nResults saved in: {save_dir}")
    if len(all_tissue_results) > 1:
        print(f"ROC-AUC scores for all tissues:")
        for tissue, results in all_tissue_results.items():
            print(f"  {tissue.upper()}: {results['roc_auc']:.4f}")
    print("="*80)

