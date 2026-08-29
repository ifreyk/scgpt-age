"""
Calculate statistics for genes from perturbation analysis run files.

This script processes gene_results_run*.csv files and calculates:
- mean_high, mean_low: Mean expression values for high and low perturbations
- p_val: Mann-Whitney U test p-value
- logfc: Log2 fold change
- p_val_corrected: Holm-corrected p-value
- H0_reject: Whether null hypothesis is rejected
- -log10p_val_corrected: Negative log10 of corrected p-value
- changes: Classification (Geroprotector, Geroaccelerator, No changes)
- MT, non-coding, HB, ribo: Gene category flags
- confidence_level: Confidence level merged from gene-confidence-level.tsv
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
import re
from tqdm import tqdm
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def load_confidence_levels(confidence_file: str = "src/data/gene-confidence-level.tsv") -> pd.DataFrame:
    """
    Load confidence levels from TSV file.
    
    Args:
        confidence_file: Path to the gene-confidence-level.tsv file
    
    Returns:
        DataFrame with gene names and confidence levels
    """
    try:
        confidence_df = pd.read_csv(confidence_file, sep='\t')
        # Remove quotes if present and clean column names
        confidence_df.columns = confidence_df.columns.str.strip('"')
        confidence_df['hgnc'] = confidence_df['hgnc'].str.strip('"')
        confidence_df['confidence_level'] = confidence_df['confidence_level'].str.strip('"')
        return confidence_df
    except FileNotFoundError:
        print(f"Warning: Confidence level file not found: {confidence_file}")
        return pd.DataFrame(columns=['hgnc', 'confidence_level'])


def _process_single_gene(gene_data):
    """
    Process a single gene's statistics. Used for parallel processing.
    
    Args:
        gene_data: Tuple of (gene, high_values, low_values)
    
    Returns:
        Tuple of (gene, mean_high, mean_low, p_val)
    """
    gene, high_values, low_values = gene_data
    mean_high = np.mean(high_values)
    mean_low = np.mean(low_values)
    u, p = mannwhitneyu(high_values, low_values)
    return (gene, mean_high, mean_low, p)


def calculate_gene_statistics(
    run_file: str, 
    output_dir: str = None,
    confidence_df: pd.DataFrame = None,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Calculate statistics for genes from a single run file.
    
    Args:
        run_file: Path to the gene_results_run*.csv file
        output_dir: Directory to save output (default: same as input file)
        confidence_df: DataFrame with confidence levels (optional)
        n_workers: Number of parallel workers (default: number of CPU cores)
    
    Returns:
        DataFrame with calculated statistics
    """
    print(f"\nProcessing {run_file}...")
    
    # Read the result file
    result_df = pd.read_csv(run_file)
    
    # Extract gene names from columns (those ending in '_high')
    columns = [
        x.split('_max')[0] 
        for x in result_df.columns 
        if x not in ['Unnamed: 0', 'cells', 'tissue', 'age_category', 'age_id'] 
        and x.endswith('_max')
    ]
    columns = [x for x in columns if not '_zero' in x]
    
    # Prepare gene data for parallel processing
    gene_data_list = []
    for gene in columns:
        if f'{gene}_max' in result_df.columns and f'{gene}_zero' in result_df.columns:
            high_values = result_df[f'{gene}_max'].values
            low_values = result_df[f'{gene}_zero'].values
            gene_data_list.append((gene, high_values, low_values))
    
    # Set number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)  # Use all cores except one
    
    print(f"Processing {len(gene_data_list)} genes with {n_workers} workers...")
    
    # Process genes in parallel
    genes_all = []
    mean_high_list = []
    mean_low_list = []
    p_vals = []
    
    if n_workers == 1:
        gene_results = map(_process_single_gene, gene_data_list)
        for gene, mean_high, mean_low, p in tqdm(
            gene_results,
            total=len(gene_data_list),
            desc=f"Processing genes in {os.path.basename(run_file)}",
        ):
            genes_all.append(gene)
            mean_high_list.append(mean_high)
            mean_low_list.append(mean_low)
            p_vals.append(p)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_gene = {
                executor.submit(_process_single_gene, gene_data): gene_data[0]
                for gene_data in gene_data_list
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_gene),
                total=len(gene_data_list),
                desc=f"Processing genes in {os.path.basename(run_file)}",
            ):
                try:
                    gene, mean_high, mean_low, p = future.result()
                    genes_all.append(gene)
                    mean_high_list.append(mean_high)
                    mean_low_list.append(mean_low)
                    p_vals.append(p)
                except Exception as exc:
                    gene = future_to_gene[future]
                    print(f'Gene {gene} generated an exception: {exc}')
    
    # Create DataFrame with basic statistics
    stat_test_df = pd.DataFrame({
        'gene': genes_all,
        'mean_high': mean_high_list,
        'mean_low': mean_low_list,
        'p_val': p_vals
    })
    
    # Calculate log fold change
    stat_test_df['logfc'] = np.log2(stat_test_df['mean_low'] / stat_test_df['mean_high'])
    
    # Apply multiple testing correction (Holm method)
    rejected, p_val_corrected, _, _ = smm.multipletests(
        stat_test_df['p_val'], 
        method='holm'
    )
    stat_test_df['p_val_corrected'] = p_val_corrected
    stat_test_df['H0_reject'] = rejected
    
    # Calculate -log10(p_val_corrected)
    stat_test_df['-log10p_val_corrected'] = -np.log10(stat_test_df['p_val_corrected'])
    
    # Classify changes
    all_changes = []
    for x in stat_test_df.index:
        if stat_test_df.loc[x, 'H0_reject']:
            if stat_test_df.loc[x, 'logfc'] < 0:
                changes = 'Geroaccelerator'
            elif stat_test_df.loc[x, 'logfc'] > 0:
                changes = 'Geroprotector'
            else:
                changes = 'No changes'
        else:
            changes = 'No changes'
        all_changes.append(changes)
    stat_test_df['changes'] = all_changes
    
    # Add gene category flags
    stat_test_df['MT'] = [
        True if re.search(r"\b(?:MT|mitochondrial)\b", x, re.IGNORECASE) 
        else False for x in stat_test_df['gene']
    ]
    
    stat_test_df['non-coding'] = [
        True if re.search(r"\b(?:lncRNA|ncRNA|miRNA|siRNA|snoRNA|piRNA|tRNA)\b", x, re.IGNORECASE) 
        else False for x in stat_test_df['gene']
    ]
    
    stat_test_df['HB'] = [
        True if re.search(r"\b(?:HB|globin|HBB|HBD)\b", x, re.IGNORECASE) 
        else False for x in stat_test_df['gene']
    ]
    
    stat_test_df['ribo'] = [
        True if re.search(r"\b(?:rRNA|ribosomal|RPS|RPL)\b", x, re.IGNORECASE) 
        else False for x in stat_test_df['gene']
    ]
    
    # Merge confidence levels if provided
    if confidence_df is not None and not confidence_df.empty:
        # Merge on gene name (hgnc column)
        stat_test_df = stat_test_df.merge(
            confidence_df[['hgnc', 'confidence_level']],
            left_on='gene',
            right_on='hgnc',
            how='left'
        )
        # Fill missing confidence levels with "Not found"
        stat_test_df['confidence_level'] = stat_test_df['confidence_level'].fillna('Not found')
        # Drop the hgnc column as it's redundant with gene
        stat_test_df = stat_test_df.drop(columns=['hgnc'])
    else:
        # If no confidence levels available, set all to "Not found"
        stat_test_df['confidence_level'] = 'Not found'
    
    return stat_test_df


def process_all_runs(
    perturbation_results_dir: str = "perturbation_results_cls_decoder", 
    output_base_dir: str = "src/data/p_val_results_cls_h0",
    confidence_file: str = "src/data/gene-confidence-level.tsv",
    n_workers: int = None
):
    """
    Process all gene_results_run*.csv files from perturbation_results directory.
    For each tissue, creates a folder in output_base_dir and processes all run files.
    
    Args:
        perturbation_results_dir: Directory containing tissue folders with run files
        output_base_dir: Base directory to save output files (default: src/data/p_val_results)
        confidence_file: Path to the gene-confidence-level.tsv file
        n_workers: Number of parallel workers (default: number of CPU cores - 1)
    """
    # Create base output directory if it doesn't exist
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # Load confidence levels
    confidence_df = load_confidence_levels(confidence_file)
    if not confidence_df.empty:
        print(f"Loaded confidence levels for {len(confidence_df)} genes")
    
    # Find all tissue directories
    perturbation_path = Path(perturbation_results_dir)
    if not perturbation_path.exists():
        print(f"Error: Perturbation results directory not found: {perturbation_results_dir}")
        return
    
    # Get all subdirectories (tissues)
    tissue_dirs = [d for d in perturbation_path.iterdir() if d.is_dir()]
    
    if not tissue_dirs:
        print(f"No tissue directories found in {perturbation_results_dir}")
        return
    
    print(f"Found {len(tissue_dirs)} tissue directories to process")
    print("=" * 80)
    
    # Process each tissue
    for tissue_dir in sorted(tissue_dirs):
        tissue_name = tissue_dir.name
        print(f"\n{'=' * 80}")
        print(f"Processing tissue: {tissue_name}")
        print(f"{'=' * 80}")
        
        # Create tissue-specific output directory
        tissue_output_dir = output_base_path / tissue_name
        tissue_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all run files (excluding intermediate files)
        run_files = sorted([
            f for f in tissue_dir.glob("gene_results_run*.csv")
            if "_intermediate" not in f.name
        ])
        
        if not run_files:
            print(f"No gene_results_run*.csv files (excluding intermediate) found in {tissue_dir}")
            continue
        
        print(f"Found {len(run_files)} run files to process for {tissue_name}")
        
        for run_file in run_files:
            # Calculate statistics
            stat_test_df = calculate_gene_statistics(
                str(run_file), 
                str(tissue_output_dir), 
                confidence_df,
                n_workers=n_workers
            )
            
            # Save the results
            output_file = tissue_output_dir / run_file.name.replace('gene_results_', 'p_val_')
            stat_test_df.to_csv(output_file, index=False)
            
            print(f"Saved results to {output_file}")
            print(f"Total genes: {len(stat_test_df)}")
            print(f"Changes distribution:\n{stat_test_df['changes'].value_counts()}")
            print("-" * 80)
        
        print(f"\nCompleted processing tissue: {tissue_name}")
    
    print("\n" + "=" * 80)
    print("All tissues processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Process all tissues from perturbation_results directory
    process_all_runs()
