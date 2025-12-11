"""
Perturbation Analysis for ImprovedCls Models Trained on scGPT Embeddings

This script performs perturbation analysis on tissue-specific ImprovedCls models:
1. Loads test and train data for a specific tissue (e.g., skin)
2. Loads the frozen scGPT encoder from src/data/models/scGPT_human/best_model.pt
3. Loads the trained ImprovedCls head from save/cls_decoder_on_embeddings/{tissue}/best_model.pt
4. Loads model config (args.json) and vocabulary (vocab.json) from src/data/models/scGPT_human
5. Samples 100 cells stratified by age_category from test data
6. For each gene, creates two perturbations:
   - Zero perturbation: sets gene expression to 0
   - Max perturbation: sets gene expression to maximum value found in train + test data
7. Runs predictions on perturbed data and saves results

Usage:
    python src/perturbation_analysis_cls_decoder.py
    
Configuration:
    TISSUE: tissue name (default: "skin")
    N_CELLS_TO_SAMPLE: number of cells to sample (default: 500)
"""

#%%
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse, csr_matrix, hstack
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from improved_cls_decoder import ImprovedCls
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt import SubsetsBatchSampler
from tqdm import tqdm
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
from typing import Dict, Optional, Union

import re
from scanpy.get import _get_obs_rep, _set_obs_rep

from scgpt import logger

# Preprocessor class (copied from perturbation_analysis.py)
class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        if (
            isinstance(self.filter_cell_by_counts, int)
            and self.filter_cell_by_counts > 0
        ):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        if self.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
            for row in layer_data:
                if row.max() == 0:
                    logger.warning(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False
        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False
        return True


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.
    """
    assert x.ndim == 1 and bins.ndim == 1
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits
    right_digits = np.digitize(x, bins, right=True)
    np.random.seed(0)
    rands = np.random.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def get_preds_and_probas(encoder_model: nn.Module, cls_decoder: nn.Module, loader: DataLoader, use_bce: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model on the evaluation data.
    First extracts embeddings from encoder, then passes through cls_decoder (ImprovedCls).
    """
    predictions = []
    preds_all = []
    encoder_model.eval()
    cls_decoder.eval()
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                # Extract embeddings from frozen encoder
                output_dict = encoder_model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS else None,
                    CLS=CLS,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                )
                cell_emb = output_dict["cell_emb"]  # Shape: [batch_size, embsize]
                
                # Pass embeddings through ClsDecoder
                logits = cls_decoder(cell_emb)
                
                if use_bce:
                    # For BCE: apply sigmoid to get probabilities
                    probs_bce = torch.sigmoid(logits[:, 1])
                    probs = torch.stack([1 - probs_bce, probs_bce], dim=1)
                    preds = (probs_bce > 0.5).long()
                    probas_ones = probs_bce.cpu().numpy()
                else:
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    probas_ones = probs[:, 1].cpu().numpy()

            predictions.append(preds.cpu().numpy())
            preds_all.append(probas_ones)
    predictions_result = np.concatenate(predictions, axis=0)
    preds_all_result = np.concatenate(preds_all, axis=0)
    return predictions_result, preds_all_result

def model_predict(encoder_model, cls_decoder, adata, gene_ids, use_bce=False):
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=filter_gene_by_counts,
        filter_cell_by_counts=True,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )

    preprocessor(adata, batch_key=None)
    
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    age_labels = adata.obs["age_id"].tolist()
    age_labels = np.array(age_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_edited = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )

    input_values_edited = random_mask_value(
        tokenized_edited["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    edited_data_pt = {
        "gene_ids": tokenized_edited["genes"],
        "values": input_values_edited,
        "target_values": tokenized_edited["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "age_labels": torch.from_numpy(age_labels).long(),
    }

    edited_loader = DataLoader(
        dataset=SeqDataset(edited_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    encoder_model.eval()
    cls_decoder.eval()
    predictions, probas = get_preds_and_probas(
        encoder_model,
        cls_decoder,
        loader=edited_loader,
        use_bce=use_bce
    )
    return {'predictions': predictions, 'probas': probas}

def create_perturbed_adata(adata, gene_name, perturbation_value):
    """
    Create perturbed adata by replacing the gene expression with perturbation_value.
    Only works for genes that already exist in the dataset.
    
    Args:
        adata: AnnData object to perturb
        gene_name: Name of the gene to perturb (must exist in adata.var_names)
        perturbation_value: Scalar value to set for all cells (e.g., 0 or max expression)
    
    Returns:
        Perturbed AnnData object with gene expression replaced
    
    Raises:
        ValueError: If gene_name is not found in adata.var_names
    """
    adata_perturbed = adata.copy()
    
    if gene_name not in adata_perturbed.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in adata.var_names. Cannot perturb non-existent gene.")
    
    # Gene exists, get its index by name
    gene_idx = adata_perturbed.var_names.get_loc(gene_name)
    
    # Verify that we got the correct gene by checking the name at that index
    gene_name_at_idx = adata_perturbed.var_names[gene_idx]
    if gene_name_at_idx != gene_name:
        raise ValueError(
            f"Gene name mismatch! Requested '{gene_name}' but found '{gene_name_at_idx}' at index {gene_idx}. "
            f"This should not happen - var_names may have duplicates or indexing issue."
        )
    
    # Ensure perturbation_value is a scalar
    if not np.isscalar(perturbation_value):
        if hasattr(perturbation_value, '__len__') and len(perturbation_value) == 1:
            perturbation_value = perturbation_value[0] if isinstance(perturbation_value, np.ndarray) else perturbation_value[0]
        else:
            raise ValueError(f"perturbation_value must be a scalar, got: {type(perturbation_value)}")
    
    # Replace gene expression
    if issparse(adata_perturbed.X):
        X_dense = adata_perturbed.X.toarray()
        # Replace gene expression
        X_dense[:, gene_idx] = perturbation_value
        adata_perturbed.X = csr_matrix(X_dense) if adata_perturbed.X.format == 'csr' else X_dense
        # Verify replacement
        new_values = adata_perturbed.X.toarray()[:, gene_idx]
    else:
        # Replace gene expression
        adata_perturbed.X[:, gene_idx] = perturbation_value
        # Verify replacement
        new_values = adata_perturbed.X[:, gene_idx]
    
    # Verify that values were actually changed (all should equal perturbation_value)
    if not np.allclose(new_values, perturbation_value, rtol=1e-10):
        raise RuntimeError(
            f"Failed to replace gene expression! Expected all values to be {perturbation_value}, "
            f"but got values: min={new_values.min()}, max={new_values.max()}, mean={new_values.mean()}"
        )
    
    return adata_perturbed

def create_zero_perturbed_adata(adata, gene_name):
    """
    Create perturbed adata by setting the gene expression to 0.
    Gene must already exist in the dataset.
    
    Args:
        adata: AnnData object to perturb
        gene_name: Name of the gene to set to zero (must exist in adata.var_names)
    
    Returns:
        Perturbed AnnData object with gene expression set to 0
    """
    return create_perturbed_adata(adata, gene_name, 0.0)

# Configuration
# List of tissues to process
TISSUES = [
    "skin",
    "bladder",
    "blood",
    "bone-marrow",
    "brain",
    "liver",
    "lung",
    "skeletal-muscle",
    "stomach"
]

SCGPT_MODEL_PATH = "src/data/models/scGPT_human"
CLS_DECODER_MODEL_PATH = "save/cls_decoder_on_embeddings"
DATA_DIR = Path("src/data/donor_divided")
N_CELLS_TO_SAMPLE = 300

# Settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = 0.0
mask_value = "auto"
include_zero_gene = False
max_seq_len = 900
n_bins = 51

# input/output representation
input_style = "binned"
output_style = "binned"

# settings for training
MLM = False
CLS = True
ADV = False
CCE = False
MVC = False
ECS = False
DAB = False
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "avg-pool"
adv_E_delay_epochs = 0
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = 0.0
dab_weight = 0.0

explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob

per_seq_batch_sample = False

# settings for optimizer
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="AgeAnno_finall",
    do_train=True,
    load_model="src/data/models/scGPT_human",
    mask_ratio=0.0,
    epochs=8,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    lr=1e-4,
    batch_size=100,
    layer_size=128,
    nlayers=4,
    nhead=5,
    dropout=0.4,
    weight_decay=3e-4,
    schedule_ratio=0.9,
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=True,
    DSBN=False,
    early_stopping_patience=3,
    label_smoothing=0.15,
    use_reduce_lr_on_plateau=True,
    reduce_lr_factor=0.5,
    reduce_lr_patience=1,
    min_lr=1e-6,
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
)
config = wandb.config
print(config)
# settings for optimizer
lr = config.lr
lr_ADV = 1e-3
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid = config.layer_size
nlayers = config.nlayers
nhead = config.nhead
dropout = config.dropout
# logging
log_interval = 100
save_eval_interval = 5
do_eval_scib_metrics = True

# validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False
filter_gene_by_counts = False
data_is_raw = False

# Main loop over tissues
for TISSUE in TISSUES:
    logger.info("=" * 80)
    logger.info(f"Processing tissue: {TISSUE}")
    logger.info("=" * 80)
    
    # Create save directory for this tissue
    save_dir = Path(f"perturbation_results_cls_decoder/{TISSUE}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    
    # Set model save directory for this tissue
    CLS_DECODER_SAVE_DIR = Path(f"{CLS_DECODER_MODEL_PATH}/{TISSUE}")
    
    # Load data
    test_path = DATA_DIR / f"subset_edited_genes_{TISSUE}_ageanno_test.h5ad.gz"
    train_path = DATA_DIR / f"subset_edited_genes_{TISSUE}_ageanno_train.h5ad.gz"

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}. Skipping tissue {TISSUE}.")
        continue
    if not train_path.exists():
        logger.error(f"Train file not found: {train_path}. Skipping tissue {TISSUE}.")
        continue

    logger.info(f"Loading test data from {test_path}")
    adata_test = sc.read_h5ad(test_path)
    adata_test.var["gene_name"] = adata_test.var_names

    logger.info(f"Loading train data from {train_path}")
    adata_train = sc.read_h5ad(train_path)
    adata_train.var["gene_name"] = adata_train.var_names

    # Prepare age annotations
    adata_test.obs["age_category"] = [x[:3] for x in adata_test.obs['orig.ident']]
    adata_train.obs["age_category"] = [x[:3] for x in adata_train.obs['orig.ident']]
    adata_test.obs["batch_id"] = 0
    adata_train.obs["batch_id"] = 0
    adata_test.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_test.obs['age_category']]
    adata_train.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_train.obs['age_category']]

    # Keep raw copies for perturbation (before preprocessing)
    adata_test_raw = adata_test.copy()
    adata_train_raw = adata_train.copy()

    # Load model configuration and vocabulary
    model_dir = Path(SCGPT_MODEL_PATH)
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"
    scgpt_model_file = model_dir / "best_model.pt"

    if not model_config_file.exists():
        logger.error(f"Model config not found: {model_config_file}. Skipping tissue {TISSUE}.")
        continue
    if not vocab_file.exists():
        logger.error(f"Vocab file not found: {vocab_file}. Skipping tissue {TISSUE}.")
        continue
    if not scgpt_model_file.exists():
        logger.error(f"scGPT model file not found: {scgpt_model_file}. Skipping tissue {TISSUE}.")
        continue

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata_test.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_test.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_test.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )

    # Load and filter Aging-related DEGs
    logger.info("Loading Aging-related DEGs table")
    ageanno_genes = pd.read_csv("src/data/Aging-related DEGs.txt", encoding='iso-8859-1')
    ageanno_genes = ageanno_genes[ageanno_genes['group']=='old vs mid']
    ageanno_genes = ageanno_genes[ageanno_genes['p_val_adj']<0.05]
    ageanno_genes = ageanno_genes[(ageanno_genes['isTissuespecific']==False) & (ageanno_genes['isCellTypespecific']==False)]
    unique_genes = ageanno_genes['gene'].unique().tolist()
    logger.info(f"Found {len(unique_genes)} genes in Aging-related DEGs after filtering")

    # Find which genes exist in data, vocab, and Aging-related DEGs
    train_gene_names = set(adata_train.var_names)
    test_gene_names = set(adata_test.var_names)
    ageanno_gene_names = set(unique_genes)

    # First find intersection of data and Aging-related DEGs
    genes_in_data_and_ageanno = train_gene_names & test_gene_names & ageanno_gene_names

    # Then filter by vocab (check each gene)
    available_genes = [gene for gene in genes_in_data_and_ageanno if gene in vocab]
    missing_in_data = list(ageanno_gene_names - (train_gene_names & test_gene_names))
    missing_in_vocab = [gene for gene in genes_in_data_and_ageanno if gene not in vocab]

    if len(missing_in_data) > 0:
        logger.warning(f"{len(missing_in_data)} genes from Aging-related DEGs are not present in the data. First 10: {missing_in_data[:10]}")
    if len(missing_in_vocab) > 0:
        logger.warning(f"{len(missing_in_vocab)} genes from Aging-related DEGs are not in vocabulary. First 10: {missing_in_vocab[:10]}")

    logger.info(f"Filtering to {len(available_genes)} genes that are present in data, vocab, and Aging-related DEGs")
    logger.info(f"First 10 selected genes: {available_genes[:10]}")

    # Filter adata_test and adata_train to only include selected genes
    logger.info(f"Filtering adata_test and adata_train to {len(available_genes)} selected genes")
    adata_test = adata_test[:, adata_test.var_names.isin(available_genes)].copy()
    adata_train = adata_train[:, adata_train.var_names.isin(available_genes)].copy()

    # Ensure all selected genes are present
    missing_genes_test = set(available_genes) - set(adata_test.var_names)
    missing_genes_train = set(available_genes) - set(adata_train.var_names)
    if len(missing_genes_test) > 0:
        logger.warning(f"{len(missing_genes_test)} genes are missing from adata_test after filtering")
    if len(missing_genes_train) > 0:
        logger.warning(f"{len(missing_genes_train)} genes are missing from adata_train after filtering")

    # Update available_genes to only include genes that are actually in both datasets
    available_genes = [g for g in available_genes if g in adata_test.var_names and g in adata_train.var_names]
    logger.info(f"Using {len(available_genes)} genes after filtering both datasets")

    # Load model config
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(f"Loading model config from {model_config_file}")
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    # Set up preprocessor
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=filter_gene_by_counts,
        filter_cell_by_counts=True,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )


    input_layer_key = {
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]

    age_labels = adata_test.obs["age_id"].tolist()
    age_labels = np.array(age_labels)

    batch_ids = adata_test.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    num_types = len(adata_test.obs["age_category"].unique())

    vocab.set_default_index(vocab["<pad>"])
    genes = adata_test.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create encoder model (frozen scGPT)
    ntokens = len(vocab)
    encoder_model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=0,  # No CLS head in encoder
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=False,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=False,
    )

    # Load scGPT encoder weights
    try:
        encoder_model.load_state_dict(torch.load(scgpt_model_file, map_location=device))
        logger.info(f"Loading all encoder params from {scgpt_model_file}")
    except Exception as e:
        logger.warning(f"Error loading full encoder: {e}, trying partial load")
        model_dict = encoder_model.state_dict()
        pretrained_dict = torch.load(scgpt_model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        encoder_model.load_state_dict(model_dict)

    # Freeze encoder
    for name, para in encoder_model.named_parameters():
        para.requires_grad = False

    encoder_model.to(device)
    encoder_model.eval()

    # Load ImprovedCls head
    cls_decoder_file = CLS_DECODER_SAVE_DIR / "best_model.pt"
    if not cls_decoder_file.exists():
        logger.error(f"ImprovedCls file not found: {cls_decoder_file}. Skipping tissue {TISSUE}.")
        continue

    # Check if we need to determine use_bce from results.json
    results_file = CLS_DECODER_SAVE_DIR / "results.json"
    use_bce = False
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
            config_from_file = results.get("config", {})
            use_bce = config_from_file.get("use_bce_loss", False) and num_types == 2
            logger.info(f"Loaded use_bce={use_bce} from results.json")
    else:
        # Default: assume BCE for binary classification
        use_bce = num_types == 2
        logger.info(f"results.json not found, defaulting to use_bce={use_bce} for binary classification")

    # Create ImprovedCls
    cls_decoder = ImprovedCls(
        d_model=embsize,
        n_cls=num_types,
        hidden_dim=1024,
        nlayers=5,
        activation=nn.GELU,
        dropout=0.1,
        use_residual=True,
        use_batch_norm=False,
    ).to(device)

    # Load ImprovedCls weights
    try:
        cls_decoder.load_state_dict(torch.load(cls_decoder_file, map_location=device))
        logger.info(f"Loading ImprovedCls from {cls_decoder_file}")
    except Exception as e:
        logger.error(f"Error loading ImprovedCls: {e}")
        continue

    cls_decoder.to(device)
    cls_decoder.eval()

    # Filter data to only include selected genes
    logger.info(f"Filtering data to {len(available_genes)} selected genes")
    adata_train_raw = adata_train_raw[:, adata_train_raw.var_names.isin(available_genes)].copy()
    adata_test_raw = adata_test_raw[:, adata_test_raw.var_names.isin(available_genes)].copy()

    # Ensure all selected genes are present
    missing_genes = set(available_genes) - set(adata_train_raw.var_names)
    if len(missing_genes) > 0:
        logger.warning(f"{len(missing_genes)} genes are missing from filtered data")
        available_genes = [g for g in available_genes if g in adata_train_raw.var_names]
        logger.info(f"Using {len(available_genes)} genes after filtering")

    # Calculate max expression for each gene from combined train + test RAW data (once, before all runs)
    logger.info("Calculating max expression values for each gene from train + test data (raw)")
    combined_adata = sc.concat([adata_train_raw, adata_test_raw], axis=0, join='inner')

    # Get max expression for each gene
    if issparse(combined_adata.X):
        max_expressions = combined_adata.X.max(axis=0).A.flatten()
    else:
        max_expressions = combined_adata.X.max(axis=0)

    max_expression_dict = dict(zip(combined_adata.var_names, max_expressions))
    logger.info(f"Calculated max expressions for {len(max_expression_dict)} genes")

    # Run perturbation analysis 10 times with different cell subsets
    n_runs = 10
    all_raw_dfs = []

    for run_idx in range(n_runs):
        logger.info(f"Starting run {run_idx + 1}/{n_runs} with random_state={run_idx}")
        
        # Create a separate random state for this run to ensure reproducibility
        rng = np.random.RandomState(run_idx)
        
        # Stratified sampling of 100 cells by age_category from RAW data
        # Use run_idx as random_state to get different subsets each time
        age_categories = adata_test_raw.obs['age_category'].unique()
        sampled_indices = []

        for age_cat in age_categories:
            age_mask = adata_test_raw.obs['age_category'] == age_cat
            age_indices = adata_test_raw.obs_names[age_mask].tolist()
            
            # Calculate how many cells to sample from this age group
            # Proportional sampling to get total of 100 cells
            n_per_age = max(1, int(N_CELLS_TO_SAMPLE / len(age_categories)))
            n_per_age = min(n_per_age, len(age_indices))
            
            # Random sample with different seed for each run
            sampled_age_indices = rng.choice(age_indices, size=n_per_age, replace=False)
            sampled_indices.extend(sampled_age_indices.tolist())

        # If we have less than 100, sample more from the largest group
        if len(sampled_indices) < N_CELLS_TO_SAMPLE:
            remaining = N_CELLS_TO_SAMPLE - len(sampled_indices)
            # Get the largest age group
            age_counts = adata_test_raw.obs['age_category'].value_counts()
            largest_age = age_counts.index[0]
            age_mask = (adata_test_raw.obs['age_category'] == largest_age) & (~adata_test_raw.obs_names.isin(sampled_indices))
            age_indices = adata_test_raw.obs_names[age_mask].tolist()
            if len(age_indices) >= remaining:
                additional_indices = rng.choice(age_indices, size=remaining, replace=False)
                sampled_indices.extend(additional_indices.tolist())

        sampled_adata = adata_test_raw[sampled_indices].copy()
        logger.info(f"Run {run_idx + 1}: Sampled {len(sampled_adata)} cells")
        logger.info(f"Run {run_idx + 1}: Age distribution: {sampled_adata.obs['age_category'].value_counts().to_dict()}")

        # Run perturbation analysis for this subset
        logger.info(f"Run {run_idx + 1}: Starting perturbation analysis for {len(sampled_adata.var_names)} genes")
        raw_df = pd.DataFrame({
            'cells': sampled_adata.obs.index,
            'age_category': sampled_adata.obs['age_category'],
            'age_id': sampled_adata.obs['age_id'],
            'run_id': run_idx,  # Track which run this data comes from
        })

        for gene_idx, gene_name in tqdm(enumerate(sampled_adata.var_names), 
                                            desc=f"Run {run_idx + 1}: Genes",
                                            total=len(sampled_adata.var_names)):
            # Get max expression for this gene
            max_expr = max_expression_dict.get(gene_name, 10000)
            
            # Create perturbed adatas from RAW data (before preprocessing)
            adata_zero = create_zero_perturbed_adata(sampled_adata, gene_name)
            adata_max = create_perturbed_adata(sampled_adata, gene_name, max_expr)
            
            # Ensure age annotations are present
            adata_zero.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_zero.obs['age_category']]
            adata_max.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_max.obs['age_category']]
            adata_zero.obs["batch_id"] = 0
            adata_max.obs["batch_id"] = 0
            adata_zero.var["gene_name"] = adata_zero.var_names
            adata_max.var["gene_name"] = adata_max.var_names
            
            # Get gene_ids for both adatas (they should have the same genes now)
            genes_list = adata_zero.var_names.tolist()
            gene_ids_perturbed = np.array(vocab(genes_list), dtype=int)
            
            # Run predictions
            result_zero = model_predict(encoder_model, cls_decoder, adata_zero, gene_ids_perturbed, use_bce=use_bce)
            result_max = model_predict(encoder_model, cls_decoder, adata_max, gene_ids_perturbed, use_bce=use_bce)
            
            # Store results
            raw_df[f'{gene_name}_zero'] = result_zero['probas']
            raw_df[f'{gene_name}_max'] = result_max['probas']
            
            # Calculate means for all genes
            zero_mean = result_zero['probas'].mean()
            max_mean = result_max['probas'].mean()
            
            # Print for every gene
            print(f'Run {run_idx + 1}, Gene {gene_idx + 1}/{len(sampled_adata.var_names)}: {gene_name}')
            print(f'  Максимальная экспрессия (max_expr): {max_expr:.4f}')
            print(f'  Zero mean: {zero_mean:.4f}')
            print(f'  Max mean: {max_mean:.4f}')
            print()
            
            # Log for every gene (not just every 50)
            logger.info(f'Run {run_idx + 1}, Gene {gene_idx + 1}/{len(sampled_adata.var_names)}: {gene_name} - Zero mean: {zero_mean:.4f}, Max mean: {max_mean:.4f}, Max expr: {max_expr:.4f}')
            
            # Save intermediate results every 50 genes
            if (gene_idx + 1) % 50 == 0:
                raw_df.to_csv(save_dir / f'gene_results_run{run_idx + 1}_intermediate.csv', index=False)

        # Save final results for this run
        raw_df.to_csv(save_dir / f'gene_results_run{run_idx + 1}.csv', index=False)
        all_raw_dfs.append(raw_df)
        logger.info(f"Run {run_idx + 1} completed. Saved to {save_dir / f'gene_results_run{run_idx + 1}.csv'}")

    # Combine all runs into a single dataframe
    if len(all_raw_dfs) > 0:
        combined_df = pd.concat(all_raw_dfs, ignore_index=True)
        combined_df.to_csv(save_dir / 'gene_results_all_runs.csv', index=False)
        logger.info(f"Combined all {n_runs} runs. Total rows: {len(combined_df)}. Saved to {save_dir / 'gene_results_all_runs.csv'}")

    logger.info(f"Completed processing tissue: {TISSUE}")
    logger.info("=" * 80)

logger.info("All tissues processed successfully!")

# %%

