# %%
"""
Sanity Check (b): Проверка предиктивного сигнала

Выгружает эмбеддинги клеток из scGPT (frozen).
На них запускает обычный LogisticRegression / XGBoost в sklearn.
Смотрит, какая там val_acc.

Если и классический ML даёт ~0.58–0.6, то, возможно, AgeAnno mid vs old 
на этих фичах просто плохо разделим. Тогда виноват не scGPT, а сами данные/формат задачи.

Запуск:
    python src/sanity_check_b_embedding_classifier.py

Результаты сохраняются в: save/sanity_check_b_embedding_classifier/
"""
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import warnings
import pandas as pd
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, will only use LogisticRegression")

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")
from scanpy.get import _get_obs_rep, _set_obs_rep

# Import Preprocessor from main script
from train_scgpt_multi_tissue import Preprocessor

SCGPT_MODEL_PATH = "src/data/models/scGPT_human"
DATA_DIR = Path("src/data/donor_divided")
SAVE_DIR = Path("save/sanity_check_b_embedding_classifier")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
config = dict(
    seed=0,
    load_model="src/data/models/scGPT_human",
    mask_ratio=0.0,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    layer_size=128,
    nlayers=12,
    nhead=8,
    dropout=0.1,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=True,  # MUST be True - frozen model
    DSBN=False,
    test_size=0.2,  # For train/val split of embeddings
)

# Set up logger
logger = scg.logger
log_file = SAVE_DIR / "run.log"
for handler in logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)
scg.utils.add_file_handler(logger, log_file)
logger.info("="*80)
logger.info("SANITY CHECK (b): Predictive Signal Check via Embeddings")
logger.info("="*80)
logger.info(f"Configuration: {json.dumps(config, indent=2)}")

# Set seed
seed = config["seed"]
set_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config["mask_ratio"]
mask_value = "auto"
include_zero_gene = config["include_zero_gene"]
max_seq_len = 2500
n_bins = config["n_bins"]
input_style = "binned"
output_style = "binned"
MLM = False
CLS = True
ADV = False
CCE = False
MVC = config["MVC"]
ECS = config["ecs_thres"] > 0
DAB = False
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "avg-pool"
explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False

fast_transformer = config["fast_transformer"]
fast_transformer_backend = "flash"
embsize = config["layer_size"]
d_hid = config["layer_size"]
nlayers = config["nlayers"]
nhead = config["nhead"]
dropout = config["dropout"]

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

filter_gene_by_counts = False
data_is_raw = True

# Load data - use first available tissue
tissue = "skin"  # Can be changed or made configurable
train_path = DATA_DIR / f"subset_edited_genes_{tissue}_ageanno_train.h5ad.gz"
test_path = DATA_DIR / f"subset_edited_genes_{tissue}_ageanno_test.h5ad.gz"

if not train_path.exists():
    logger.error(f"Training file not found: {train_path}")
    sys.exit(1)

logger.info(f"Loading data from {train_path}")
adata_train = sc.read_h5ad(train_path)
adata_test = sc.read_h5ad(test_path)

# Prepare data
adata_train.var["gene_name"] = adata_train.var_names
adata_test.var["gene_name"] = adata_test.var_names
adata_train.obs["age_category"] = [x[:3] for x in adata_train.obs['orig.ident']]
adata_test.obs["age_category"] = [x[:3] for x in adata_test.obs['orig.ident']]
adata_train.obs["batch_id"] = 0
adata_test.obs["batch_id"] = 0
adata_train.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_train.obs['age_category']]
adata_test.obs["age_id"] = [1 if x == 'old' else 0 for x in adata_test.obs['age_category']]

# Load aging-related DEGs
ageanno_genes_path = Path("src/data/Aging-related DEGs.txt")
if not ageanno_genes_path.exists():
    logger.error(f"Aging-related DEGs file not found: {ageanno_genes_path}")
    sys.exit(1)

ageanno_genes = pd.read_csv(ageanno_genes_path, encoding='iso-8859-1')
ageanno_genes = ageanno_genes[ageanno_genes['group']=='old vs mid']
unique_genes = ageanno_genes['gene'].unique().tolist()

train_gene_names = set(adata_train.var_names)
test_gene_names = set(adata_test.var_names)
available_genes = list(set(unique_genes) & train_gene_names & test_gene_names)

if len(available_genes) == 0:
    logger.error("No aging-related DEGs found in the data")
    sys.exit(1)

logger.info(f"Filtering to {len(available_genes)} aging-related DEGs")
adata_train = adata_train[:, available_genes]
adata_test = adata_test[:, available_genes]

# Load vocab and model
with open(f"{SCGPT_MODEL_PATH}/vocab.json", "r") as f:
    vocab = json.load(f)

model_dir = Path(f"{SCGPT_MODEL_PATH}")
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
shutil.copy(vocab_file, SAVE_DIR / "vocab.json")
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

adata_train.var["id_in_vocab"] = [
    1 if gene in vocab else -1 for gene in adata_train.var["gene_name"]
]
gene_ids_in_vocab = np.array(adata_train.var["id_in_vocab"])
logger.info(
    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
    f"in vocabulary of size {len(vocab)}."
)

with open(model_config_file, "r") as f:
    model_configs = json.load(f)
logger.info(f"Loading model from {model_file}")
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

# Preprocess
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

preprocessor(adata_train, batch_key=None)
preprocessor(adata_test, batch_key=None)

def return_data_age_batch(adata_to_use):
    input_layer_key = {
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]
    all_counts = (
        adata_to_use.layers[input_layer_key].A
        if issparse(adata_to_use.layers[input_layer_key])
        else adata_to_use.layers[input_layer_key]
    )
    age_labels = adata_to_use.obs["age_id"].tolist()
    age_labels = np.array(age_labels)
    batch_ids = adata_to_use.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
    return all_counts, age_labels, batch_ids

train_data, train_age_labels, train_batch_labels = return_data_age_batch(adata_train)
test_data, test_age_labels, test_batch_labels = return_data_age_batch(adata_test)

batch_ids = adata_train[adata_train.obs["batch_id"] == 0].obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
genes = adata_train.var["gene_name"].tolist()

vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
tokenized_test = tokenize_and_pad_batch(
    test_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)

logger.info(
    f"train set: {tokenized_train['genes'].shape[0]} samples"
)
logger.info(
    f"test set: {tokenized_test['genes'].shape[0]} samples"
)

def prepare_data_for_embeddings(tokenized_data, age_labels, batch_labels):
    masked_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    
    input_gene_ids = tokenized_data["genes"]
    input_values = masked_values
    
    tensor_batch_labels = torch.from_numpy(batch_labels).long()
    tensor_age_labels = torch.from_numpy(age_labels).long()
    
    data_pt = {
        "gene_ids": input_gene_ids,
        "values": input_values,
        "target_values": tokenized_data["values"],
        "batch_labels": tensor_batch_labels,
        "age_labels": tensor_age_labels,
    }
    return data_pt

class SeqDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def prepare_dataloader(
    data_pt: dict,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    dataset = SeqDataset(data_pt)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )
    return data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_types = len(adata_train.obs["age_category"].unique())
ntokens = len(vocab)

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=5,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config["DSBN"],
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style="inner product",
    ecs_threshold=config["ecs_thres"],
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config["pre_norm"],
)

try:
    model.load_state_dict(torch.load(model_file))
    logger.info(f"Loading all model params from {model_file}")
except:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Freeze encoder (MUST be True)
for name, para in model.named_parameters():
    if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
        logger.info(f"Freezing weights for: {name}")
        para.requires_grad = False

model.to(device)
model.eval()  # Set to eval mode for embedding extraction

logger.info("Extracting embeddings from frozen scGPT model...")

def extract_embeddings(model, data_pt, age_labels, batch_size=64):
    """Extract cell embeddings from frozen model"""
    model.eval()
    embeddings = []
    labels = []
    
    loader = prepare_dataloader(data_pt, batch_size=batch_size, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            age_labels_batch = batch_data["age_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config["amp"]):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config["DSBN"] else None,
                    CLS=CLS,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                )
                # Extract cell embeddings
                cell_emb = output_dict["cell_emb"]  # Shape: [batch_size, embsize]
                embeddings.append(cell_emb.cpu().numpy())
                labels.append(age_labels_batch.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

# Extract embeddings from training data
train_data_pt = prepare_data_for_embeddings(tokenized_train, train_age_labels, train_batch_labels)
train_embeddings, train_labels = extract_embeddings(model, train_data_pt, train_age_labels)

logger.info(f"Extracted train embeddings: {train_embeddings.shape}")
logger.info(f"Train labels: {train_labels.shape}, class distribution: {np.bincount(train_labels)}")

# Split train into train/val for classifier training
X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings, train_labels, 
    test_size=config["test_size"], 
    random_state=seed,
    stratify=train_labels
)

logger.info(f"Train split: {X_train.shape[0]} samples")
logger.info(f"Val split: {X_val.shape[0]} samples")

# Train classifiers
results = {}

# 1. Logistic Regression
logger.info("="*80)
logger.info("Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=seed,
    solver='lbfgs',
    n_jobs=-1
)
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_val_pred = lr_model.predict(X_val)
lr_train_proba = lr_model.predict_proba(X_train)[:, 1]
lr_val_proba = lr_model.predict_proba(X_val)[:, 1]

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_val_acc = accuracy_score(y_val, lr_val_pred)
lr_val_auc = roc_auc_score(y_val, lr_val_proba)

results["logistic_regression"] = {
    "train_acc": float(lr_train_acc),
    "val_acc": float(lr_val_acc),
    "val_auc": float(lr_val_auc),
}

logger.info(f"Logistic Regression - Train Acc: {lr_train_acc:.4f}, Val Acc: {lr_val_acc:.4f}, Val AUC: {lr_val_auc:.4f}")

# 2. XGBoost (if available)
if HAS_XGBOOST:
    logger.info("="*80)
    logger.info("Training XGBoost...")
    xgb_model = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False,
    )
    xgb_model.fit(X_train, y_train)

    xgb_train_pred = xgb_model.predict(X_train)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]

    xgb_train_acc = accuracy_score(y_train, xgb_train_pred)
    xgb_val_acc = accuracy_score(y_val, xgb_val_pred)
    xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)

    results["xgboost"] = {
        "train_acc": float(xgb_train_acc),
        "val_acc": float(xgb_val_acc),
        "val_auc": float(xgb_val_auc),
    }

    logger.info(f"XGBoost - Train Acc: {xgb_train_acc:.4f}, Val Acc: {xgb_val_acc:.4f}, Val AUC: {xgb_val_auc:.4f}")
else:
    logger.info("XGBoost not available, skipping")

# Diagnosis
best_val_acc = max(results["logistic_regression"]["val_acc"], 
                   results.get("xgboost", {}).get("val_acc", 0))

logger.info("="*80)
logger.info("SANITY CHECK RESULTS")
logger.info("="*80)
logger.info(f"Best validation accuracy (classical ML): {best_val_acc:.4f}")

if best_val_acc < 0.6:
    logger.warning("="*80)
    logger.warning("⚠️  LOW PREDICTIVE SIGNAL DETECTED")
    logger.warning(f"Classical ML gives val_acc ~{best_val_acc:.4f} (close to random ~0.5)")
    logger.warning("This suggests that AgeAnno mid vs old on these features is poorly separable.")
    logger.warning("The problem may not be with scGPT, but with the data/task format itself.")
    logger.warning("="*80)
    diagnosis = "LOW_SIGNAL"
elif best_val_acc >= 0.6:
    logger.info("="*80)
    logger.info("✓ PREDICTIVE SIGNAL EXISTS")
    logger.info(f"Classical ML achieves val_acc ~{best_val_acc:.4f}")
    logger.info("This suggests the embeddings contain useful information.")
    logger.info("If scGPT performs worse, the issue is likely with scGPT training, not the data.")
    logger.info("="*80)
    diagnosis = "SIGNAL_EXISTS"
else:
    diagnosis = "UNCLEAR"

results["diagnosis"] = diagnosis
results["best_val_acc"] = float(best_val_acc)
results["config"] = config

# Save results
with open(SAVE_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
methods = list(results.keys())
if "diagnosis" in methods:
    methods.remove("diagnosis")
if "best_val_acc" in methods:
    methods.remove("best_val_acc")
if "config" in methods:
    methods.remove("config")

train_accs = [results[m]["train_acc"] for m in methods]
val_accs = [results[m]["val_acc"] for m in methods]

x = np.arange(len(methods))
width = 0.35

axes[0].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, val_accs, width, label='Val', alpha=0.8)
axes[0].set_xlabel('Method')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Classification Accuracy')
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
axes[0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Threshold')

# AUC comparison
val_aucs = [results[m]["val_auc"] for m in methods]
axes[1].bar(methods, val_aucs, alpha=0.8)
axes[1].set_xlabel('Method')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_title('Validation ROC-AUC')
axes[1].grid(alpha=0.3, axis='y')
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
axes[1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig(SAVE_DIR / "classification_results.png", dpi=150, bbox_inches='tight')
plt.close()

logger.info(f"Results saved to {SAVE_DIR}")
logger.info(f"Diagnosis: {diagnosis}")

