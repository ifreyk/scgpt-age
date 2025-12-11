# %%
"""
Sanity Check (a): Переобучение на крошечном куске данных

Берёт 2000 клеток и запускает ту же модель с:
- freeze=True
- epochs=50
- без early stopping
- без шедулеров

Если:
- train acc не лезет выше 0.7–0.75 и loss болтается ~0.65–0.68 → что-то сломано в самой постановке задачи/коде
- модель легко уходит в overfit (acc→0.95+, loss→0.1) → значит, всё ок с кодом

Запуск:
    python src/sanity_check_a_tiny_training.py

Результаты сохраняются в: save/sanity_check_a_tiny_training/
"""
import copy
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from tqdm import tqdm
import warnings
import pandas as pd
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed
from sklearn.metrics import accuracy_score

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")
from scanpy.get import _get_obs_rep, _set_obs_rep

# Import Preprocessor from main script
from train_scgpt_multi_tissue import Preprocessor

SCGPT_MODEL_PATH = "src/data/models/scGPT_human"
DATA_DIR = Path("src/data/donor_divided")
SAVE_DIR = Path("save/sanity_check_a_tiny_training")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration for sanity check
config = dict(
    seed=0,
    load_model="src/data/models/scGPT_human",
    mask_ratio=0.0,
    epochs=50,  # Fixed 50 epochs
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    lr=1e-3,
    batch_size=32,
    layer_size=128,
    nlayers=12,
    nhead=8,
    dropout=0.1,
    weight_decay=1e-5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=True,  # MUST be True
    DSBN=False,
    # NO early stopping
    # NO schedulers
    n_cells=2000,  # Use only 2000 cells
)

# Set up logger
logger = scg.logger
log_file = SAVE_DIR / "run.log"
for handler in logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)
scg.utils.add_file_handler(logger, log_file)
logger.info("="*80)
logger.info("SANITY CHECK (a): Tiny Training on 2000 cells")
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

lr = config["lr"]
batch_size = config["batch_size"]
eval_batch_size = config["batch_size"]
epochs = config["epochs"]  # Fixed 50 epochs, no early stopping

fast_transformer = config["fast_transformer"]
fast_transformer_backend = "flash"
embsize = config["layer_size"]
d_hid = config["layer_size"]
nlayers = config["nlayers"]
nhead = config["nhead"]
dropout = config["dropout"]
weight_decay = config.get("weight_decay", 1e-4)

log_interval = 10  # More frequent logging for sanity check

# Input/output setup
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

# SUBSET TO 2000 CELLS
n_cells = config["n_cells"]
if adata_train.shape[0] > n_cells:
    logger.info(f"Subsetting training data from {adata_train.shape[0]} to {n_cells} cells")
    # Randomly sample n_cells
    np.random.seed(seed)
    indices = np.random.choice(adata_train.shape[0], size=n_cells, replace=False)
    adata_train = adata_train[indices].copy()
    logger.info(f"After subsetting: {adata_train.shape[0]} cells, {adata_train.shape[1]} genes")

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
valid_data, valid_age_labels, valid_batch_labels = return_data_age_batch(adata_test)

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
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)

logger.info(
    f"train set: {tokenized_train['genes'].shape[0]} samples, "
    f"feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set: {tokenized_valid['genes'].shape[0]} samples, "
    f"feature length: {tokenized_valid['genes'].shape[1]}"
)

def prepare_data(sort_seq_batch=False):
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    tensor_age_labels_train = torch.from_numpy(train_age_labels).long()
    tensor_age_labels_valid = torch.from_numpy(valid_age_labels).long()

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "age_labels": tensor_age_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "age_labels": tensor_age_labels_valid,
    }
    return train_data_pt, valid_data_pt

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
    intra_domain_shuffle: bool = False,
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

# Freeze encoder (MUST be True for sanity check)
pre_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)

for name, para in model.named_parameters():
    if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
        logger.info(f"Freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)

logger.info(f"Pre-freeze trainable params: {pre_freeze_param_count:,}")
logger.info(f"Post-freeze trainable params: {post_freeze_param_count:,}")

model.to(device)

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params, lr=lr, weight_decay=weight_decay, eps=1e-4 if config["amp"] else 1e-8
)
# NO SCHEDULER - fixed learning rate

scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

def train(model: nn.Module, loader: DataLoader) -> dict:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_acc = 0.0
    total_num = 0

    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        age_labels = batch_data["age_labels"].to(device)

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

            loss = 0.0
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], age_labels)
                loss = loss + loss_cls
                accuracy = (output_dict["cls_output"].argmax(1) == age_labels).sum().item() / age_labels.size(0)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls += loss_cls.item() if CLS else 0.0
        total_acc += accuracy if CLS else 0.0
        total_num += 1

    return {
        "loss": total_loss / total_num,
        "cls_loss": total_cls / total_num,
        "acc": total_acc / total_num,
    }

def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_num = 0

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            age_labels = batch_data["age_labels"].to(device)

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
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, age_labels)
                accuracy = (output_values.argmax(1) == age_labels).sum().item() / len(input_gene_ids)

            total_loss += loss.item()
            total_acc += accuracy
            total_num += 1

    return {
        "loss": total_loss / total_num,
        "acc": total_acc / total_num,
    }

# Training loop - NO early stopping, fixed 50 epochs
logger.info("="*80)
logger.info("STARTING TRAINING - 50 epochs, NO early stopping, NO schedulers")
logger.info("="*80)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in tqdm(range(1, epochs + 1)):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data()
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    train_metrics = train(model, train_loader)
    val_metrics = evaluate(model, valid_loader)

    train_losses.append(train_metrics["loss"])
    train_accs.append(train_metrics["acc"])
    val_losses.append(val_metrics["loss"])
    val_accs.append(val_metrics["acc"])

    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| epoch {epoch:3d}/{epochs} | time: {elapsed:5.2f}s | "
        f"train loss {train_metrics['loss']:5.4f} | train acc {train_metrics['acc']:5.4f} | "
        f"val loss {val_metrics['loss']:5.4f} | val acc {val_metrics['acc']:5.4f}"
    )
    logger.info("-" * 89)

# Final analysis
final_train_acc = train_accs[-1]
final_train_loss = train_losses[-1]
max_train_acc = max(train_accs)
min_train_loss = min(train_losses)

logger.info("="*80)
logger.info("SANITY CHECK RESULTS")
logger.info("="*80)
logger.info(f"Final train acc: {final_train_acc:.4f}")
logger.info(f"Final train loss: {final_train_loss:.4f}")
logger.info(f"Max train acc: {max_train_acc:.4f}")
logger.info(f"Min train loss: {min_train_loss:.4f}")

# Diagnosis
if final_train_acc < 0.75 and final_train_loss > 0.65:
    logger.warning("="*80)
    logger.warning("⚠️  PROBLEM DETECTED: Train acc < 0.75 and loss > 0.65")
    logger.warning("This suggests something is broken in the task setup/code:")
    logger.warning("  - Check labels (label_key, age_id mapping)")
    logger.warning("  - Check classification head")
    logger.warning("  - Check data preprocessing")
    logger.warning("="*80)
    diagnosis = "BROKEN"
elif max_train_acc > 0.95 and min_train_loss < 0.1:
    logger.info("="*80)
    logger.info("✓ SANITY CHECK PASSED: Model easily overfits")
    logger.info("This means the code is working correctly!")
    logger.info("On full dataset, you can now tune regularization, LR, etc.")
    logger.info("="*80)
    diagnosis = "OK"
else:
    logger.info("="*80)
    logger.info("? INTERMEDIATE RESULT: Model shows some learning but not clear overfit")
    logger.info(f"Max train acc: {max_train_acc:.4f}, Min train loss: {min_train_loss:.4f}")
    logger.info("="*80)
    diagnosis = "UNCLEAR"

# Save results
results = {
    "diagnosis": diagnosis,
    "final_train_acc": float(final_train_acc),
    "final_train_loss": float(final_train_loss),
    "max_train_acc": float(max_train_acc),
    "min_train_loss": float(min_train_loss),
    "train_accs": train_accs,
    "train_losses": train_losses,
    "val_accs": val_accs,
    "val_losses": val_losses,
    "config": config,
}

with open(SAVE_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label="Train Acc", linewidth=2)
plt.plot(val_accs, label="Val Acc", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / "training_curves.png", dpi=150, bbox_inches='tight')
plt.close()

logger.info(f"Results saved to {SAVE_DIR}")
logger.info(f"Diagnosis: {diagnosis}")

