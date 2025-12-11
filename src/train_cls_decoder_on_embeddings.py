# %%
"""
Train separate ImprovedCls heads on scGPT embeddings for each tissue.

Based on sanity_check_b_embedding_classifier.py structure, but uses ImprovedCls
instead of sklearn classifiers.

For each tissue:
1. Load pre-trained scGPT model (frozen)
2. Extract cell embeddings from train and test data
3. Train a separate ImprovedCls head on embeddings
4. Save head weights and metrics
"""
#%%
import copy
import json
import logging
from tqdm import tqdm
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
from torch.nn import functional as F
from typing import List, Tuple, Dict, Union, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix as sk_confusion_matrix,
)

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from improved_cls_decoder import ImprovedCls, SimpleCls
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed
from scanpy.get import _get_obs_rep, _set_obs_rep

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

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
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
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
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=(
                    self.filter_gene_by_counts
                    if isinstance(self.filter_gene_by_counts, int)
                    else None
                ),
            )

        # step 2: filter cells
        if (
            isinstance(self.filter_cell_by_counts, int)
            and self.filter_cell_by_counts > 0
        ):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=(
                    self.filter_cell_by_counts
                    if isinstance(self.filter_cell_by_counts, int)
                    else None
                ),
            )

        # step 3: normalize total
        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=(
                    self.normalize_total
                    if isinstance(self.normalize_total, float)
                    else None
                ),
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
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

        # step 5: subset hvg
        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=(
                    self.subset_hvg if isinstance(self.subset_hvg, int) else None
                ),
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

        # step 6: binning
        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
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
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
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
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
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

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    np.random.seed(0)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


# Configuration
SCGPT_MODEL_PATH = "src/data/models/scGPT_human"
DATA_DIR = Path("src/data/donor_divided")
SAVE_DIR = Path("save/cls_decoder_on_embeddings_2500_cells")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = [
    "skin",
    #"bladder",
    "blood",
    #"bone-marrow",
    #"brain",
    #"liver",
    #"lung",
    #"pancreas",
    #"skeletal-muscle",
    #"stomach",
]
#%%
# Configuration
config = dict(
    seed=42,
    load_model="src/data/models/scGPT_human",
    mask_ratio=0.0,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    layer_size=128,
    nlayers=12,
    nhead=8,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=True,  # MUST be True - frozen model
    DSBN=False,
    # ImprovedCls training config
    batch_size=256,
    lr=1e-4,
    epochs=20,
    weight_decay=1e-7,
    label_smoothing=0.15,
    use_bce_loss=False,  # Use BCEWithLogitsLoss for binary classification
    hidden_dim=1024,  # Hidden dimension (None = same as d_model)
    dropout=0.4,
    min_lr=1e-6
)
# Set up logger
logger = scg.logger
log_file = SAVE_DIR / "run.log"
for handler in logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)
scg.utils.add_file_handler(logger, log_file)
logger.info("="*80)
logger.info("Training ImprovedCls heads on scGPT embeddings")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load aging-related DEGs for filtering
ageanno_genes_path = Path("src/data/Aging-related DEGs.txt")
if not ageanno_genes_path.exists():
    logger.error(f"Aging-related DEGs file not found: {ageanno_genes_path}")
    sys.exit(1)

ageanno_genes = pd.read_csv("src/data/Aging-related DEGs.txt", encoding='iso-8859-1')
ageanno_genes = ageanno_genes[ageanno_genes['group']=='old vs mid']
# ageanno_genes = ageanno_genes[ageanno_genes['p_val_adj']<0.05]
# ageanno_genes = ageanno_genes[(ageanno_genes['isTissuespecific']==False) & (ageanno_genes['isCellTypespecific']==False)]
unique_genes = ageanno_genes['gene'].unique().tolist()
logger.info(f"Loaded {len(unique_genes)} aging-related DEGs for filtering")

# Load vocab and model config
model_dir = Path(SCGPT_MODEL_PATH)
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

with open(model_config_file, "r") as f:
    model_configs = json.load(f)

logger.info(f"Loading model from {model_file}")
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]
#%%
# Create model
ntokens = len(vocab)
num_types = 2  # binary classification: old vs mid

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=0,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=1,
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
#%%
# Load model weights
try:
    model.load_state_dict(torch.load(model_file, map_location=device))
    logger.info(f"Loading all model params from {model_file}")
except:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    logger.info(f"Loading {len(pretrained_dict)} compatible parameters out of {len(torch.load(model_file, map_location=device).keys())} total")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Freeze encoder (MUST be True)
for name, para in model.named_parameters():
    if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
        para.requires_grad = False

model.to(device)
model.eval()  # Set to eval mode for embedding extraction

logger.info("Model loaded and frozen successfully")


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


def extract_embeddings(model, data_pt, age_labels, batch_size=64):
    """Extract cell embeddings from frozen model"""
    model.eval()
    embeddings = []
    labels = []
    
    loader = prepare_dataloader(data_pt, batch_size=batch_size, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Extracting embeddings"):
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


def train_cls_decoder(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    embsize,
    n_classes,
    config,
):
    """Train ImprovedCls head on embeddings, validate on test set"""
    logger.info("Training ImprovedCls head...")

    # # Create model with improved architecture
    # cls_decoder = ImprovedCls(
    #     d_model=embsize,
    #     n_cls=n_classes,
    #     nlayers=5,
    #     activation=nn.SiLU,
    #     dropout=config.get("dropout", 0.1),
    #     use_residual=config.get("use_residual", True),
    #     use_batch_norm=config.get("use_batch_norm", False),
    #     hidden_dim=config.get("hidden_dim", None),
    # ).to(device)
    cls_decoder = ImprovedCls(embsize, n_classes, dropout=config.get("dropout", 0.2),
                              nlayers=5, activation=nn.GELU, use_residual=True, use_batch_norm=False, hidden_dim=1024).to(device)
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_embeddings).float(),
        torch.from_numpy(train_labels).long(),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_embeddings).float(),
        torch.from_numpy(test_labels).long(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Loss and optimizer
    use_bce = config.get("use_bce_loss", False) and n_classes == 2
    if use_bce:
        # Calculate class weights for balanced loss
        pos_count = np.sum(train_labels == 1)
        neg_count = np.sum(train_labels == 0)
        if pos_count > 0 and neg_count > 0:
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
            logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f} (pos: {pos_count}, neg: {neg_count})")
        else:
            pos_weight = None
            logger.info("Using BCEWithLogitsLoss without class weights")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Using BCEWithLogitsLoss for binary classification")
    else:
        criterion = nn.CrossEntropyLoss()
            # label_smoothing=config["label_smoothing"]
            # )
        logger.info("Using CrossEntropyLoss")
    
    optimizer = torch.optim.AdamW(
        cls_decoder.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True,min_lr=config["min_lr"]
    )

    # Training loop
    best_test_auc = 0.0
    best_test_loss = float("inf")
    best_model_state = None

    for epoch in range(config["epochs"]):
        # Train
        cls_decoder.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = cls_decoder(batch_emb)
            
            if use_bce:
                # For BCE: use logits for class 1, convert labels to float
                logits_bce = logits[:, 1]
                labels_bce = batch_labels.float()
                loss = criterion(logits_bce, labels_bce)
            else:
                loss = criterion(logits, batch_labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if use_bce:
                # For BCE: apply sigmoid and threshold at 0.5
                probs = torch.sigmoid(logits[:, 1])
                preds = (probs > 0.5).long()
            else:
                preds = logits.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += len(batch_labels)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Evaluate on test set
        cls_decoder.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_test_probs = []
        all_test_labels = []

        with torch.no_grad():
            for batch_emb, batch_labels in test_loader:
                batch_emb = batch_emb.to(device)
                batch_labels = batch_labels.to(device)

                logits = cls_decoder(batch_emb)
                
                if use_bce:
                    # For BCE: use logits for class 1, convert labels to float
                    logits_bce = logits[:, 1]
                    labels_bce = batch_labels.float()
                    loss = criterion(logits_bce, labels_bce)
                    # For BCE: apply sigmoid to get probabilities
                    probs_bce = torch.sigmoid(logits_bce)
                    probs = torch.stack([1 - probs_bce, probs_bce], dim=1)
                    preds = (probs_bce > 0.5).long()
                else:
                    loss = criterion(logits, batch_labels)
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                test_loss += loss.item()
                test_correct += (preds == batch_labels).sum().item()
                test_total += len(batch_labels)

                all_test_probs.append(probs.cpu().numpy())
                all_test_labels.append(batch_labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total

        all_test_probs = np.concatenate(all_test_probs, axis=0)
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        test_auc = roc_auc_score(all_test_labels, all_test_probs[:, 1])

        scheduler.step(test_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}"
            )

        # Track best model based on test AUC (higher is better)
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_loss = test_loss
            best_model_state = copy.deepcopy(cls_decoder.state_dict())

    # Load best model
    if best_model_state is not None:
        cls_decoder.load_state_dict(best_model_state)
    else:
        # If no best model was found (shouldn't happen), use current state
        best_model_state = cls_decoder.state_dict()

    return cls_decoder, best_model_state, {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_auc": best_test_auc,
        "test_loss": best_test_loss,
    }


def evaluate_cls_decoder(cls_decoder, test_embeddings, test_labels, device, use_bce=False):
    """Evaluate ImprovedCls on test set"""
    cls_decoder.eval()

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_embeddings).float(),
        torch.from_numpy(test_labels).long(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_emb, batch_labels in test_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            logits = cls_decoder(batch_emb)
            
            if use_bce:
                # For BCE: apply sigmoid to get probabilities
                probs_bce = torch.sigmoid(logits[:, 1])
                probs = torch.stack([1 - probs_bce, probs_bce], dim=1)
                preds = (probs_bce > 0.5).long()
            else:
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    cm = sk_confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "labels": all_labels.tolist(),
    }


def plot_roc_curve(y_true, y_scores, roc_auc, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=3, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC = 0.50)"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    plt.title("ROC Curve", fontsize=16, fontweight="bold", pad=20)
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

#%%
# Main execution - process each tissue
results_summary = []

for tissue in TISSUES:
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing tissue: {tissue}")
    logger.info(f"{'='*80}")

    # Create save directory
    save_dir = SAVE_DIR / tissue
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = DATA_DIR / f"subset_edited_genes_{tissue}_ageanno_train.h5ad.gz"
    test_path = DATA_DIR / f"subset_edited_genes_{tissue}_ageanno_test.h5ad.gz"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Data files not found for {tissue}, skipping...")
        continue

    logger.info(f"Loading data from {train_path} and {test_path}")
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

    # Filter to aging-related DEGs
    train_gene_names = set(adata_train.var_names)
    test_gene_names = set(adata_test.var_names)
    available_genes = list(set(unique_genes) & train_gene_names & test_gene_names)

    if len(available_genes) == 0:
        logger.error(f"No aging-related DEGs found in the data for {tissue}, skipping...")
        continue

    logger.info(f"Filtering to {len(available_genes)} aging-related DEGs")
    adata_train = adata_train[:, available_genes]
    adata_test = adata_test[:, available_genes]

    # Check gene names in vocab
    adata_train.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_train.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_train.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )

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

    # Prepare data
    train_data, train_age_labels, train_batch_labels = return_data_age_batch(adata_train)
    test_data, test_age_labels, test_batch_labels = return_data_age_batch(adata_test)

    genes = adata_train.var["gene_name"].tolist()
    
    # Save unique genes to CSV
    genes_df = pd.DataFrame({"gene": sorted(set(genes))})
    genes_file = save_dir / "genes_for_train.csv"
    genes_df.to_csv(genes_file, index=False)
    logger.info(f"Saved {len(genes_df)} unique genes to {genes_file}")
    
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    # Tokenize
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

    logger.info(f"Train set: {tokenized_train['genes'].shape[0]} samples")
    logger.info(f"Test set: {tokenized_test['genes'].shape[0]} samples")

    # Extract embeddings
    logger.info("Extracting train embeddings...")
    train_data_pt = prepare_data_for_embeddings(tokenized_train, train_age_labels, train_batch_labels)
    train_embeddings, train_labels = extract_embeddings(
        model, train_data_pt, train_age_labels, batch_size=config["batch_size"]
    )

    logger.info(f"Extracted train embeddings: {train_embeddings.shape}")
    logger.info(f"Train labels: {train_labels.shape}, class distribution: {np.bincount(train_labels)}")

    logger.info("Extracting test embeddings...")
    test_data_pt = prepare_data_for_embeddings(tokenized_test, test_age_labels, test_batch_labels)
    test_embeddings, test_labels = extract_embeddings(
        model, test_data_pt, test_age_labels, batch_size=config["batch_size"]
    )

    logger.info(f"Extracted test embeddings: {test_embeddings.shape}")

    logger.info(f"Train: {train_embeddings.shape[0]} samples")
    logger.info(f"Test: {test_embeddings.shape[0]} samples")

    # Train classifier head (train on full train set, validate on test set)
    n_classes = len(np.unique(train_labels))
    use_bce = config.get("use_bce_loss", False) and n_classes == 2
    cls_decoder, best_model_state, train_metrics = train_cls_decoder(
        train_embeddings, train_labels, test_embeddings, test_labels, embsize, n_classes, config
    )

    # Final evaluation on test set
    test_metrics = evaluate_cls_decoder(
        cls_decoder, test_embeddings, test_labels, device, use_bce=use_bce
    )

    # Save model weights (best model state)
    torch.save(best_model_state, save_dir / "best_model.pt")
    logger.info(f"Saved best model state to {save_dir / 'best_model.pt'}")
    
    # Also save current state for reference
    torch.save(cls_decoder.state_dict(), save_dir / "cls_decoder_weights.pt")
    logger.info(f"Saved current model weights to {save_dir / 'cls_decoder_weights.pt'}")

    # Save metrics
    results = {
        "tissue": tissue,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "config": config,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot ROC curve
    plot_roc_curve(
        test_metrics["labels"],
        np.array(test_metrics["probabilities"])[:, 1],
        test_metrics["roc_auc"],
        save_dir / "roc_curve.png",
    )

    logger.info(f"\n{tissue.upper()} Results:")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Test F1: {test_metrics['f1']:.4f}")

    results_summary.append(
        {
            "tissue": tissue,
            "test_accuracy": test_metrics["accuracy"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
        }
    )

    # Clean up memory
    del adata_train, adata_test, train_embeddings, test_embeddings
    del cls_decoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Save summary
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(SAVE_DIR / "summary.csv", index=False)
logger.info(f"\nSummary saved to {SAVE_DIR / 'summary.csv'}")
logger.info("\n" + summary_df.to_string(index=False))

logger.info("\n" + "=" * 80)
logger.info("Training complete!")
logger.info("=" * 80)

# %%
