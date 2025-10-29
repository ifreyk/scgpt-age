# scGPT Age Prediction

A machine learning project for age prediction using single-cell RNA sequencing data with scGPT (single-cell GPT) transformer models.

## Overview

This project implements age prediction models using single-cell RNA sequencing (scRNA-seq) data. It leverages the scGPT framework to train transformer-based models that can predict cellular age from gene expression profiles.

## Features

- **Age Prediction Models**: Train scGPT models for cellular age prediction
- **Subset Training**: Support for training on subsampled datasets
- **Comprehensive Evaluation**: Multiple metrics for model performance assessment
- **Data Preprocessing**: Built-in data loading and preprocessing utilities
- **Visualization**: Plotting and visualization tools for results analysis

## Project Structure

```
scgpt-age/
├── main.py                    # Main entry point
├── pyproject.toml            # Project dependencies and configuration
├── uv.lock                   # Locked dependency versions
├── src/
│   ├── train_scgpt.py        # Main training script
│   ├── train_scgpt_subset.py # Subset training script
│   └── data/                 # Data directory
│       ├── ageAnno_subsampled_data_v2.h5ad.gz
│       ├── train_data_subsampled_cells.h5ad.gz
│       ├── test_data_subsampled_cells.h5ad.gz
│       ├── models/
│       │   └── scGPT_human/
│       │       ├── args.json
│       │       ├── best_model.pt
│       │       └── vocab.json
│       └── run_data/
└── save/                     # Model checkpoints and logs
    └── dev_AgeAnno_finall-*/
        ├── run.log
        └── vocab.json
```

## Installation

### Prerequisites

- Python >= 3.11
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd scgpt-age
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### Key Dependencies

- `scgpt>=0.2.4` - Core scGPT framework
- `torch>=2.0.0` - PyTorch for deep learning
- `scvi-tools>=0.20.0,<1.0.0` - Single-cell variational inference
- `anndata>=0.10.9` - Annotated data arrays
- `scanpy` - Single-cell analysis in Python
- `flash-attn<1.0.5` - Flash attention for efficient training

## Usage

### Basic Training

Run the main training script:
```bash
python src/train_scgpt.py
```

### Subset Training

For training on subsampled data:
```bash
python src/train_scgpt_subset.py
```

### Configuration

The training scripts support various configuration options including:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing options
- Evaluation metrics

## Data

The project expects data in AnnData format (.h5ad files). The data directory contains:
- Training and test datasets
- Pre-trained model checkpoints
- Vocabulary files for gene tokenization

### Data Format

- **Input**: Single-cell RNA-seq data in AnnData format
- **Features**: Gene expression counts
- **Target**: Age annotations for cells

## Model Architecture

The project uses scGPT transformer models with:
- Gene tokenization for expression data
- Masked language modeling objectives
- Adversarial training components
- Age prediction heads

## Training Process

1. **Data Loading**: Load and preprocess scRNA-seq data
2. **Tokenization**: Convert gene expression to tokens
3. **Model Training**: Train transformer with masked objectives
4. **Evaluation**: Assess age prediction performance
5. **Saving**: Store model checkpoints and results

## Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Pearson correlation coefficient
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

## Results

Model checkpoints and training logs are saved in the `save/` directory with timestamps. Each run includes:
- Model weights (`best_model.pt`)
- Training arguments (`args.json`)
- Vocabulary file (`vocab.json`)
- Training logs (`run.log`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information]
```

## Acknowledgments

- scGPT framework for single-cell transformer models
- Single-cell analysis community for tools and datasets

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Flash attention installation**: Ensure compatible CUDA version
3. **Data loading errors**: Check AnnData file format and paths

### Getting Help

- Check the logs in `save/` directory for detailed error messages
- Ensure all dependencies are properly installed
- Verify data format compatibility

## Future Work

- [ ] Support for additional model architectures
- [ ] Integration with more single-cell datasets
- [ ] Enhanced visualization tools
- [ ] Distributed training support
