# Discovery of Age-Associated Genes Using Large Transcriptome Foundation Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of a fine-tuned scGPT model for age prediction and systematic gene perturbation analysis to identify pro- and anti-aging genes. The work demonstrates the application of large transcriptomic foundation models to aging research, achieving strong predictive performance (ROC-AUC = 0.9) and providing a methodological framework for discovering novel aging-related genes.

## Authors

- **Erik Tadevosyan**¹ ([orcidA{}](https://orcid.org/0000-0000-0000-000X))
- **Evgeniy Efimov**¹²
- **Ekaterina Khrameeva**¹
- **Dmitrii Kriukov**²* ([kriukov@airi.net](mailto:kriukov@airi.net))

¹ Skolkovo Institute of Science and Technology, Moscow, Russia  
² Artificial Intelligence Research Institute, Moscow, Russia

*Corresponding author

## Abstract

Aging is a progressive functional decline driven by complex genetic, epigenetic, environmental, and stochastic interactions that traditional linear models fail to capture. Using human single-cell RNA-seq data from the multi-tissue AgeAnno dataset, we fine-tuned scGPT, a large transcriptomic model, to predict chronological age, achieving ROC-AUC of 0.9. To identify genes influencing age predictions, we systematically perturbed individual genes *in silico* and quantified their effects, classifying them as pro- or anti-aging. Several identified genes were absent from existing aging biomarker databases, highlighting them as potential novel candidates for future experimental validation as anti-aging interventions.

## Key Features

- **Age Prediction**: Fine-tuned scGPT model for binary age classification (mid-age vs old-age)
- **Gene Perturbation Analysis**: Systematic *in silico* perturbations to identify pro- and anti-aging genes
- **Multi-tissue Analysis**: Trained on diverse human tissues from the AgeAnno dataset
- **High Performance**: ROC-AUC of 0.9 across tissues
- **Stability Assessment**: Resampling-based validation of perturbation predictions

## Dataset

The analysis uses the **AgeAnno dataset** comprising scRNA-seq data from 13 human tissues across donors aged 20-100 years. Key preprocessing steps include:

- Log-normalization of raw count matrices
- Exclusion of genes expressed in <1% of cells
- Removal of mitochondrial, ribosomal, and hemoglobin genes
- Subsampling to 3,000 age-associated genes due to computational constraints

## Model Architecture

- **Base Model**: scGPT (Transformer-based large transcriptomic model)
- **Task**: Binary age classification (mid-age: 20-59 vs old-age: 60-100)
- **Hyperparameters**:
  - Epochs: 10
  - Learning rate: 1×10⁻⁴
  - Batch size: 10
  - Hidden layer size: 128
  - Transformer layers: 4
  - Attention heads: 4
  - Dropout: 0.4
  - Max sequence length: 3,001
  - Expression bins: 51

## Results

### Model Performance
- **Accuracy**: 0.832
- **Precision**: 0.839
- **Recall**: 0.825
- **ROC-AUC**: 0.9

### Perturbation Analysis
- Systematic gene-level perturbations (knockdown vs overexpression)
- Classification of genes as pro-aging or anti-aging based on prediction shifts
- Statistical significance testing with Holm correction
- Identification of novel candidate genes not present in existing aging databases

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended: NVIDIA RTX 4090 with 24GB memory)

### Dependencies
```bash
uv sync
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ifreyk/scgpt-age.git
cd scgpt-age
```

2. Install all dependencies:
```bash
uv sync
```

3. Install the package in development mode:
```bash
uv pip install -e .
```

## Usage

The pipeline consists of three main scripts that should be run sequentially:

### 1. Training with Age-Associated Genes (Recommended for Reproducibility)
```bash
python src/train_scgpt_diffexpressed_genes.py
```
This trains the scGPT model using differentially expressed genes identified in the AgeAnno study, providing the most reproducible results.

### 2. Training with Random Genes (Alternative Approach)
```bash
python src/train_scgpt_random_genes.py
```
Alternative training approach using randomly selected genes for comparison and robustness testing.

### 3. Gene Perturbation Analysis
```bash
python src/perturbation_analysis.py
```
Performs systematic *in silico* gene perturbations to identify pro- and anti-aging genes. This script automatically loads the trained model and performs the perturbation analysis.

## Project Structure

```
scgpt-age/
├── src/
│   ├── data/
│   │   ├── models/                   # Pretrained models
│   │   └── [dataset files]           # AgeAnno dataset and processed data
│   ├── perturbation_analysis.py      # Main perturbation analysis script
│   ├── train_scgpt_random_genes.py   # Training with random gene selection
│   ├── train_scgpt_diffexpressed_genes.py  # Training with DE genes
│   └── scgpt_age.egg-info/           # Package metadata
├── save/                             # Model checkpoints and logs
├── pyproject.toml                    # Project configuration
├── uv.lock                          # Dependency lock file
└── README.md
```

## Key Scripts

- **`train_scgpt_diffexpressed_genes.py`**: Main training pipeline using age-associated genes (recommended for reproducibility)
- **`train_scgpt_random_genes.py`**: Alternative training pipeline with random gene selection
- **`perturbation_analysis.py`**: Gene perturbation analysis to identify pro- and anti-aging genes


## Methodology Details

### Age Classification
The model performs binary classification distinguishing between:
- **Mid-age**: 20-59 years
- **Old-age**: 60-100 years

### Perturbation Analysis
For each gene of interest:
1. Simulate knockdown (expression = 0)
2. Simulate overexpression (expression = 10,000 counts)
3. Compare prediction shifts using paired statistical tests
4. Classify as pro-aging or anti-aging based on probability ratios

### Stability Assessment
- Multiple resampling iterations with random cell subsets
- Fraction of iterations where gene maintains classification
- Holm correction for multiple testing

## Limitations

1. **Computational Constraints**: Limited to 3,000 genes due to memory requirements
2. **Black-box Nature**: Transformer models provide limited mechanistic insights
3. **Chronological Age**: May not fully capture biological aging complexity
4. **Validation Required**: Predictions need experimental validation

## Future Directions

- Validation against independent aging intervention datasets
- Experimental testing of top candidate genes
- Extension to functional aging metrics beyond chronological age
- Integration with multi-omics data

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{tadevosyan2025discovery,
  title={Discovery of age-associated genes using large transcriptome foundation model},
  author={Tadevosyan, Erik and Efimov, Evgeniy and Khrameeva, Ekaterina and Kriukov, Dmitrii},
  journal={IJMS},
  year={2025},
  publisher={MDPI}
}
```

## Funding

This study was supported by the Russian Science Foundation [25-71-20017 to E.K.].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The authors acknowledge the AgeAnno dataset creators and the scGPT development team for providing the foundation models used in this research.

## Contact

For questions or collaborations, please contact:
- **Dmitrii Kriukov**: [kriukov@airi.net](mailto:kriukov@airi.net)
- **Ekaterina Khrameeva**: Skolkovo Institute of Science and Technology

---

**Note**: This repository contains a methodological proof-of-concept. The identified gene perturbations require thorough experimental validation before any biological interpretations or applications.</content>
</xai:function_call">## Data Availability

No new data were created or analyzed in this study. All analyses use the publicly available AgeAnno dataset.

## Conflict of Interest

The authors declare no conflicts of interest.

---

*This README was generated based on the LaTeX manuscript. For the complete scientific manuscript with detailed methods, results, and discussion, please refer to the published article.*