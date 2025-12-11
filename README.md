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

Aging is a progressive functional decline driven by complex genetic, epigenetic, environmental, and stochastic interactions that traditional linear models fail to capture. Using human single-cell RNA-seq data from the multi-tissue AgeAnno dataset, we fine-tuned scGPT, a large transcriptomic model, to predict chronological age, achieving an average ROC-AUC of 0.91 (compared to 0.89 for logistic regression). To identify genes influencing age predictions, we systematically perturbed individual genes *in silico* and quantified their effects, classifying them as pro- or anti-aging candidates. The top-ranked genes showed high stability across repeated subsampling iterations, with 33--50% overlap with experimentally supported aging-related genes from the Open Genes database. This work presents a methodological framework demonstrating the potential of large transcriptomic models to generate biologically meaningful hypotheses in aging research.

## Key Features

- **Age Prediction**: Fine-tuned scGPT model for binary age classification (mid-age vs old-age)
- **Gene Perturbation Analysis**: Systematic *in silico* perturbations to identify pro- and anti-aging genes
- **Multi-tissue Analysis**: Trained on 9 diverse human tissues from the AgeAnno dataset
- **High Performance**: Average ROC-AUC of 0.91 (compared to 0.89 for logistic regression)
- **Stability Assessment**: Resampling-based validation across 10 iterations per tissue
- **Biological Validation**: Comparison with Open Genes database showing 33--50% overlap for top-ranked genes

## Dataset

The analysis uses the **AgeAnno dataset** [Huang et al., 2023] comprising scRNA-seq data from human tissues obtained from donors whose chronological ages ranged from 0 to 100+ years. The dataset includes four age groups: "youth" (0--19), "mid" (20--59 years), "old" (60--100 years), and "supold" (>100 years), from which we used only the mid and old age groups. We analyzed 9 tissues (blood, skin, brain, lung, skeletal muscle, stomach, bladder, liver, and bone marrow) after omitting tissues with low sample coverage. Key preprocessing steps include:

- Filtering with scanpy Python package (version 1.11.5)
- Exclusion of genes expressed in <1% of cells
- Discretization of gene expression values into 51 bins (following scGPT preprocessing protocol)
- Selection of 869 age-associated genes from the AgeAnno dataset (adjusted p-value < 0.05)

## Model Architecture

- **Base Model**: scGPT (Transformer-based large transcriptomic model)
- **Task**: Binary age classification (mid-age: 20-59 vs old-age: 60-100)
- **Fine-tuning Strategy**: Only classification head parameters optimized; pretrained transformer encoder weights frozen
- **Classification Head**:
  - Five linear layers, each with 1024 hidden units
  - GELU activation functions
  - Layer normalization after scGPT embeddings
  - Dropout rate: 0.4
  - Average pooling to aggregate gene-level embeddings
- **Training Hyperparameters**:
  - Epochs: 20
  - Learning rate: 1×10⁻⁴
  - Batch size: 256
  - Optimizer: AdamW with weight decay of 1×10⁻⁷
  - Loss: Binary cross-entropy with logits (BCEWithLogitsLoss) with class-specific weights
- **Hardware**: NVIDIA RTX 4090 GPU (24 GB memory)

## Results

### Fine-Tuning scGPT for Age Classification

We fine-tuned the scGPT model for binary age classification, distinguishing between cells retrieved from the mid-age (20--59 years) and the old-age (60--100 years) donors, as defined in the AgeAnno dataset. The input gene set for model fine-tuning comprised 869 genes reported as age-associated in the original AgeAnno publication [Huang et al., 2023].

Fine-tuned scGPT model captures age-related transcriptomic patterns in the AgeAnno dataset. (a) Schematic of the proposed in silico perturbation analysis for identifying candidates for pro- and anti-aging perturbations. (b) ROC AUC values illustrating classification accuracy of scGPT and LR models on aging-associated genes from the AgeAnno dataset (mid-age versus old-age cells). (c) Example of model output for blood, showing predicted pro-aging (red) and anti-aging (blue) gene candidates, subject to thorough post-hoc validation. Colored dots highlight genes with significant adjusted p-values across 10 out of 10 random subsampling iterations, while gray horizontal bars represent standard deviations across the same iterations. X and Y axes show average log2 (P_knockout / P_overexpression) and maximal log10 (adjusted p-value) values across all iterations, respectively. White stars at dot centers mark known aging-associated genes according to the Open Genes database [Rafikova et al., 2024]. Other tissues are shown in Figure A2.

### Model Performance

Across all tissues, the fine-tuned scGPT model yielded generally comparable performance metrics, with consistent ROC-AUC values. While certain tissues (such as stomach and blood) exhibited more pronounced differences in model performance, the aggregate results across tissues showed that scGPT achieved a marginally higher average ROC-AUC of 0.91 compared to 0.89 for logistic regression (LR) applied to the same dataset and gene subset (Figure 1b).

Importantly, this modest improvement does not imply that scGPT is inherently superior---or inferior---to LR for predicting chronological age. Rather, it demonstrates that scGPT achieves performance on par with a standard baseline model, thereby validating its suitability as a tool for downstream analyses. Specifically, this confirms that the fine-tuned scGPT can reliably be used to investigate the individual contribution of each gene to age-related transcriptional changes, including through in silico perturbation experiments.

### Perturbation Analysis Identifies Candidates for Pro- and Anti-Aging Gene Perturbations

To investigate which genes contribute most to age predictions, we conducted an in silico perturbation analysis in each tissue. Because extracting gene-level importance from transformer models is non-trivial, we performed the following experiment: for each gene, predictions were obtained from the fine-tuned scGPT model under two conditions: (1) setting the gene's expression to zero (simulating a knockout) and (2) setting the gene's expression to its maximal value (simulating strong overexpression) (see Materials and Methods for details). This procedure yielded per-tissue lists of predicted perturbation candidates, along with their adjusted p-values (Figure 1c; Table S2).

Importantly, these predictions require further validation before they can be considered as robust candidate perturbations.

### Predicted Candidates Are Stable Across Repeated Subsampling

To evaluate the robustness of the predicted candidate perturbations, we repeated the perturbation identification procedure under multiple rounds of random subsampling in each tissue (see Materials and Methods). The top-ranked genes showed highly consistent perturbation directions and statistical significance across iterations (Table S2). Ranking genes by their stability revealed that many remained significant (adjusted p < 0.05) in all subsampling runs, and their effect estimates exhibited uniformly small standard deviations, indicating strong reproducibility of both direction and magnitude (Figures 1c and A2).

These findings support the stability and statistical robustness of the perturbation results. Comparison of the top-ranked genes with the Open Genes database [Rafikova et al., 2024] of experimentally supported aging-related genes shows a substantial overlap (33--50% of genes significant in all subsampling runs per tissue, Figures 1c and A2), further supporting the biological relevance of obtained predictions. However, we note that these perturbation outcomes still represent model-based predictions and do not imply biological causation. They therefore serve as hypothesis-generation tools rather than mechanistic claims.

Thus, we present a methodological framework for identifying candidates for pro- and anti-aging gene perturbations across tissues, demonstrating the potential of large transcriptomic models such as scGPT to capture age-related patterns and generate biologically meaningful hypotheses in aging research (Table 1).

#### Comparison of scGPT-based Framework with Classical Transcriptomic Aging Models

| Aspect | Classical Approaches (LR, Elastic Net, etc.) | scGPT (Foundation Model) | Implications for Aging Research |
|--------|----------------------------------------------|--------------------------|--------------------------------|
| **Model architecture** | Linear models with fixed feature relationships | Transformer-based model with contextual embeddings | Captures nonlinear gene--gene interactions relevant to aging |
| **Data representation** | Uses preselected or summarized features | Learns latent representations across tissues and conditions | Detects shared aging signatures |
| **Interpretability** | Statistically transparent but biologically limited: indicate correlation, not causation | More interpretable: perturbation analysis reveals gene influence on predicted age | Enables hypothesis generation about key regulatory genes |
| **Predictive accuracy** | Moderate; limited to linear effects | Potentially higher; models nonlinear and higher-order dependencies | Designed to be sensitive to subtle aging-related transcriptomic shifts |
| **Generalizability** | Dataset-specific; often tissue-limited | Pretrained on large multi-tissue data | Facilitates cross-tissue and cross-study generalization |
| **Causal inference** | Based on association only; weak causal inference | In silico perturbation directly tests the model's response to gene changes | Provides clues about potential pro- and anti-aging regulators |
| **Computational requirements** | Low; easily reproducible on standard hardware | Very high GPU and memory demand | Much higher costs but greater modeling capacity and discovery potential |
| **Data requirements** | Works with small, well-curated datasets | Benefits from large, heterogeneous single-cell datasets | Leverages large-scale data to uncover universal aging patterns |
| **Experimental integration** | Descriptive correlations without actionable targets | Provides ranked candidate genes for downstream perturbation studies | Bridges predictive modeling and experimental validation |
| **Overall role** | Descriptive and correlative | Predictive and hypothesis-generating | Advances from correlations to hypothesis generation |

## Discussion

We present the first example of using a foundation model for gene expression in aging research to identify candidates for gene perturbations---pro- or anti-aging ones. Importantly, this study should not be viewed as intended to produce a robust list of anti-aging candidate genes. Instead, it serves as a methodological proof-of-concept showing that large transcriptomic models like scGPT can be used in principle not only as predictive tools but also as instruments for hypothesis generation in aging research.

The modest performance gap in prediction metrics between scGPT and LR likely reflects that both models are trained on a curated set of aging-associated genes, which already provides a strong age signal, allowing even simple linear models achieve high accuracy. This pattern is consistent with prior works, where relatively simple models often perform competitively with more complex architectures [Fleischer et al., 2018; Meyer et al., 2021; Shokhirev et al., 2020; Mao et al., 2023]. Furthermore, both the BiT age [Meyer et al., 2021] and the recent DeepQA framework [Qi et al., 2025] suggest that performance gains from more complex architectures can be modest when age-informative genes are preselected. In this context, scGPT's advantage lies not in maximizing AUC but in its ability, learned during pre-training, to implicitly model the gene regulatory network [Ahlmann et al., 2025], thereby capturing how perturbing a single gene propagates through the transcriptome even when that gene lacks a direct linear association with age. We validate this approach by confirming that scGPT's representations retain the cellular age signal as effectively as raw transcriptomic data (as measured by LR), ensuring its suitability for identifying potential pro- and anti-aging gene candidates. Accordingly, the value of scGPT in this work is reflected primarily in its hypothesis-generation capabilities rather than in an improvement in classification accuracy.

Beyond methodological perspective, our study illustrates how perturbation analysis can be used to provide insights into the high-order structure of aging-related gene regulation. By simulating the up- or downregulation of individual genes and observing corresponding shifts in predicted age, we can infer which genes are most influential in shaping transcriptomic aging signatures. This analysis is designed to map how local gene expression changes propagate through the latent representation learned by the model, revealing regulatory leverage points that may control broader aging-related programs. However, it is important to emphasize that the perturbation outcomes do not imply biological causality. The observed shifts in predicted age represent model-predicted associations learned from training data, not experimentally validated mechanistic effects. These outputs should therefore be interpreted solely as hypothesis-generating signals, guiding downstream experimental prioritization rather than providing causal conclusions.

The proposed framework also complements existing systems-biology methods. Integrating the predicted genetic perturbations effects with pathway and network analyses could reveal aging-relevant subnetworks and highlight best candidates for downstream experimental validation. Such integration would help connect the predictive capabilities of deep learning models with mechanistic insight derived from empirical biology. In this sense, scGPT acts as a hypothesis-generating engine, narrowing the search space for follow-up validation and accelerating discovery in geroscience.

Nevertheless, several limitations must be acknowledged. First, although scGPT can rank genes by predicted "pro-" or "anti-aging" effects, its black-box nature prevents explaining these choices or revealing underlying mechanisms. Second, scGPT requires substantial computational resources; e.g., our analysis was limited to a subset of 869 genes to fit within reasonable memory constraints. Finally, a conceptual limitation is the usage of chronological age labels provided in the AgeAnno dataset. Chronological age does not account for individual medical history, lifestyle, or individual molecular aging trajectories. Future work should incorporate biologically informed aging labels (e.g., frailty index, intrinsic capacity score, or biological age [Searle et al., 2008; Howlett et al., 2021; Klemera et al., 2006]) when such data become available for gene expression at single-cell resolution.

Looking forward, future research could explore multi-omics integration to combine transcriptomic perturbations with epigenetic or proteomic data, enhancing mechanistic resolution. Another promising direction is cross-species transfer learning, where foundation models pre-trained on diverse organisms could uncover evolutionary conserved regulators of aging. Furthermore, incorporating explainability methods, such as feature attribution (e.g., saliency mapping, attention and rationale models [Zhou et al., 2022]) could help dissect which biological processes or pathways contribute most to the model's predictions.

Most importantly, future work should focus on developing rigorous validation strategies for anti-aging perturbations predicted by large models. As the first step, we propose testing predicted genes against existing independent datasets of anti-aging interventions using causal inference approaches that statistically evaluate whether effects persist across datasets [Meinshausen et al., 2016]. As the second step, top candidates should be confirmed through classical experimental perturbations such as gene knockdowns or overexpression in cell lines and model organisms.

## Materials and Methods

### Dataset

We used the AgeAnno dataset [Huang et al., 2023], comprising scRNA-seq data from human tissues obtained from donors whose chronological ages ranged from 0 to 100+ years. The exact chronological ages are binned into four age groups in the dataset [Huang et al., 2023]: "youth" (0--19), "mid" (20--59 years), "old" (60--100 years), and "supold" (>100 years), from which we used only the mid and old age groups due to limited numbers of donors in other groups (Figure A1; Table: cell count per tissue). We omitted tissues with low (<3) sample coverage per age group, resulting in 9 tissues (blood, skin, brain, lung, skeletal muscle, stomach, bladder, liver, and bone marrow). Raw count matrices were filtered with the scanpy [Wolf et al., 2018] Python package (version 1.11.5). Genes expressed in fewer than 1% of cells were excluded. Gene expression values were discretized into 51 bins following the preprocessing protocol established by the original scGPT authors [Cui et al., 2024]. This binning strategy is a core component of the scGPT input representation and was applied as recommended in the official documentation and reference implementation (https://github.com/bowang-lab/scGPT/, accessed on 25 October 2025).

The total number of cells included in the analysis for each tissue and age group is provided in Table: cell count per tissue.

For each tissue type, the training and testing splits were performed at the donor level (i.e., all cells from a given donor were assigned exclusively to either the training or the test set). This donor-wise partitioning prevents data leakage and ensures that the model is evaluated on truly unseen individuals. No within-dataset batch correction was applied. All experiments were conducted with a fixed random seed to guarantee reproducibility across runs.

For the preliminary complexity reduction step, we subsampled the gene set by selecting genes previously reported (in AgeAnno) to be significantly associated with aging in the comparison between mid-age and old-age groups (adjusted p-value < 0.05), further filtered to exclude genes exhibiting tissue-specific or cell-type-specific expression changes, and then intersected the filtered subset with the scGPT training genes, resulting in total 869 genes (Table S1).

### scGPT Fine-Tuning Configuration

During fine-tuning, we optimized only the parameters of the classification head, while keeping the pretrained transformer encoder weights entirely frozen. This strategy allowed the model to retain general transcriptomic representations learned during pretraining while adapting specifically to age prediction. All computations were performed on an NVIDIA RTX 4090 GPU (Santa Clara, CA, USA) with 24 GB of memory. We used the original scGPT model without any modifications to its core architecture. Only a custom fully connected prediction head was trained on top of the frozen scGPT encoder. To obtain a single embedding representing each cell, we aggregated the gene-level embeddings using average pooling. The prediction head consisted of five linear layers, each with 1024 hidden units, interleaved with GELU activation functions. Layer normalisation was applied immediately after the scGPT embeddings, and a dropout rate of 0.4 was used throughout the head.

The model was trained for 20 epochs with a batch size of 256 and a learning rate of 1×10⁻⁴. We employed the AdamW optimizer with a weight decay of 1×10⁻⁷ and used the binary cross-entropy with logits loss (BCEWithLogitsLoss) with class-specific weights to account for label imbalance [Ansel et al., 2024].

The underlying scGPT configuration remained unchanged.

### Logistic Regression Configuration

Logistic regression (LR) was employed with default parameters and L1 regularization using the scikit-learn Python package (version 1.7.2) [Pedregosa et al., 2011]. The model was configured as a pipeline consisting of two components: a feature scaling step (StandardScaler function) and a multinomial logistic regression classifier (LogisticRegression function with parameters: random_state = 42, max_iter = 1000, multi_class = 'multinomial').

The inclusion of the StandardScaler ensured that each feature had zero mean and unit variance, improving model convergence and stability. The LR model was trained using the same input features as the scGPT fine-tuning experiment, allowing for a fair comparison of predictive performance.

### Identification of Putative Pro-Aging and Anti-Aging Perturbations

For the perturbation analysis, 300 cells were randomly selected per tissue (150 from the mid-age group and 150 from the old-age group) to balance numbers of cells per tissue per age group. Downsampling numbers of cells per tissue is an important step to avoid potential confounding from tissue composition. For every gene of interest, predictions were obtained from the fine-tuned scGPT model under two simulated perturbation conditions: knockout and overexpression. In silico perturbations were performed for all genes included in the model training set, as well as for additional curated aging-related genes from the OpenGenes database [Rafikova et al., 2024] (restricted to entries with "high" or "highest" confidence levels) and the GenAge database [de Magalhães et al., 2024]. This combined gene set enabled a comprehensive assessment of how both model-informed and literature-validated aging-associated genes influence predicted chronological age under knockout and overexpression conditions.

To simulate these perturbations, we applied a gene-specific strategy in which the maximal raw expression value of each gene was identified across all cells in a particular tissue. Perturbed profiles were then generated by replacing the gene expression value with zero to approximate knockout, or with its empirical maximum to approximate strong overexpression. This procedure follows the perturbation logic used in existing frameworks such as CellOracle [Kamimoto et al., 2023] and ensures that all simulated perturbations remain within the observed range of the training data. These perturbations represent intentionally extreme manipulations and are therefore intended solely for hypothesis generation rather than for quantitative estimates of experimentally observed expression changes.

Model outputs corresponded to the predicted probability distribution across age groups for each cell, allowing us to estimate how each perturbation shifted the predicted cellular age. Subsequently, p-values were computed for each gene using Mann–Whitney statistical test, followed by Holm correction. Genes showing a statistically significant difference between age groups, as well as P(old class probability with low expression) / P(old class probability with high expression) > 1 were classified as candidates for anti-aging perturbations, whereas genes with P(old class probability with low expression) / P(old class probability with high expression) < 1 were classified as candidates for pro-aging perturbations.

We emphasize that the perturbation procedure reflects changes in the scGPT model's output and cannot serve as an evidence of biological causality. The results represent model-derived predictions, not experimentally validated effects.

### Estimating Stability of Predictions

To evaluate the robustness of our procedure for identifying putative pro-aging and anti-aging perturbations, we performed a stability assessment experiment. In the perturbation identification workflow described above, random subsets of 300 cells per tissue are sampled for analysis. We repeated this procedure 10 times, each time drawing a new independent subset of 300 cells from each tissue, thereby generating 10 lists of pro-aging and anti-aging candidates per tissue. For each gene, stability was quantified as the proportion of iterations (out of 10) in which the gene was classified as "pro-aging" or "anti-aging," respectively, with an adjusted p-value < 0.05. This proportion provides an empirical measure of the robustness of both the inferred perturbation direction and its statistical significance across subsampling iterations.

### Comparison with the Open Genes Database

We compared the predicted pro-aging and anti-aging candidates with the Open Genes database [Rafikova et al., 2024] which contains manually curated evidence-based confidence levels for genes associated with aging. The database was downloaded in tab-separated format, containing 2405 human genes annotated with confidence levels ranging from "highest" to "lowest" based on published experimental evidence.

## Conclusions

Our findings demonstrate that scGPT can capture aging-related transcriptional patterns and enable systematic in silico perturbation analysis at the gene level. Although its predictive performance is similar to that of simpler models, scGPT's key advantage lies in its ability to capture latent regulatory structure and generate hypotheses about genes that may underlie aging-associated transcriptomic changes. These predictions should, however, be interpreted with caution, as they remain computational inferences and require careful downstream experimental validation. Future work should expand dataset coverage (including phenotypes that better approximate biological aging than chronological age), integrate higher-confidence annotations, and incorporate causal inference frameworks for evidence aggregation across datasets. Ultimately, definitive assessment will require targeted experimental validation. By outlining a perturbation-based interpretability workflow for a large transcriptomic model, this study illustrates both the potential and current limitations of applying foundation models to uncover new biological insights into aging.

## Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.0.0 - 2.2.0 (tested with 2.1.2+cu121)
- CUDA-compatible GPU (recommended: NVIDIA RTX 4090 with 24GB memory or higher)
- UV package manager (for dependency management)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ifreyk/scgpt-age.git
cd scgpt-age
```

2. Install all dependencies using UV:
```bash
uv sync
```
This command will:
- Read dependencies from `pyproject.toml`
- Install all required packages using the locked versions in `uv.lock`
- Set up the project environment

3. Install the package in development mode:
```bash
uv pip install -e .
```

**Note**: The project uses `pyproject.toml` for project configuration and `uv.lock` for dependency locking to ensure reproducible builds.

## Usage

The pipeline consists of several scripts that can be run to reproduce the analysis:

### 1. Train Classification Head on scGPT Embeddings
```bash
python src/train_cls_decoder_on_embeddings.py
```
Trains separate ImprovedCls classification heads on scGPT embeddings for each tissue. This script:
- Loads the pre-trained frozen scGPT encoder
- Extracts cell embeddings from train and test data
- Trains a separate ImprovedCls head for each tissue
- Saves model weights and performance metrics

### 2. Gene Perturbation Analysis
```bash
python src/perturbation_analysis_cls_decoder.py
```
Performs systematic *in silico* gene perturbations to identify pro- and anti-aging gene candidates. This script:
- Loads the frozen scGPT encoder and trained classification heads
- Samples cells from test data (stratified by age category)
- For each gene, creates knockout (zero expression) and overexpression (max expression) perturbations
- Computes statistical significance and classifies genes as pro-aging or anti-aging candidates
- Saves results for downstream analysis

**Configuration**: Edit the script to set `TISSUE` (default: "skin") and `N_CELLS_TO_SAMPLE` (default: 500).

### 3. Compare with Logistic Regression Baseline
```bash
python src/compare_logistic_regression_multi_tissue.py
```
Compares scGPT performance with logistic regression baseline across multiple tissues. This script:
- Trains logistic regression models on the same gene set
- Evaluates performance metrics (ROC-AUC, accuracy, etc.)
- Generates comparison plots and statistics

### 4. Calculate Gene Statistics (Post-processing)
```bash
python src/calculate_gene_statistics.py
```
Processes perturbation results to calculate gene-level statistics across multiple runs. This script:
- Aggregates results from multiple perturbation analysis runs
- Computes stability metrics and significance across iterations
- Generates summary statistics for candidate genes

## Project Structure

```
scgpt-age/
├── src/
│   ├── data/
│   │   ├── models/                              # Pretrained scGPT models
│   │   │   ├── scGPT_human/                    # Base scGPT model
│   │   │   └── scGPT_best_age/                 # Fine-tuned age model
│   │   ├── perturbation_results/                # Perturbation analysis results
│   │   └── [dataset files]                      # AgeAnno dataset files
│   ├── train_cls_decoder_on_embeddings.py      # Train classification heads on embeddings
│   ├── perturbation_analysis_cls_decoder.py    # Main perturbation analysis script
│   ├── compare_logistic_regression_multi_tissue.py  # Compare with LR baseline
│   ├── calculate_gene_statistics.py             # Post-process perturbation results
│   ├── improved_cls_decoder.py                 # ImprovedCls decoder implementation
│   └── scgpt_age.egg-info/                     # Package metadata
├── save/                                        # Model checkpoints and logs
├── pyproject.toml                               # Project configuration and dependencies
├── uv.lock                                      # Locked dependency versions
└── README.md
```

## Key Scripts

- **`train_cls_decoder_on_embeddings.py`**: Trains ImprovedCls classification heads on scGPT embeddings for each tissue
- **`perturbation_analysis_cls_decoder.py`**: Performs systematic gene perturbations to identify pro- and anti-aging candidates
- **`compare_logistic_regression_multi_tissue.py`**: Compares scGPT performance with logistic regression baseline
- **`calculate_gene_statistics.py`**: Processes and aggregates perturbation results across multiple runs


## Limitations

1. **Computational Constraints**: Limited to 869 genes due to memory requirements
2. **Black-box Nature**: Transformer models provide limited mechanistic insights
3. **Chronological Age**: Does not account for individual medical history, lifestyle, or individual molecular aging trajectories
4. **Validation Required**: Predictions need experimental validation and do not imply biological causality

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

**Note**: This repository contains a methodological proof-of-concept. The identified gene perturbations require thorough experimental validation before any biological interpretations or applications.

## Data Availability

No new data were created or analyzed in this study. All analyses use the publicly available AgeAnno dataset.

## Conflict of Interest

The authors declare no conflicts of interest.

---

*This README was generated based on the LaTeX manuscript. For the complete scientific manuscript with detailed methods, results, and discussion, please refer to the published article.*