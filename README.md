# Experimentation Overview

The following notebooks implement different approaches for early detection of antibiotic effects on bacterial growth.

## Tran_methodology.ipynb (Baseline)

As a baseline, `Tran_methodology.ipynb` includes all the functions from `EXP-23-BZ3167.ipynb` for:
- Processing Omnipose segmentation masks and computing bacterial areas
- Fitting a rolling-window exponential model *A(t) = a·e^(bt)* to extract growth rates
- Generating figures (area curves, growth-rate curves, normalized plots)
- Computing the **detection time**, the earliest point where treatment diverges from reference

These functions will be reused throughout the experimentation for area/growth-rate computation and figure generation.

![Tran Methodology Results](figures/tran_figure.png)

The figure shows the normalized growth rate of treated versus untreated conditions. The detection time *T* is the earliest time point at which treated and untreated growth rates differ significantly (Welch t-test, p < 0.05).

## Finetuning_Sam_experiments.ipynb

The initial research question focused on whether **improving cell segmentation** could reduce noise in area-based growth measurements and, as a result, enable earlier detection of antibiotic effects.

For this purpose, the **general SAM model** is fine-tuned using LoRA with different configurations. The dataset is split into train, validation, and test sets. The best configuration was **LoRA rank 32**, achieving Dice = 0.9275, IoU = 0.8649, Precision = 0.9113, Recall = 0.9443, F1 = 0.9275.

However, the fine-tuned model must be evaluated on an unseen test dataset to generate figures and compute the detection time, allowing comparison with the baseline Omnipose model.

## Heterogeneity_analysis.ipynb

Since fine-tuning a model using masks from another model as ground truth is not scientifically correct, an alternative approach was explored, analyzing **heterogeneity**. Some cells may continue growing rapidly while others slow down or stop.

Each chamber (both reference and treated) is divided into 3 horizontal patches, and the area of each region is measured over time. Growth rates are computed for every patch, and the patch with the highest average growth is labeled as the **hotspot**. The hotspot growth curve is then compared with the **background** (average of the remaining patches).

![Heterogeneity Results](figures/heterogeneity_result.png)

The results in the figure are not very clear as the dataset used does not exhibit strong heterogeneity. However, visual inspection added in the notebook shows that the algorithm finds logical regions that appear to grow faster than the other patches.

## Early_detection_Sam2.ipynb

Another approach was to use a model suitable for video tracking with inference only (no training on the dataset). **SAM2** is used for segmentation and mask propagation, where each cell is given an initial mask and SAM2 propagates it across all frames. From these masks, single-cell area curves and growth rates are computed.

Two detection methods are considered:
- **Method A (Chamber-Level Mean Growth)**: Growth rates are averaged across all cells in a chamber, then treated vs. reference curves are compared to find when they diverge. The same approach as Tran et al. for creating figures and computing detection time was applied.
- **Method B (Single-Cell Distribution Analysis)**: The goal is to define a single-cell antibiotic response metric that detects drug effect earlier than the usual "total area per chamber" growth rate. This method uses the full distribution of single-cell growth rates with three detection signals:
  - *M_cell(t)*: Separation metric comparing mean and std of growth rates between treated and control cells (values > 2 indicate clear separation)
  - *f_slow(t)*: Fraction of cells with growth rate below the 10th percentile of control baseline
  - Welch t-test p-value: Statistical test for significant difference between groups (p < 0.05)

  To prevent false detections from noise, a signal is valid only if the condition holds for three consecutive frames.

**Note**: SAM2's tracking results are unreliable (cells change size, disappear, or merge inconsistently). The Methods section is also not well validated. Results from this notebook should be interpreted with caution.

