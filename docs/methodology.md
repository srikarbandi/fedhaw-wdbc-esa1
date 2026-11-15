# Methodology

## 1. Objective

This study evaluates the original FedHAW aggregation algorithm [1, 2] for binary
breast-cancer classification on the Wisconsin Diagnostic Breast Cancer (WDBC)
dataset [3]. The data are spread across five *simulated* hospital clients under IID
and non-IID conditions. FedHAW is compared with FedAvg [5], FedProx [6] and a
centralized model trained on pooled data. The hospitals are a simulation created by
partitioning one public dataset. They are not real institutions, and no result here
constitutes clinical validation.

## 2. Data

WDBC [3, 4] contains 569 fine-needle-aspirate cases (357 benign, 212 malignant). Each
case has 30 real-valued features: the mean, standard error and "worst" value of 10
cell-nucleus measurements computed from digitised images. The images themselves are
not part of the dataset, and WDBC contains no mammograms. We used the local copy
`breast+cancer+wisconsin+diagnostic/wdbc.data` (original UCI format) without
modification. The loader verifies programmatically that the file has 569 rows,
32 columns, 30 numeric features, diagnosis values {B, M}, no missing values, no
duplicate rows or IDs, and no duplicate feature vectors
(`data/processed/dataset_report.json`). The ID column is discarded. Labels are
mapped as benign = 0 and malignant = 1 (positive class).

## 3. Preprocessing and splits

For each seed s ∈ {42, 123, 2026}, the data are split 70/15/15 into training
(397), validation (86) and test (86) sets, stratified by diagnosis. A `StandardScaler`
is fitted on the training split only and applied to all three splits. The test split
is used exclusively for final evaluation. Split indices, patient IDs and scaler
parameters are saved in `data/processed/split_seed{s}.json`. Fitting the scaler on
the union of hospital training data is compatible with federated learning: the same
statistics can be computed from per-hospital sums and sums of squares
(`federated_scaler_stats`, verified to be identical).

## 4. Simulated hospitals and partitioning

The training split is divided among K = 5 hospitals so that every sample belongs to
exactly one hospital.

* **IID.** Each class is shuffled and dealt round-robin. Hospitals receive 79–80
  samples with about 63% benign.
* **Dirichlet non-IID** [7]. For each class c, proportions p_c ~ Dir(α·1₅) assign
  class-c samples to hospitals, with α ∈ {10, 1, 0.1} (mild, moderate and severe
  heterogeneity). A draw is rejected and repeated if any hospital receives fewer than
  5 samples (similar in spirit to the minimum-size rule of [8]). The number of
  rejected draws is recorded: 0 for α ∈ {10, 1}; 2, 14 and 7 for α = 0.1 at seeds 42,
  123 and 2026. At α = 0.1, 4–5 of the 5 hospitals hold only one class. These cases
  are reported, not hidden (`data/partitions/`, `results/tables/partitions.csv`,
  `results/plots/partitions_seed*.png`).

**Per-hospital evaluation subsets.** To evaluate the final global model "at each
hospital" without using training data, the test split is divided among hospitals
with the *same* per-class proportions p_c (or the same round-robin dealing for IID).
Each hospital's test subset therefore reflects its own case mix. These subsets are
small (1–52 patients). Metrics are reported as undefined whenever their denominator
is zero (for example, recall on a subset with no malignant cases).

## 5. Model

Every hospital and the server use an identical MLP: 30 → 64 (ReLU) → 32 (ReLU) → 2
logits (4,130 parameters, dropout 0). It is trained with cross-entropy on raw logits.
P(malignant) is softmax(logits)[1], and the predicted class is malignant when this
is ≥ 0.5. Model parameters are exchanged as flat vectors using PyTorch's
`parameters_to_vector`, which follows a fixed parameter order.

## 6. Algorithms

All federated methods run T = 50 rounds. In every round, all 5 hospitals start from
the current global model and run E = 1 local epoch of mini-batch SGD (batch 16).

* **FedAvg** [5]: θ^{t+1} = Σ_k (N_k/N) θ_k^t.
* **FedProx** [6]: local loss L_k(θ) + (μ_prox/2)‖θ − θ^t‖², with N_k/N aggregation.
* **FedHAW** [1, 2]: θ^{t+1} = e^{γ} Σ_k softmax(λ)_k θ_k^t. γ and λ are updated by
  hypergradient descent from round 1 onward, exactly as in the original
  implementation (see `docs/fedhaw_reproduction_notes.md`). FedHAW is reported with two
  step-size settings: the original repository defaults (lr_γ = 1e-3, lr_λ = 1e-2) and
  validation-tuned values.
* **Centralized**: the same MLP and optimizer trained on the pooled training split for
  T·E = 50 epochs (the same number of passes over the data).

Fairness controls: within a (distribution, seed) pair, every method uses the same
split, scaler, partition, initial parameters and RNG state for mini-batch shuffling,
as well as the same rounds, epochs, batch size and learning rate. The final-round
model is evaluated, with no early stopping.

## 7. Hyperparameter selection

A pilot with the suggested initial setting (SGD learning rate 0.001, 50 rounds)
showed that all methods remain under-trained (FedAvg validation accuracy 0.69).
Hyperparameters were therefore selected on the **validation split only**, with a
criterion fixed in advance: the lowest mean final-round validation cross-entropy over
the 4 distributions × 3 seeds (`scripts/tune.py`).

| Stage | Grid | Selected |
|---|---|---|
| Shared learning rate (via FedAvg) | {0.001, 0.01, 0.05, 0.1} | 0.05 |
| FedHAW lr_γ × lr_λ | {1e-6 … 1e-2} × {1e-3 … 1} | 1e-5, 1e-2 |
| FedProx μ_prox | {0.001, 0.01, 0.1, 1} | 0.01 |

Two complete experiment configurations are reported:
**main** (selected values) and **suggested_lr0.001** (the brief's initial setting,
lr = 0.001, with FedHAW at the repository defaults).

## 8. Evaluation

Test metrics, with malignant as the positive class, are: accuracy, recall
(sensitivity), specificity, false-positive rate, precision, F1, ROC-AUC (computed from
P(malignant)) and the confusion matrix (scikit-learn implementations). On the
hospital test subsets we report the same metrics plus the mean hospital accuracy,
mean / worst / standard deviation of hospital recall, and the best-to-worst recall
gap. All values are mean ± standard deviation over the 3 seeds. Per-round training
loss, validation loss, validation accuracy, validation malignant recall, γ, hospital
weights, local training time and aggregation time are logged
(`results/raw/<config>/<distribution>/seed<s>/<method>/round_log.csv`).

**Statistical testing.** With n = 3 seeds, a paired test cannot reach conventional
significance (for example, the smallest possible two-sided Wilcoxon p-value is 0.25).
No significance claims are made. Per-seed paired differences against FedAvg are
reported descriptively (`results/tables/*_paired_vs_fedavg.csv`).

## 9. Software and hardware

Python 3.11.16 in a project virtual environment; PyTorch 2.15.0 (CPU,
single-threaded, deterministic algorithms); scikit-learn 1.9.1. Runs were sequential
on an Apple M2 laptop with 8 GB of RAM. The complete pipeline (tests, tuning, experiments, report) runs in a few minutes.

## References

1. A. Nakai-Kasai and T. Wadayama, "Federated learning with hypergradient-based online
   update of aggregation weights," arXiv:2605.00458, 2026. <https://arxiv.org/abs/2605.00458>
2. A. Nakai-Kasai, FedHAW source code. <https://github.com/a-nakai-k/FedHAW>
   (commit 6f565520ba51e3e673520e1e22eaa5f9f0ed244c).
3. W. Wolberg, O. Mangasarian, N. Street and W. Street, "Breast Cancer Wisconsin
   (Diagnostic)," UCI Machine Learning Repository, 1995.
   <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>,
   doi:10.24432/C5DW2B.
4. W. N. Street, W. H. Wolberg and O. L. Mangasarian, "Nuclear feature extraction for
   breast tumor diagnosis," *IS&T/SPIE International Symposium on Electronic Imaging:
   Science and Technology*, vol. 1905, pp. 861–870, 1993.
5. B. McMahan, E. Moore, D. Ramage, S. Hampson and B. A. y Arcas,
   "Communication-efficient learning of deep networks from decentralized data,"
   *AISTATS*, PMLR 54, pp. 1273–1282, 2017.
6. T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar and V. Smith, "Federated
   optimization in heterogeneous networks," *Proceedings of Machine Learning and
   Systems (MLSys)*, vol. 2, 2020.
7. T.-M. H. Hsu, H. Qi and M. Brown, "Measuring the effects of non-identical data
   distribution for federated visual classification," arXiv:1909.06335, 2019.
8. Q. Li, Y. Diao, Q. Chen and B. He, "Federated learning on non-IID data silos: An
   experimental study," *IEEE ICDE*, 2022.
