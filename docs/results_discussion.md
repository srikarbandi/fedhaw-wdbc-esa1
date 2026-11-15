# Results and discussion

Numbers come from `results/reports/main_report.md` and
`results/reports/suggested_lr0.001_report.md`, which are generated from the raw run
outputs. Values are mean ± std over seeds 42, 123 and 2026, on 86 test patients per
seed (54 benign / 32 malignant). With 86 patients, one misclassified patient changes
accuracy by 0.0116, and one missed malignant case changes recall by 0.031.

## 1. Main configuration (validation-tuned: lr = 0.05, 50 rounds)

| Distribution | Centralized | FedAvg | FedProx | FedHAW (repo defaults) | FedHAW (val-tuned) |
|---|---|---|---|---|---|
| IID – accuracy | 0.965 ± 0.016 | 0.973 ± 0.020 | 0.973 ± 0.020 | 0.965 ± 0.016 | 0.973 ± 0.020 |
| IID – recall | 0.958 ± 0.015 | 0.979 ± 0.015 | 0.979 ± 0.015 | 0.958 ± 0.015 | 0.979 ± 0.015 |
| α=10 – accuracy | 0.965 ± 0.016 | 0.977 ± 0.010 | 0.977 ± 0.010 | 0.961 ± 0.020 | 0.977 ± 0.010 |
| α=1 – accuracy | 0.965 ± 0.016 | 0.973 ± 0.011 | 0.973 ± 0.011 | 0.953 ± 0.025 | 0.973 ± 0.011 |
| α=0.1 – accuracy | 0.965 ± 0.016 | 0.973 ± 0.011 | 0.973 ± 0.011 | 0.961 ± 0.011 | 0.973 ± 0.011 |
| α=0.1 – recall | 0.958 ± 0.015 | 0.948 ± 0.030 | 0.948 ± 0.030 | 0.958 ± 0.015 | 0.948 ± 0.030 |
| α=0.1 – specificity | 0.969 ± 0.023 | 0.988 ± 0.018 | 0.988 ± 0.018 | 0.963 ± 0.015 | 0.988 ± 0.018 |
| α=0.1 – ROC-AUC | 0.985 ± 0.009 | 0.993 ± 0.005 | 0.993 ± 0.005 | 0.977 ± 0.011 | 0.992 ± 0.005 |

Observations:

* **All methods reach 95–98% test accuracy in every distribution, including α = 0.1**,
  where 4–5 of the 5 hospitals hold a single class. On WDBC, under this protocol,
  heterogeneity barely affects final accuracy. The partition is label- and quantity-skewed,
  but the pooled training data stays the same, and FedAvg with one local epoch at a
  moderate learning rate does not drift far.
* **FedHAW with validation-tuned step sizes (lr_γ = 1e-5, lr_λ = 1e-2) gave the same
  test predictions (identical confusion matrices) as FedAvg in all 12 distribution × seed
  runs.** At these step sizes, exp(γ) stays within 0.1% of 1 (final 1.0006–1.0008), and the
  weights move at most 0.024 from N_k/N. The tuning grid shows the reason: the lowest
  validation loss is reached at the smallest lr_γ, and larger lr_γ only increases it
  (`results/plots/tuning_fedhaw_val_loss.png`).
* **FedHAW with the repository defaults (1e-3, 1e-2)** has its adaptation active:
  exp(γ) grows to 1.010–1.036 by round 50. It was slightly *worse* than FedAvg in
  accuracy (by 0.008–0.019 depending on the distribution; FedAvg was never lower in a
  seed) and in ROC-AUC (0.977–0.984 vs 0.993–0.994). Its false-positive rate was higher under
  Dirichlet splits (0.037–0.049 vs 0.012–0.025) and equal under IID. The one exception is
  α = 0.1, where mean recall was 0.010 higher than FedAvg (1 of 3 seeds higher, 2 equal).
  Repeated scaling by e^γ > 1 inflates the logits, which pushes predictions towards
  higher confidence. Validation cross-entropy confirms this: 0.556 vs 0.059 for FedAvg.
* **FedProx with μ_prox = 0.01 matched FedAvg almost exactly.** At this lr and with one
  local epoch, the proximal pull (μ_prox·‖θ − θ^t‖) is tiny compared with the loss
  gradient. Larger μ_prox did not lower validation loss.
* **The centralized model is not better than the federated ones here** (0.965 vs
  0.973). It takes about 25 SGD steps per epoch on pooled data, with no averaging, and
  the final-epoch model is used without early stopping. The differences are within
  1–2 test patients.
* **Per-hospital results** (`results/plots/main/per_client_recall.png`) are dominated
  by small-sample effects. At α = 0.1, Hospital 4's test subset has no malignant case
  in any seed, so its recall is undefined. Hospital 3's mean recall of 0.5 at α = 0.1 comes from
  only 1–2 malignant test cases in the two seeds where it is defined. Best-to-worst hospital recall
  gaps are 0.10–0.17 for IID, α = 10 and α = 1, and 0.46 for every method at α = 0.1.
  No method systematically narrowed the gap between hospitals.

## 2. Suggested initial configuration (lr = 0.001, 50 rounds)

| Distribution | Centralized | FedAvg | FedProx | FedHAW (repo defaults) |
|---|---|---|---|---|
| IID – accuracy | 0.915 ± 0.031 | 0.655 ± 0.171 | 0.655 ± 0.171 | 0.671 ± 0.165 |
| α=10 – accuracy | 0.915 ± 0.031 | 0.659 ± 0.174 | 0.659 ± 0.174 | 0.682 ± 0.130 |
| α=1 – accuracy | 0.915 ± 0.031 | 0.671 ± 0.171 | 0.671 ± 0.171 | 0.760 ± 0.071 |
| α=0.1 – accuracy | 0.915 ± 0.031 | 0.748 ± 0.088 | 0.748 ± 0.088 | 0.860 ± 0.000 |
| α=0.1 – recall | 0.771 ± 0.082 | 0.344 ± 0.259 | 0.344 ± 0.259 | 0.698 ± 0.082 |

* With lr = 0.001, 50 rounds of one local epoch is far from convergence. Federated
  models are unstable across seeds: in some seeds FedAvg predicts almost everything
  malignant (accuracy 0.42, recall 1.0), and at α = 0.1 seed 123 it predicts everything
  benign (recall 0.0).
* **In this under-trained regime FedHAW helped.** Its mean accuracy was higher than
  FedAvg's in every distribution (+0.016 to +0.112). At α = 0.1 it was higher in all
  3 seeds, with recall rising from 0.34 to 0.70. Here γ acts as a learned global step
  size that grows when successive updates point in a consistent direction, which speeds
  up progress when local steps are too small. This matches the role of γ in the FedHAW
  paper, where the MNIST setting also uses μ = 0.001.
* The centralized model, with 25 steps per epoch instead of about 5 per hospital per
  round, is much further along (0.915).
* The hospital aggregation weights a_k stayed within 0.03 of N_k/N in both configurations
  (`results/plots/*/fedhaw_*_weights_seed42.png`). With 5 small hospitals and a small
  model, the λ hypergradient is weak. The visible effect of FedHAW on WDBC comes
  almost entirely from γ.

## 3. Answer to the research question

On WDBC with five simulated hospitals:

1. Original FedHAW runs stably (all values finite) under IID and all Dirichlet
   settings, and reproduces the original implementation exactly.
2. When local training is well configured (lr = 0.05), FedHAW gave no accuracy,
   recall or AUC advantage over FedAvg. Validation-based tuning drives its step sizes
   towards zero, which makes it equivalent to FedAvg. With the repository's default
   step sizes, it is slightly overconfident and slightly less accurate.
3. When local training is too slow (lr = 0.001), FedHAW's adaptive global scale
   improves accuracy and recall over FedAvg, most clearly under severe non-IID skew
   (α = 0.1).
4. All gaps are a few test patients and rest on 3 seeds. They are descriptive, not
   statistically established.

## 4. Limitations

* WDBC is small (569 cases) and almost linearly separable. Test sets have 86 patients,
  and hospital subsets have 1–52.
* Only 3 seeds; no significance testing is possible at this sample size.
* The hospitals are simulated from a single-source dataset. Only label and quantity
  skew are modelled; there is no real feature or acquisition shift between sites.
* Final-round models, a fixed 0.5 threshold and no calibration. One shared set of
  hyperparameters for all distributions.
* A Dirichlet minimum-size rule (≥ 5 samples per hospital) was needed. Without it,
  some hospitals at α = 0.1 would be empty.
* This is an academic research prototype, not a clinically validated diagnostic system.
