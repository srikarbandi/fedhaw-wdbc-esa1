# Original FedHAW: reproduction notes

## 1. Source inspected

* Repository: <https://github.com/a-nakai-k/FedHAW>, cloned to `external/FedHAW`.
  Commit `6f565520ba51e3e673520e1e22eaa5f9f0ed244c` (8 May 2026).
* Paper: A. Nakai-Kasai and T. Wadayama, *Federated learning with hypergradient-based
  online update of aggregation weights*, arXiv:2605.00458, 2026.
* Files read in full:

| File | What was learned |
|---|---|
| `README.md` | Methods compared, datasets, per-dataset step sizes (MNIST lr_γ=1e-3 / lr_λ=1e-2 with SGD μ=1e-3; CIFAR-10 1e-5/1e-2; Dogs 1e-7/1e-5) |
| `run_serverside.py` | Initialisation of γ and λ, the round loop, how `params_candidates_pre` is passed between rounds, shared `init_params` across methods, one `make_model(args)` per round |
| `functions/training.py` | `train_fedhaw`, `train_fedavg`, `train_fedprox`, `_local_train`, `_make_optimizer`, `softmax_vec` |
| `functions/networks.py` | MLP (`Net`, `Net_dogs`): 3 Linear layers, ReLU, `log_softmax` output |
| `functions/datasets.py` | Pre-partitioned client `.npz` files. No partition script is shipped |
| `functions/utils.py` | `test()` (argmax accuracy), plotting helpers |
| `functions/training_witherror.py`, `run_*_witherror.py` | Skimmed only: client-error experiments, outside this project's scope |

## 2. The original algorithm, statement by statement

Initialisation (`run_serverside.py`):

```python
Nk = [len(train_datasets[k]) for k in range(args.K)]
init_lambdat = [np.log(Nk[k] / N) for k in range(args.K)]   # λ_k = log(N_k/N)
init_gammat  = np.log(1.0)                                   # γ = 0
params_candidates_pre = None
```

One round of `train_fedhaw(..., loop=t, params_candidates_pre, gammat, lambdat, lr_gamma, lr_lambda)`:

```python
# 1. local training: every client starts from params_vector_pre (θ^t)
params_candidates = [_local_train(...)[0] for node in range(K)]        # θ_k^t
with torch.no_grad():
    if loop > 0:                                                       # 2. skipped in round 0
        sm_lambdatpre = softmax_vec(lambdat)                           # a^(t-1) from OLD λ
        params_tmpsum = Σ_k sm_lambdatpre[k] * params_candidates[k]
        tmpvec = params_vector_pre - params_tmpsum                     # d = θ^t − Σ a_k θ_k^t
        gammat = gammat - lr_gamma * exp(gammat) / mu * <tmpvec, params_vector_pre>
        lambdat = [lambdat[k] - lr_lambda * exp(2*gammat) * a_k (1-a_k) / mu
                   * <tmpvec, params_candidates_pre[k]>  for k]        # uses NEW γ, θ_k^(t-1)
    sm_lambdat = softmax_vec(lambdat)                                  # 3. a^t from NEW λ
    params_vector_global = Σ_k sm_lambdat[k] * params_candidates[k]    # 4. aggregate
    params_vector_global = params_vector_global * exp(gammat)          # 5. global scaling
return params_vector_global, loss, gammat, lambdat, params_candidates  # 6. θ_k^t kept for t+1
```

In equations, with μ the local learning rate and ⟨·,·⟩ the inner product over the full
flattened parameter vector:

* a_k = exp(λ_k) / Σ_j exp(λ_j)
* d^t = θ^t − Σ_k a_k^{(t−1)} θ_k^t
* γ ← γ − lr_γ · e^{γ} / μ · ⟨d^t, θ^t⟩
* λ_k ← λ_k − lr_λ · e^{2γ_new} · a_k^{(t−1)}(1 − a_k^{(t−1)}) / μ · ⟨d^t, θ_k^{t−1}⟩
* θ^{t+1} = e^{γ_new} · Σ_k a_k^{(t)} θ_k^t

### Points that are easy to get wrong (all reproduced exactly)

1. **Update order.** γ is updated first. The λ update then uses the **updated** γ,
   because `gammat` has already been reassigned before `exp(2*gammat)` is evaluated.
2. **Two different weight vectors.** `d` uses softmax(λ) from *before* this round's
   update. The final aggregation uses softmax(λ) from *after* it.
3. **Previous-round candidates.** The λ gradient uses `params_candidates_pre`, the client
   vectors from round t−1. The γ gradient uses the global vector θ^t.
4. **Round 0.** `loop > 0` is false, so round 0 is FedAvg with weights N_k/N, and
   γ = 0 (scale e^0 = 1).
5. **Diagonal Jacobian.** The λ update uses only a_k(1 − a_k), the diagonal of the
   softmax Jacobian, as in the original code.
6. **Global scaling.** e^{γ} multiplies the entire aggregated parameter vector,
   including biases, every round.
7. **Softmax form.** `np.exp(x)/np.sum(np.exp(x))`, with no max-shift and no clipping
   of γ or λ.
8. **μ dependence.** Both hypergradients are divided by the local learning rate μ, so
   the effective size of a step-size setting depends on μ.

## 3. Mapping to this project

| Original | This project |
|---|---|
| server part of `train_fedhaw` | `src/federated/aggregation.py::fedhaw_update` (same statements, same variable names) |
| γ, λ, `params_candidates_pre` kept by `run_serverside.py` | `src/federated/fedhaw.py::FedHAW` (set in `reset`, updated in `aggregate`, never re-initialised) |
| init (`np.log(Nk/N)`, `np.log(1.0)`) | `aggregation.fedhaw_init` |
| `_local_train` | `src/federated/client.py::local_train` |
| `_make_optimizer` | `client.make_optimizer` |
| round loop in `run_serverside.py` | `src/federated/server.py::run_federated` |
| `parameters_to_vector` / `vector_to_parameters` | `src/federated/params.py` |

## 4. Adaptations for WDBC (all documented, none touch the FedHAW equations)

| # | Adaptation | Reason |
|---|---|---|
| 1 | MNIST/Dogs input reshaping removed from local training | WDBC is already a 30-dimensional vector |
| 2 | Model: MLP 30→64→32→2 with raw logits + `CrossEntropyLoss`, instead of 784→128→128→10 with `log_softmax` + `NLLLoss` | Project brief. CrossEntropyLoss(z) ≡ NLLLoss(log_softmax(z)), so the training objective is unchanged |
| 3 | K = 5 hospitals instead of 10 | Project brief |
| 4 | Client data produced by our own IID / Dirichlet partitioner | The repository ships pre-partitioned files and no partition script |
| 5 | Per-round validation metrics, timings, γ and weights are logged | Analysis only. The server returns the same values |
| 6 | Local training calls `model.train()` | No effect with dropout = 0 (the default). Needed only if dropout is enabled |
| 7 | Empty-client guard: a client with 0 samples returns the global vector unchanged | Never triggered with the default minimum client size (5) |
| 8 | Step sizes lr_γ and lr_λ chosen on validation data, plus a run with the repository defaults (1e-3, 1e-2) | The authors also set them per dataset (see README) |

Stable softmax was considered but **not** used. It is mathematically equivalent but
changes floating-point rounding, which would break the bit-exact test in §5. The λ
values here stay small (|λ| ≈ 5 at most, since log(5/397) ≈ −4.4 for the smallest hospital),
so overflow cannot occur.

## 5. Verification against the original code (`tests/test_fedhaw.py`)

* `test_full_training_bit_exact_vs_original`: imports `external/FedHAW/functions/training.py`
  and runs 6 rounds of the **original** `train_fedhaw` (driven like `run_serverside.py`)
  and 6 rounds of our server loop, from the same RNG state, data, model and loss. Global
  parameters (`torch.equal`), γ and λ are **identical in every round**. It is checked at
  the repository defaults and at large step sizes (lr_γ = 0.05, lr_λ = 5), where the
  adaptation is clearly active.
* `test_aggregation_matches_original_with_synthetic_candidates`: replaces the original
  `_local_train` with fixed synthetic client vectors and compares the server update
  alone. Identical.
* Further tests cover: γ = 0 at start; λ = log(N_k/N); round 0 skips the hypergradient
  (equals FedAvg); later rounds change γ and every λ_k; the λ update uses the new γ;
  previous candidates are stored and used; weights sum to 1; aggregation uses the
  current candidates; e^{γ} scaling is applied; dimensions are preserved; training stays
  finite. There is also an independent float64 NumPy re-derivation of the equations.

## 6. Differences from the original that remain

* **FedProx baseline weighting** (not FedHAW): the original `train_fedprox` weights
  clients by number of mini-batches. We use N_k/N (FedProx paper and FedAvg).
* **FedAvg** is identical to the original `train_fedavg`.
* **FedHAW: none.**
