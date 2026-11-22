# Buffer Size, Reuse, and Retention as Privacy Parameters in Streaming Federated Learning for IoT

Code and per-seed results for the paper by **Fatimah Saeed Alhafiz** (University of Bisha), submitted to the *IEEE Internet of Things Journal*.

The experiments study how a client's sample buffer in streaming federated learning — its size **B**, the number of times each sample is reused **K**, and the eviction policy (single-use "slice" FIFO, FIFO with reuse, reservoir, keep-all) — changes membership leakage from the model updates it sends, and what that costs in accuracy, under several arrival orders, with and without secure aggregation and DP-SGD.

## Repository layout

```
src/        experiment scripts (PyTorch, CPU is enough)
analysis/   make_tables.py: recomputes the paper's tables and confidence intervals from results/
results/    per-seed CSV outputs of every run reported in the paper
data/       place the datasets here (not redistributed; see data/README.md)
```

## Script → paper mapping

| Script | Settings | Produces |
|---|---|---|
| `src/exp_grid_and_exposure.py` | MotionSense | Table III (B×K grid, baselines) and the exposure / age simulation in Sec. VI-B |
| `src/exp_arrival_order.py` | `DATASET=motionsense\|wisdm` | Arrival-order control (ordered, offset, block, shuffled) |
| `src/exp_main_and_dp.py` | `EXP=main SEEDS=0,1,2,3,4` | Table IV, Supplementary Table S1, LiRA vs. SecAgg aggregate |
| `src/exp_main_and_dp.py` | `EXP=dp EPS=0,1,8` | Table VI (DP-SGD sweep, together with ε = 4 from below) |
| `src/exp_correlation_and_protection.py` | `EXP=corr` | Table V and Fig. 1 (temporal-correlation dose-response) |
| `src/exp_correlation_and_protection.py` | `EXP=rq4` | DP-SGD at ε = 4 and gradient-alignment MIA under SecAgg |
| `analysis/make_tables.py` | — | Bootstrap CIs, paired differences, Table S1 from `results/` |

In `results/`, the five-seed main grid is seeds 0–2 in `results_v5_*` and seeds 3–4 in `results_v7_main_*`; both were produced by the same code path, so `EXP=main SEEDS=0,1,2,3,4` regenerates all five.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the datasets as described in `data/README.md`.

## Running

Scripts write their output folder into the current directory. Run them from `src/`:

```bash
cd src
DATASET=motionsense EXP=main SEEDS=0,1,2,3,4 python exp_main_and_dp.py
DATASET=wisdm       EXP=main SEEDS=0,1,2,3,4 python exp_main_and_dp.py
DATASET=wisdm       EXP=dp   EPS=0,1,8       python exp_main_and_dp.py
DATASET=wisdm       EXP=corr                 python exp_correlation_and_protection.py
DATASET=wisdm       EXP=rq4                  python exp_correlation_and_protection.py
DATASET=motionsense                          python exp_arrival_order.py
python exp_grid_and_exposure.py
```

Optional environment variables: `MOTIONSENSE_DIR`, `WISDM_FILE` (dataset paths), `THREADS` (CPU threads, default 2).

To recompute the paper's numbers from the stored results:

```bash
python analysis/make_tables.py
```

## Main settings

- Model: two 1-D convolution layers (32 channels, kernel 5) + global average pooling + linear head (7,302 parameters on MotionSense, 5,862 on WISDM); 3 local gradient steps, learning rate 0.3.
- Clients: MotionSense 24 subjects; WISDM v1.1, 40-sample windows, users with ≥ 400 windows (33 clients × 407 windows).
- Attacks: per-update online LiRA (16 shadow models, multi-round score = summed log-likelihood ratio) and a gradient-alignment membership test, each against three target clients per run.
- DP-SGD: event-level, zCDP accounting, δ = 10⁻⁵, clipping norm 1.
- Statistics: percentile bootstrap over seeds, 4,000 resamples; paired differences per seed.

## Citation

See `CITATION.cff`. Until the paper is published, please cite the repository.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses and are not included.
