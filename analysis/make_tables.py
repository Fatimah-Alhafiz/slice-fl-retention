"""Recompute the paper's summary numbers from the per-run CSVs in ../results.

Run from the repository root:  python analysis/make_tables.py
Bootstrap: percentile, 4,000 resamples, seed = resampling unit (paired differences per seed).
"""
import os
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, float)
    m = rng.choice(x, (4000, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def load(name):
    return pd.read_csv(os.path.join(RES, name, 'fl_grid.csv'))


# Five-seed main grid = seeds 0-2 (results_v5_*) + seeds 3-4 (results_v7_main_*), identical code path
main = pd.concat([pd.concat([load(f'results_v5_{d}'), load(f'results_v7_main_{d}')]).assign(dataset=d)
                  for d in ('motionsense', 'wisdm')], ignore_index=True)

print('Table IV - multi-round LiRA AUC, single-use FIFO (K = 1), mean [95% CI]')
for (d, o), f in main[(main.policy == 'fifo') & (main.K == 1)].groupby(['dataset', 'order']):
    print(f'  {d:12s} {o:9s} B=4: {ci(f[f.B == 4].lira_agg_auc)}   B=32: {ci(f[f.B == 32].lira_agg_auc)}')

print('\nSection VI-F/G - paired differences per seed, multi-round LiRA AUC')
for (d, o), f in main.groupby(['dataset', 'order']):
    for B in (4, 8, 16, 32):
        k1 = f[(f.policy == 'fifo') & (f.B == B) & (f.K == 1)].set_index('seed').lira_agg_auc
        k4 = f[(f.policy == 'fifo') & (f.B == B) & (f.K == 4)].set_index('seed').lira_agg_auc
        rs = f[(f.policy == 'reservoir') & (f.B == B)].set_index('seed').lira_agg_auc
        print(f'  {d:12s} {o:9s} B={B:2d}  K4-K1 {ci(k4 - k1)}   reservoir-slice {ci(rs - k1)}')

print('\nSupplementary Table S1 - mean AUC (TPR@5%FPR) at B = 4 / B = 32')
for (d, o), f in main.groupby(['dataset', 'order']):
    cells = []
    for lab, q in (('FIFO K=1', (f.policy == 'fifo') & (f.K == 1)), ('FIFO K=4', (f.policy == 'fifo') & (f.K == 4)),
                   ('Reservoir', f.policy == 'reservoir')):
        g = f[q]
        cells.append(lab + ' ' + ' / '.join(f"{g[g.B == B].lira_agg_auc.mean():.2f} ({g[g.B == B].lira_agg_tpr5.mean():.2f})"
                                            for B in (4, 32)))
    print(f'  {d:12s} {o:9s} ' + ' | '.join(cells))

print('\nTable VI - macro-F1 under DP (ordered streams, 3 seeds), mean [95% CI]')
for d in ('motionsense', 'wisdm'):
    dp = pd.concat([load(f'results_v7_dp_{d}'), load(f'results_v6_rq4_{d}').query('eps == 4')])
    for (eps, K), g in dp.groupby(['eps', 'K']):
        print(f'  {d:12s} eps={eps:g} K={int(K)}  ' + '  '.join(f'B={B}: {ci(g[g.B == B].macro_f1)}' for B in (4, 8, 16, 32)))

print('\nTable V - correlation dose-response (B = 8, means over 3 seeds)')
for d in ('motionsense', 'wisdm'):
    c = load(f'results_v6_corr_{d}')
    print(c.groupby(['K', 'order'])[['run_len', 'lira_agg_auc', 'mia_single', 'macro_f1']].mean().round(3).to_string())
