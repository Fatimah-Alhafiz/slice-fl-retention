# %% [markdown]
# # Pilot round 4: second dataset (WISDM) — same design as round 3
# # Pilot round 3 (control): does arrival ORDER explain FIFO vs reservoir?
# Same code as round 2, run under three arrival orders: ordered (round 2), offset (each client's ordered stream
# rotated by a random amount → clients no longer see the same activity at the same time), shuffled (i.i.d. arrivals).
# Also logs sample AGE on the device (rounds since arrival) inside the FL runs.
# Round-2 notes:
# # Pilot round 2: buffer size (B) and retention (K) as privacy parameters in streaming FL (G1)
# **Dataset:** MotionSense (Malekzadeh et al., IoTDI 2019), MIT licence, 24 subjects = 24 clients, 12 channels @ 50 Hz.
#
# **Changes from round 1**
# 1. Gradient clipping in local training (round 1 had one diverged run).
# 2. Accuracy logged every round → utility compared at an **equal number of rounds** as well as at equal data.
# 3. Macro-F1 added (activities are imbalanced).
# 4. **Paired** membership inference: every member is paired with a fixed non-member from the same recording,
#    scored on the same updates → single-update and aggregated AUC are comparable across all policies.
# 5. Gradient inversion reported against a **random-guess baseline** (gain = attack − baseline).
# 6. **Baseline buffer policies** from streaming FL: reservoir sampling and keep-all (train on a random B-subset).
# 7. Exposure simulation: three arrival scenarios (steady / burst / variable) and **sample age on device**.
# 8. 5 seeds, results saved after every run (resume after a Colab disconnect), bootstrap 95 % CIs and PASS/FAIL checks.
#
# Model assumptions (as in the analytic model): full-buffer local update, amount trigger, one update per round.
# FIFO(B, K): C = B/K new samples per round; each sample enters exactly K updates and is then deleted (K = 1 = slice).

# %%
import os, glob, zipfile, urllib.request, random, math, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from sklearn.metrics import roc_auc_score, f1_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

DATASET = os.environ.get('DATASET', 'wisdm')        # 'motionsense' or 'wisdm'
WISDM_FILE = os.environ.get('WISDM_FILE', '../data/WISDM_ar_v1.1_raw.txt')
MIN_STREAM = 400            # WISDM: keep users with >= 400 stream windows
QUICK = False            # True = tiny grid, 1 seed (~3 min) to check that everything runs
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = f'results_v4_{DATASET}'; os.makedirs(OUT, exist_ok=True)

CFG = dict(
    WIN=64 if DATASET == 'motionsense' else 40, TEST_FRAC=0.2,
    B_GRID=[4, 8, 16, 32] if not QUICK else [4, 16],
    K_GRID=[1, 4],
    BASE_C=4,                                 # baselines (reservoir, keep-all) receive C = 4 new samples per round
    BASE_B=[4, 8, 16, 32] if not QUICK else [16],
    ORDERS=['ordered', 'offset', 'shuffled'],
    LOCAL_STEPS=3, LR=0.3, CLIP=1.0,
    STREAM_BUDGET=None if not QUICK else 160,
    TARGETS=[1, 2, 3],                        # clients whose updates the server-side attacker analyses
    R_EQ=19 if not QUICK else 5,              # equal-rounds comparison point (fewest rounds in the grid)
    INV_ROUNDS=2, INV_ITERS=400 if not QUICK else 100, INV_B_MAX=16,
    SEEDS=[0] if QUICK else [0, 1, 2],
)
print(DEV, CFG)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# %% Load dataset (DATASET = 'motionsense' or 'wisdm')
def windows(x, w):
    n = len(x) // w
    return x[:n * w].reshape(n, w, -1).transpose(0, 2, 1)

per = {}
if DATASET == 'motionsense':
    URL = 'https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip'
    ROOT = os.environ.get('MOTIONSENSE_DIR', '../data/motionsense')
    if not glob.glob(f'{ROOT}/**/sub_1.csv', recursive=True):
        os.makedirs(ROOT, exist_ok=True); zp = f'{ROOT}/A.zip'
        if not os.path.exists(zp): urllib.request.urlretrieve(URL, zp)
        zipfile.ZipFile(zp).extractall(ROOT)
    ACTS = ['dws', 'ups', 'wlk', 'jog', 'std', 'sit']
    COLS = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x', 'gravity.y', 'gravity.z',
            'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
            'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']
    for f in [f for f in glob.glob(f'{ROOT}/**/sub_*.csv', recursive=True) if '__MACOSX' not in f]:
        act, trial = os.path.basename(os.path.dirname(f)).split('_')
        sub = int(os.path.basename(f)[4:-4])
        x = pd.read_csv(f)[COLS].values.astype(np.float32)
        cut = int(len(x) * (1 - CFG['TEST_FRAC']))
        per.setdefault(sub, []).append((int(trial), ACTS.index(act), windows(x[:cut], CFG['WIN']), windows(x[cut:], CFG['WIN'])))
else:
    # WISDM v1.1 (Kwapisz et al., 2011): 36 users, phone accelerometer x/y/z at 20 Hz, 6 activities.
    # Official source: https://www.cis.fordham.edu/wisdm/dataset.php  (place WISDM_ar_v1.1_raw.txt next to this notebook)
    # "recording" = a contiguous run of one activity by one user, in file (session) order.
    ACTS = ['Downstairs', 'Upstairs', 'Walking', 'Jogging', 'Standing', 'Sitting']
    rows = []
    for line in open(WISDM_FILE):
        for rec in line.strip().split(';'):
            p = rec.strip().strip(',').split(',')
            if len(p) != 6: continue
            try: rows.append((int(p[0]), p[1], float(p[3]), float(p[4]), float(p[5])))
            except ValueError: pass
    d = pd.DataFrame(rows, columns=['u', 'a', 'x', 'y', 'z'])
    d['seg'] = (d.u.ne(d.u.shift()) | d.a.ne(d.a.shift())).cumsum()
    for sg, g in d.groupby('seg', sort=True):
        x = g[['x', 'y', 'z']].values.astype(np.float32)
        if len(x) < 5 * CFG['WIN']: continue
        cut = int(len(x) * (1 - CFG['TEST_FRAC']))
        per.setdefault(int(g.u.iloc[0]), []).append((int(sg), ACTS.index(g.a.iloc[0]), windows(x[:cut], CFG['WIN']), windows(x[cut:], CFG['WIN'])))
    n_win = {u: sum(len(r[2]) for r in recs) for u, recs in per.items()}
    per = {u: r for u, r in per.items() if n_win[u] >= MIN_STREAM}       # drop users with very short streams

CLIENTS = {}
for sub, recs in per.items():
    recs.sort(key=lambda r: r[0])
    cat = lambda j: np.concatenate([r[j] for r in recs])
    lab = lambda j, v: np.concatenate([[r[v]] * len(r[j]) for r in recs])
    CLIENTS[sub] = dict(Xs=cat(2), ys=lab(2, 1), rs=lab(2, 0), Xt=cat(3), yt=lab(3, 1), rt=lab(3, 0))
NCH = next(iter(CLIENTS.values()))['Xs'].shape[1]

allX = np.concatenate([c['Xs'] for c in CLIENTS.values()])
MU, SD = allX.mean((0, 2), keepdims=True)[0], allX.std((0, 2), keepdims=True)[0] + 1e-6
for c in CLIENTS.values():
    for k in ('Xs', 'Xt'): c[k] = torch.tensor((c[k] - MU) / SD)
    for k in ('ys', 'yt'): c[k] = torch.tensor(c[k], dtype=torch.long)
SUBS = sorted(CLIENTS)
ORIG = {s: {k: CLIENTS[s][k].clone() if torch.is_tensor(CLIENTS[s][k]) else CLIENTS[s][k].copy() for k in ('Xs', 'ys', 'rs')} for s in SUBS}
def set_order(order, seed):
    rng = np.random.default_rng(500 + seed)
    for s in SUBS:
        n = len(ORIG[s]['ys'])
        perm = (np.arange(n) if order == 'ordered' else np.roll(np.arange(n), rng.integers(n)) if order == 'offset' else rng.permutation(n))
        for k in ('Xs', 'ys', 'rs'): CLIENTS[s][k] = ORIG[s][k][perm]
    NM_MAP.clear()
NM_MAP = {}
N_STREAM = min(len(CLIENTS[s]['ys']) for s in SUBS)
if CFG['STREAM_BUDGET']: N_STREAM = min(N_STREAM, CFG['STREAM_BUDGET'])
XT_ALL = torch.cat([CLIENTS[s]['Xt'] for s in SUBS]).to(DEV); YT_ALL = torch.cat([CLIENTS[s]['yt'] for s in SUBS]).to(DEV)
print(f'{len(SUBS)} clients, stream windows per client: {N_STREAM}, test windows: {len(YT_ALL)}')

# %% Model + helpers
class Net(nn.Module):
    def __init__(s, c=None, k=6):
        c = c or NCH
        super().__init__()
        s.c1 = nn.Conv1d(c, 32, 5, padding=2); s.c2 = nn.Conv1d(32, 32, 5, padding=2); s.fc = nn.Linear(32, k)
    def forward(s, x):
        x = F.relu(s.c1(x)); x = F.max_pool1d(x, 2); x = F.relu(s.c2(x)); return s.fc(x.mean(-1))

def flat(ps): return torch.cat([p.reshape(-1) for p in ps])

def per_sample_grads(model, params, X, Y):
    def loss(p, x, y): return F.cross_entropy(functional_call(model, p, (x[None],)), y[None])
    g = vmap(grad(loss), in_dims=(None, 0, 0))(params, X, Y)
    return torch.cat([g[k].reshape(len(X), -1) for k in params], 1)

@torch.no_grad()
def evaluate(model):
    p = model(XT_ALL).argmax(1)
    return (p == YT_ALL).float().mean().item(), f1_score(YT_ALL.cpu(), p.cpu(), average='macro')

# %% Streaming FL with a buffer policy
#   fifo(B, K):  C = B/K arrive, train on the B oldest, delete the C oldest  → every sample used exactly K times
#   reservoir(B): C arrive, reservoir sampling keeps a uniform sample of the stream in B slots, train on all B
#   keepall(B):   C arrive, nothing is deleted, train on a random B-subset of everything held
def run_fl(policy, B, C, seed):
    seed_all(seed); rng = np.random.default_rng(seed)
    g = Net().to(DEV)
    ptr = {s: 0 for s in SUBS}; held = {s: [] for s in SUBS}
    uses = {s: np.zeros(N_STREAM, int) for s in SUBS}
    logs = {t: [] for t in CFG['TARGETS']}
    acc_curve, held_trace, rounds = [], [], 0
    age_trace, age_max = [], 0
    while not any(ptr[s] + C > N_STREAM for s in SUBS):
        w0 = {k: v.detach().clone() for k, v in g.named_parameters()}
        states, n_train = None, 0
        for s in SUBS:
            new = list(range(ptr[s], ptr[s] + C)); ptr[s] += C
            if policy == 'reservoir':
                for i in new:
                    if len(held[s]) < B: held[s].append(i)
                    else:
                        j = rng.integers(0, i + 1)
                        if j < B: held[s][j] = i
            else:
                held[s] += new
            held_trace.append(len(held[s]))
            if held[s]:
                a = (ptr[s] - np.asarray(held[s])).astype(float)   # age in stream windows since arrival (= data-time)
                age_trace.append(a.mean()); age_max = max(age_max, a.max())
            if len(held[s]) < B: continue
            batch = (held[s][:B] if policy == 'fifo' else
                     list(held[s]) if policy == 'reservoir' else
                     list(rng.choice(held[s], B, replace=False)))
            idx = torch.tensor(batch); uses[s][idx.numpy()] += 1
            X = CLIENTS[s]['Xs'][idx].to(DEV); Y = CLIENTS[s]['ys'][idx].to(DEV)
            m = Net().to(DEV); m.load_state_dict(g.state_dict())
            opt = torch.optim.SGD(m.parameters(), lr=CFG['LR'])
            for _ in range(CFG['LOCAL_STEPS']):
                opt.zero_grad(); F.cross_entropy(m(X), Y).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), CFG['CLIP']); opt.step()
            if s in logs:
                delta = flat([w0[k] - p.detach() for k, p in m.named_parameters()])
                logs[s].append(dict(w0={k: v.cpu() for k, v in w0.items()}, delta=delta.cpu(), members=list(batch)))
            sd = m.state_dict()
            states = {k: v.clone() for k, v in sd.items()} if states is None else {k: states[k] + sd[k] for k in sd}
            n_train += 1
            if policy == 'fifo': held[s] = held[s][C:]
        if n_train:
            g.load_state_dict({k: v / n_train for k, v in states.items()}); rounds += 1
            acc_curve.append(evaluate(g)[0])
    u = np.concatenate([uses[s][:ptr[s]] for s in SUBS])
    return g, logs, rounds, acc_curve, dict(age_mean=float(np.mean(age_trace)), age_max=float(age_max), held_mean=float(np.mean(held_trace)), held_max=int(np.max(held_trace)),
                                            K_eff=float(u[u > 0].mean()), never_used=float((u == 0).mean()))

# %% Attack 1: paired gradient-alignment membership inference
def nm_map(t, seed):
    """Fixed non-member for every stream window: a held-out window from the same recording."""
    if (t, seed) not in NM_MAP:
        rng = np.random.default_rng(1000 + seed); c = CLIENTS[t]
        NM_MAP[(t, seed)] = np.array([rng.choice(np.where(c['rt'] == r)[0]) for r in c['rs']])
    return NM_MAP[(t, seed)]

def mia(logs, seed):
    model = Net().to(DEV)
    s_single, y_single, per_m, per_n = [], [], {}, {}
    for t, L in logs.items():
        c = CLIENTS[t]; nm = nm_map(t, seed)
        for e in L:
            params = {k: v.to(DEV) for k, v in e['w0'].items()}
            mem = np.array(e['members']); non = nm[mem]
            X = torch.cat([c['Xs'][mem], c['Xt'][non]]).to(DEV); Y = torch.cat([c['ys'][mem], c['yt'][non]]).to(DEV)
            sc = F.cosine_similarity(per_sample_grads(model, params, X, Y), e['delta'].to(DEV)[None], dim=1).cpu().numpy()
            sm, sn = sc[:len(mem)], sc[len(mem):]
            s_single += list(sm) + list(sn); y_single += [1] * len(sm) + [0] * len(sn)
            for i, a, b in zip(mem, sm, sn):
                per_m.setdefault((t, i), []).append(a); per_n.setdefault((t, i), []).append(b)
    # aggregated: average each member (and its paired non-member) over every update the member entered
    agg = [np.mean(v) for v in per_m.values()] + [np.mean(v) for v in per_n.values()]
    y_agg = [1] * len(per_m) + [0] * len(per_n)
    return roc_auc_score(y_single, s_single), roc_auc_score(y_agg, agg)

# %% Attack 2: gradient inversion vs random-guess baseline
def matched_corr(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
    corr = a @ b.T / a.shape[1]; r, c = linear_sum_assignment(-corr)
    return float(corr[r, c].mean())

def invert(entry, client, seed):
    model = Net().to(DEV)
    params = {k: v.to(DEV).requires_grad_(True) for k, v in entry['w0'].items()}
    target = entry['delta'].to(DEV)
    mem = torch.tensor(entry['members']); X = client['Xs'][mem].to(DEV); Y = client['ys'][mem].to(DEV)
    torch.manual_seed(seed); xh = torch.randn_like(X, requires_grad=True)
    guess = torch.randn_like(X)                      # baseline: random signals, same matching procedure
    opt = torch.optim.Adam([xh], lr=0.05)
    for _ in range(CFG['INV_ITERS']):
        opt.zero_grad()
        gv = flat(torch.autograd.grad(F.cross_entropy(functional_call(model, params, (xh,)), Y),
                                      list(params.values()), create_graph=True))
        loss = 1 - F.cosine_similarity(gv, target, dim=0) + 1e-3 * (xh[..., 1:] - xh[..., :-1]).abs().mean()
        loss.backward(); opt.step()
    Xn = X.cpu().numpy()
    return matched_corr(Xn, xh.detach().cpu().numpy()), matched_corr(Xn, guess.cpu().numpy())

# %% Run the grid (saved after every run; re-running the cell resumes)
CSV = f'{OUT}/fl_grid.csv'
done = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
JOBS = [('fifo', B, B // K, K) for B in CFG['B_GRID'] for K in CFG['K_GRID'] if B % K == 0] + \
       [('reservoir', B, CFG['BASE_C'], np.nan) for B in CFG['BASE_B']]
for order in CFG['ORDERS']:
  for seed in CFG['SEEDS']:
    set_order(order, seed)
    for policy, B, C, K in JOBS:
        if len(done) and ((done.order == order) & (done.seed == seed) & (done.policy == policy) & (done.B == B) & (done.C == C)).any(): continue
        t0 = time.time()
        g, logs, R, curve, ex = run_fl(policy, B, C, seed)
        acc, f1 = evaluate(g)
        auc1, aucA = mia(logs, seed)
        inv = base = np.nan
        if B <= CFG['INV_B_MAX']:
            L = logs[CFG['TARGETS'][0]]; picks = np.linspace(len(L) // 2, len(L) - 1, CFG['INV_ROUNDS']).astype(int)
            r = [invert(L[p], CLIENTS[CFG['TARGETS'][0]], seed) for p in picks]
            inv, base = float(np.mean([a for a, _ in r])), float(np.mean([b for _, b in r]))
        row = dict(order=order, seed=seed, policy=policy, B=B, C=C, K=K, rounds=R, acc=acc, macro_f1=f1,
                   acc_at_Req=curve[CFG['R_EQ'] - 1] if len(curve) >= CFG['R_EQ'] else np.nan,
                   mia_single=auc1, mia_agg=aucA, inv_corr=inv, inv_base=base, inv_gain=inv - base, **ex)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(CSV, index=False)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, f'{time.time() - t0:.0f}s', flush=True)
df = done.copy()

# %% Summary: policy × order
S = df.groupby(['order', 'policy', 'B', 'C']).agg(K=('K', 'first'), acc=('acc', 'mean'), f1=('macro_f1', 'mean'),
        mia1=('mia_single', 'mean'), miaA=('mia_agg', 'mean'), inv=('inv_gain', 'mean'),
        age=('age_mean', 'mean'), age_max=('age_max', 'mean'), Keff=('K_eff', 'mean')).round(3)
S.to_csv(f'{OUT}/summary_by_order.csv'); pd.set_option('display.width', 250); print(S.to_string())

# %% Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 4, figsize=(20, 11))
for i, order in enumerate(CFG['ORDERS']):
    d0 = df[df.order == order].groupby(['policy', 'B', 'C']).mean(numeric_only=True).reset_index()
    for j, (col, t) in enumerate([('acc', 'Accuracy'), ('mia_single', 'MIA single'), ('inv_gain', 'Inversion gain'), ('age_mean', 'Mean age (windows of data-time)')]):
        a = ax[i, j]
        for K in CFG['K_GRID']:
            d = d0[(d0.policy == 'fifo') & (d0.K == K)]; a.plot(d.B, d[col], 'o-', label=f'FIFO K={K}')
        d = d0[d0.policy == 'reservoir']; a.plot(d.B, d[col], 's--', label='reservoir')
        a.set_xscale('log', base=2); a.set_title(f'{order}: {t}'); a.grid(alpha=.3)
ax[0, 0].legend(); plt.tight_layout(); plt.savefig(f'{OUT}/order_control.png', dpi=150); plt.show()
