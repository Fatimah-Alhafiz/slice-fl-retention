# %% [markdown]
# # Pilot round 7 (scale-up): EXP=main (5 seeds, ordered + shuffled, FIFO and reservoir) and EXP=dp (eps sweep);
# adds LiRA against the secure-aggregation aggregate (server sees only the mean update of all clients).
# # Pilot round 6: (a) temporal-correlation dose-response, (b) RQ4 sizing under DP and secure aggregation
# EXP=corr : B=8, K in {1,4}, block-shuffled streams (block m keeps runs of ~m windows) → leakage vs label-run length
# EXP=rq4  : ordered streams, B x K grid, no protection vs sample-level DP-SGD at a fixed (eps, delta);
#            secure aggregation modelled as the server seeing only the average update of all clients.
# # Pilot round 5: stronger membership attack (per-update online LiRA, Carlini et al. S&P 2022)
# Server-side attacker that knows the training procedure and the candidate pool (members + paired non-members)
# trains N_SH shadow updates per observed update, each on a random half of the pool, and scores each candidate by a
# likelihood ratio of its logit-scaled confidence under the observed local model. Multi-update attack = sum of LRs.
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

DATASET = os.environ.get('DATASET', 'wisdm')         # 'motionsense' or 'wisdm'
WISDM_FILE = os.environ.get('WISDM_FILE', '../data/WISDM_ar_v1.1_raw.txt')
MIN_STREAM = 400            # WISDM: keep users with >= 400 stream windows
QUICK = False            # True = tiny grid, 1 seed (~3 min) to check that everything runs
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EXP = os.environ.get('EXP', 'main')
OUT = f'results_v7_{EXP}_{DATASET}'
torch.set_num_threads(int(os.environ.get('THREADS', '2'))); os.makedirs(OUT, exist_ok=True)

CFG = dict(
    WIN=64 if DATASET == 'motionsense' else 40, TEST_FRAC=0.2,
    B_GRID=[4, 8, 16, 32] if not QUICK else [4, 16],
    K_GRID=[1, 4],
    BASE_C=4,                                 # baselines (reservoir, keep-all) receive C = 4 new samples per round
    BASE_B=[4, 8, 16, 32] if not QUICK else [16],
    ORDERS=['ordered', 'shuffled'],
    N_SH=16,
    LOCAL_STEPS=3, LR=0.3, CLIP=1.0,
    STREAM_BUDGET=None if not QUICK else 160,
    TARGETS=[1, 2, 3],                        # clients whose updates the server-side attacker analyses
    R_EQ=19 if not QUICK else 5,              # equal-rounds comparison point (fewest rounds in the grid)
    INV_ROUNDS=2, INV_ITERS=400 if not QUICK else 100, INV_B_MAX=0,   # inversion off in this round
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
        if order.startswith('block'):                      # shuffle blocks of m consecutive windows
            m = int(order[5:]); blocks = [np.arange(i, min(i + m, n)) for i in range(0, n, m)]
            perm = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        else:
            perm = (np.arange(n) if order == 'ordered' else np.roll(np.arange(n), rng.integers(n)) if order == 'offset' else rng.permutation(n))
        for k in ('Xs', 'ys', 'rs'): CLIENTS[s][k] = ORIG[s][k][perm]
    NM_MAP.clear()
def run_length():
    out = []
    for s_ in SUBS:
        y = CLIENTS[s_]['ys'][:N_STREAM].numpy(); out.append(len(y) / (1 + np.count_nonzero(y[1:] != y[:-1])))
    return float(np.mean(out))
NM_MAP = {}
DP_SIGMA = 0.0; DP_CLIP = 1.0
def rho_from_eps(eps, delta=1e-5):
    lo, hi = 1e-8, eps
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid + 2 * math.sqrt(mid * math.log(1 / delta)) > eps: hi = mid
        else: lo = mid
    return lo
def dp_sigma(eps, B, K):     # Gaussian noise std on the averaged per-sample-clipped gradient, per local step
    rho = rho_from_eps(eps); return DP_CLIP * math.sqrt(2 * K * CFG['LOCAL_STEPS'] / rho) / B
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
            if DP_SIGMA > 0:                                   # sample-level DP-SGD: per-sample clip + Gaussian noise
                p = {k: v.detach().clone() for k, v in m.named_parameters()}
                for _ in range(CFG['LOCAL_STEPS']):
                    G = per_sample_grads(m, p, X, Y)
                    G = G * torch.clamp(DP_CLIP / (G.norm(dim=1, keepdim=True) + 1e-6), max=1.0)
                    gb = G.mean(0) + DP_SIGMA * torch.randn(G.shape[1], device=DEV); i0 = 0
                    for k in p: n_ = p[k].numel(); p[k] = p[k] - CFG['LR'] * gb[i0:i0 + n_].view_as(p[k]); i0 += n_
                with torch.no_grad():
                    for k, v in m.named_parameters(): v.copy_(p[k])
            else:
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
            agg = flat([w0[k] - p.detach() for k, p in g.named_parameters()]).cpu()   # what the server sees under SecAgg
            for t_ in logs:
                if logs[t_] and logs[t_][-1].get('round') is None: logs[t_][-1]['round'] = rounds; logs[t_][-1]['agg'] = agg; logs[t_][-1]['n_train'] = n_train
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

def mia(logs, seed, key='delta'):
    model = Net().to(DEV)
    s_single, y_single, per_m, per_n = [], [], {}, {}
    for t, L in logs.items():
        c = CLIENTS[t]; nm = nm_map(t, seed)
        for e in L:
            params = {k: v.to(DEV) for k, v in e['w0'].items()}
            mem = np.array(e['members']); non = nm[mem]
            X = torch.cat([c['Xs'][mem], c['Xt'][non]]).to(DEV); Y = torch.cat([c['ys'][mem], c['yt'][non]]).to(DEV)
            sc = F.cosine_similarity(per_sample_grads(model, params, X, Y), e[key].to(DEV)[None], dim=1).cpu().numpy()
            sm, sn = sc[:len(mem)], sc[len(mem):]
            s_single += list(sm) + list(sn); y_single += [1] * len(sm) + [0] * len(sn)
            for i, a, b in zip(mem, sm, sn):
                per_m.setdefault((t, i), []).append(a); per_n.setdefault((t, i), []).append(b)
    # aggregated: average each member (and its paired non-member) over every update the member entered
    agg = [np.mean(v) for v in per_m.values()] + [np.mean(v) for v in per_n.values()]
    y_agg = [1] * len(per_m) + [0] * len(per_n)
    return roc_auc_score(y_single, s_single), roc_auc_score(y_agg, agg)


# %% Attack 3: per-update online LiRA (shadow updates on random halves of the candidate pool)
from sklearn.metrics import roc_curve
def _tpr_at(y, s, fpr_t):
    f, t, _ = roc_curve(y, s); return float(np.interp(fpr_t, f, t))

def make_train(model):
    def loss_fn(p, X, Y): return F.cross_entropy(functional_call(model, p, (X,)), Y)
    def loss1(p, x, y): return F.cross_entropy(functional_call(model, p, (x[None],)), y[None])
    def train_dp(p, X, Y):                    # identical to the DP client
        for _ in range(CFG['LOCAL_STEPS']):
            gs = vmap(grad(loss1), in_dims=(None, 0, 0))(p, X, Y)
            nrm = torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in gs.values()))
            coef = torch.clamp(DP_CLIP / (nrm + 1e-6), max=1.0)
            p = {k: p[k] - CFG['LR'] * ((gs[k] * coef.view(-1, *[1] * (gs[k].dim() - 1))).mean(0) + DP_SIGMA * torch.randn_like(p[k])) for k in p}
        return p
    def train(p, X, Y):                       # identical to the client: SGD + total-norm clipping
        if DP_SIGMA > 0: return train_dp(p, X, Y)
        for _ in range(CFG['LOCAL_STEPS']):
            g = grad(loss_fn)(p, X, Y)
            norm = torch.sqrt(sum((v ** 2).sum() for v in g.values()))
            coef = torch.clamp(CFG['CLIP'] / (norm + 1e-6), max=1.0)
            p = {k: p[k] - CFG['LR'] * coef * g[k] for k in p}
        return p
    return train

def make_phi(model):
    def phi(p, X, Y):
        lp = F.log_softmax(functional_call(model, p, (X,)), 1)
        ly = lp.gather(1, Y[:, None])[:, 0]
        return ly - torch.log1p(-torch.exp(ly).clamp(max=1 - 1e-6))    # log(p/(1-p))
    return phi

def lira(logs, seed):
    model = Net().to(DEV); train = make_train(model); phi = make_phi(model)
    rng = np.random.default_rng(2000 + seed); N = CFG['N_SH']
    y1, s1, per_m, per_n = [], [], {}, {}
    ys, ss, pm_s, pn_s = [], [], {}, {}
    def score(obs, ph, inc):
        mu_in = np.array([ph[inc[:, j], j].mean() for j in range(ph.shape[1])]); mu_out = np.array([ph[~inc[:, j], j].mean() for j in range(ph.shape[1])])
        sd_in = max(np.std(ph[inc] - mu_in[np.where(inc)[1]]), 1e-3); sd_out = max(np.std(ph[~inc] - mu_out[np.where(~inc)[1]]), 1e-3)
        return (-(obs - mu_in) ** 2 / (2 * sd_in ** 2) - np.log(sd_in)) - (-(obs - mu_out) ** 2 / (2 * sd_out ** 2) - np.log(sd_out))
    for t, L in logs.items():
        c = CLIENTS[t]; nm = nm_map(t, seed)
        for e in L:
            w0 = {k: v.to(DEV) for k, v in e['w0'].items()}
            mem = np.array(e['members']); B = len(mem); non = nm[mem]
            Xc = torch.cat([c['Xs'][mem], c['Xt'][non]]).to(DEV); Yc = torch.cat([c['ys'][mem], c['yt'][non]]).to(DEV)
            inc = np.zeros((N, 2 * B), bool)
            for r in range(N // 2):                  # complementary pairs → every candidate is IN in exactly N/2 shadows
                inc[r, rng.choice(2 * B, B, replace=False)] = True; inc[r + N // 2] = ~inc[r]
            sel = torch.tensor(np.stack([np.where(r)[0] for r in inc]), device=DEV)
            P = vmap(train, in_dims=(None, 0, 0), randomness='different')(w0, Xc[sel], Yc[sel])          # N shadow local models
            ph = vmap(phi, in_dims=(0, None, None))(P, Xc, Yc).detach().cpu().numpy()   # (N, 2B)
            d = e['delta'].to(DEV); i0, p_real = 0, {}
            for k, v in w0.items(): n = v.numel(); p_real[k] = v - d[i0:i0 + n].view_as(v); i0 += n
            obs = phi(p_real, Xc, Yc).detach().cpu().numpy()
            mu_in = np.array([ph[inc[:, j], j].mean() for j in range(2 * B)]); mu_out = np.array([ph[~inc[:, j], j].mean() for j in range(2 * B)])
            sd_in = max(np.std(ph[inc] - mu_in[np.where(inc)[1]]), 1e-3); sd_out = max(np.std(ph[~inc] - mu_out[np.where(~inc)[1]]), 1e-3)
            lr = (-(obs - mu_in) ** 2 / (2 * sd_in ** 2) - np.log(sd_in)) - (-(obs - mu_out) ** 2 / (2 * sd_out ** 2) - np.log(sd_out))
            y1 += [1] * B + [0] * B; s1 += list(lr)
            for j, i in enumerate(mem):
                per_m.setdefault((t, i), []).append(lr[j]); per_n.setdefault((t, i), []).append(lr[B + j])
            if 'agg' in e:     # SecAgg: shadow global = w0 - agg - (D_r - mean D)/n ; observed global = w0 - agg
                a = e['agg'].to(DEV); n_c = e['n_train']; i0, ga = 0, {}
                for k, v in w0.items(): n = v.numel(); ga[k] = a[i0:i0 + n].view_as(v); i0 += n
                Dm = {k: (w0[k] - P[k]) for k in w0}
                Gs = {k: w0[k] - ga[k] - (Dm[k] - Dm[k].mean(0)) / n_c for k in w0}
                phs = vmap(phi, in_dims=(0, None, None))(Gs, Xc, Yc).detach().cpu().numpy()
                obs_s = phi({k: w0[k] - ga[k] for k in w0}, Xc, Yc).detach().cpu().numpy()
                lrs = score(obs_s, phs, inc); ys += [1] * B + [0] * B; ss += list(lrs)
                for j, i in enumerate(mem):
                    pm_s.setdefault((t, i), []).append(lrs[j]); pn_s.setdefault((t, i), []).append(lrs[B + j])
    yS = [1] * len(pm_s) + [0] * len(pn_s); sS = [np.sum(v) for v in pm_s.values()] + [np.sum(v) for v in pn_s.values()]
    sec = dict(secagg_lira_auc=roc_auc_score(ys, ss), secagg_lira_agg_auc=roc_auc_score(yS, sS),
               secagg_lira_agg_tpr5=_tpr_at(yS, sS, .05)) if ys else {}
    yA = [1] * len(per_m) + [0] * len(per_n); sA = [np.sum(v) for v in per_m.values()] + [np.sum(v) for v in per_n.values()]
    return dict(lira_auc=roc_auc_score(y1, s1), lira_tpr5=_tpr_at(y1, s1, .05), lira_tpr1=_tpr_at(y1, s1, .01),
                lira_agg_auc=roc_auc_score(yA, sA), lira_agg_tpr5=_tpr_at(yA, sA, .05), lira_agg_tpr1=_tpr_at(yA, sA, .01), **sec)

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

# %% Run the experiment grid (saved after every run; resumes)
CSV = f'{OUT}/fl_grid.csv'
done = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
CFG['SEEDS'] = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]
if EXP == 'main':
    ORDERS = ['ordered', 'shuffled']
    JOBS = [('fifo', B, B // K, K, 0.0) for B in (4, 8, 16, 32) for K in (1, 4)] + [('reservoir', B, 4, np.nan, 0.0) for B in (4, 8, 16, 32)]
else:
    ORDERS = ['ordered']
    JOBS = [('fifo', B, B // K, K, e) for e in [float(x) for x in os.environ.get('EPS', '0,1,8').split(',')] for B in (4, 8, 16, 32) for K in (1, 4)]
for order in ORDERS:
  for seed in CFG['SEEDS']:
    set_order(order, seed); RL = run_length()
    for policy, B, C, K, eps in JOBS:
        if len(done) and ((done.order == order) & (done.seed == seed) & (done.policy == policy) & (done.B == B) & (done.C == C) & (done.eps == eps)).any(): continue
        t0 = time.time(); DP_SIGMA = dp_sigma(eps, B, K) if eps > 0 else 0.0
        g, logs, R, curve, ex = run_fl(policy, B, C, seed)
        acc, f1 = evaluate(g)
        auc1, aucA = mia(logs, seed); sa1, saA = mia(logs, seed, key='agg')
        LR = lira(logs, seed)
        row = dict(order=order, run_len=RL, seed=seed, policy=policy, B=B, C=C, K=K, eps=eps, sigma=DP_SIGMA, rounds=R,
                   acc=acc, macro_f1=f1, mia_single=auc1, mia_agg=aucA, secagg_mia_single=sa1, secagg_mia_agg=saA, **LR, **ex)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(CSV, index=False)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, f'{time.time() - t0:.0f}s', flush=True)
df = done.copy()
S = df.groupby(['order', 'policy', 'eps', 'B', 'C']).mean(numeric_only=True)[['run_len', 'sigma', 'rounds', 'macro_f1', 'mia_single', 'secagg_mia_single', 'lira_auc', 'lira_agg_auc', 'lira_agg_tpr5', 'secagg_lira_auc', 'secagg_lira_agg_auc', 'age_mean']].round(3)
S.to_csv(f'{OUT}/summary.csv'); pd.set_option('display.width', 250); print(S.to_string())
