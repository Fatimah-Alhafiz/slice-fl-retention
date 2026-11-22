# %% [markdown]
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

QUICK = False            # True = tiny grid, 1 seed (~3 min) to check that everything runs
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = 'results_v2'; os.makedirs(OUT, exist_ok=True)

CFG = dict(
    WIN=64, TEST_FRAC=0.2,
    B_GRID=[4, 8, 16, 32] if not QUICK else [4, 16],
    K_GRID=[1, 2, 4] if not QUICK else [1, 4],
    BASE_C=4,                                 # baselines (reservoir, keep-all) receive C = 4 new samples per round
    BASE_B=[4, 8, 16, 32] if not QUICK else [16],
    LOCAL_STEPS=3, LR=0.3, CLIP=1.0,
    STREAM_BUDGET=None if not QUICK else 160,
    TARGETS=[1, 2, 3],                        # clients whose updates the server-side attacker analyses
    R_EQ=19 if not QUICK else 5,              # equal-rounds comparison point (fewest rounds in the grid)
    INV_ROUNDS=2, INV_ITERS=400 if not QUICK else 100, INV_B_MAX=16,
    SEEDS=[0] if QUICK else [0, 1, 2, 3, 4],
)
print(DEV, CFG)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# %% Download + load MotionSense
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

def windows(x, w):
    n = len(x) // w
    return x[:n * w].reshape(n, w, -1).transpose(0, 2, 1)

files = [f for f in glob.glob(f'{ROOT}/**/sub_*.csv', recursive=True) if '__MACOSX' not in f]
per = {}
for f in files:
    act, trial = os.path.basename(os.path.dirname(f)).split('_')
    sub = int(os.path.basename(f)[4:-4])
    x = pd.read_csv(f)[COLS].values.astype(np.float32)
    cut = int(len(x) * (1 - CFG['TEST_FRAC']))
    per.setdefault(sub, []).append((int(trial), ACTS.index(act), windows(x[:cut], CFG['WIN']), windows(x[cut:], CFG['WIN'])))

CLIENTS = {}
for sub, recs in per.items():
    recs.sort(key=lambda r: r[0])
    cat = lambda j: np.concatenate([r[j] for r in recs])
    lab = lambda j, v: np.concatenate([[r[v]] * len(r[j]) for r in recs])
    CLIENTS[sub] = dict(Xs=cat(2), ys=lab(2, 1), rs=lab(2, 0), Xt=cat(3), yt=lab(3, 1), rt=lab(3, 0))

allX = np.concatenate([c['Xs'] for c in CLIENTS.values()])
MU, SD = allX.mean((0, 2), keepdims=True)[0], allX.std((0, 2), keepdims=True)[0] + 1e-6
for c in CLIENTS.values():
    for k in ('Xs', 'Xt'): c[k] = torch.tensor((c[k] - MU) / SD)
    for k in ('ys', 'yt'): c[k] = torch.tensor(c[k], dtype=torch.long)
SUBS = sorted(CLIENTS)
N_STREAM = min(len(CLIENTS[s]['ys']) for s in SUBS)
if CFG['STREAM_BUDGET']: N_STREAM = min(N_STREAM, CFG['STREAM_BUDGET'])
XT_ALL = torch.cat([CLIENTS[s]['Xt'] for s in SUBS]).to(DEV); YT_ALL = torch.cat([CLIENTS[s]['yt'] for s in SUBS]).to(DEV)
print(f'{len(SUBS)} clients, stream windows per client: {N_STREAM}, test windows: {len(YT_ALL)}')

# %% Model + helpers
class Net(nn.Module):
    def __init__(s, c=12, k=6):
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
    return g, logs, rounds, acc_curve, dict(held_mean=float(np.mean(held_trace)), held_max=int(np.max(held_trace)),
                                            K_eff=float(u[u > 0].mean()), never_used=float((u == 0).mean()))

# %% Attack 1: paired gradient-alignment membership inference
NM_MAP = {}
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
       [(p, B, CFG['BASE_C'], np.nan) for p in ['reservoir', 'keepall'] for B in CFG['BASE_B']]
for seed in CFG['SEEDS']:
    for policy, B, C, K in JOBS:
        if len(done) and ((done.seed == seed) & (done.policy == policy) & (done.B == B) & (done.C == C)).any(): continue
        t0 = time.time()
        g, logs, R, curve, ex = run_fl(policy, B, C, seed)
        acc, f1 = evaluate(g)
        auc1, aucA = mia(logs, seed)
        inv = base = np.nan
        if B <= CFG['INV_B_MAX']:
            L = logs[CFG['TARGETS'][0]]; picks = np.linspace(len(L) // 2, len(L) - 1, CFG['INV_ROUNDS']).astype(int)
            r = [invert(L[p], CLIENTS[CFG['TARGETS'][0]], seed) for p in picks]
            inv, base = float(np.mean([a for a, _ in r])), float(np.mean([b for _, b in r]))
        row = dict(seed=seed, policy=policy, B=B, C=C, K=K, rounds=R, acc=acc, macro_f1=f1,
                   acc_at_Req=curve[CFG['R_EQ'] - 1] if len(curve) >= CFG['R_EQ'] else np.nan,
                   mia_single=auc1, mia_agg=aucA, inv_corr=inv, inv_base=base, inv_gain=inv - base, **ex)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(CSV, index=False)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, f'{time.time() - t0:.0f}s', flush=True)
df = done.copy()

# %% Summary with bootstrap 95 % CIs over seeds
def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1); return tuple(np.percentile(m, [2.5, 97.5]))

METRICS = ['acc', 'macro_f1', 'acc_at_Req', 'mia_single', 'mia_agg', 'inv_gain', 'held_mean', 'K_eff', 'rounds']
rows = []
for key, d in df.groupby(['policy', 'B', 'C']):
    r = dict(policy=key[0], B=key[1], C=key[2], K=d.K.iloc[0], n_seeds=len(d))
    for mtr in METRICS:
        lo, hi = boot_ci(d[mtr]); r[mtr] = f'{d[mtr].mean():.3f} [{lo:.3f}, {hi:.3f}]'
    rows.append(r)
summary = pd.DataFrame(rows); summary.to_csv(f'{OUT}/summary_ci.csv', index=False)
pd.set_option('display.width', 250); print(summary.to_string(index=False))

# %% RQ2: exposure simulation — amount vs timer, three arrival scenarios, sample age
def rate_fn(scenario, lam, rng):
    state = dict(left=0.0)
    def f(t, dt):
        if scenario == 'steady': return lam
        if scenario == 'variable': return lam * (1 + 0.8 * math.sin(2 * math.pi * t / 500))
        if state['left'] <= 0 and rng.random() < 0.002 * dt: state['left'] = 50.0   # burst: ~9 % of time at 10x
        state['left'] -= dt; return lam * (10 if state['left'] > 0 else 1)
    return f

def exposure(B, K, policy, scenario, lam=1.0, tau=2.0, T_end=10000, seed=0):
    rng = np.random.default_rng(seed); C = B // K; dt = 0.1; T = C / lam; rate = rate_fn(scenario, lam, rng)
    uses, arr = [], []; held, age_mean, age_max = [], [], []
    dropped = arrived = 0; training_until, snap, next_timer = -1, None, T
    for step in range(int(T_end / dt)):
        t = step * dt; n = rng.poisson(rate(t, dt) * dt); arrived += n
        if policy == 'amount':
            keep = min(n, max(B + C - len(uses), 0)); dropped += n - keep
        else: keep = n
        uses += [0] * keep; arr += [t] * keep
        if snap is not None and t >= training_until:
            for i in range(snap): uses[i] += 1
            kept = [(u, a) for u, a in zip(uses, arr) if u < K]
            uses = [u for u, _ in kept]; arr = [a for _, a in kept]; snap = None
        if snap is None:
            if policy == 'amount' and sum(u == 0 for u in uses) >= C: snap = min(len(uses), B); training_until = t + tau
            if policy == 'timer' and t >= next_timer: snap = len(uses); training_until = t + tau; next_timer += T
        held.append(len(uses))
        if arr: age_mean.append(t - sum(arr) / len(arr)); age_max.append(t - arr[0])
    h = np.array(held)
    return dict(B=B, K=K, policy=policy, scenario=scenario, L_mean=h.mean(), L_max=h.max(), L_p99=np.percentile(h, 99),
                age_mean=np.mean(age_mean), age_max=np.max(age_max), drop_rate=dropped / max(arrived, 1),
                L_model=B * (1 - 1 / (2 * K)) + lam * tau)

ex = pd.DataFrame([exposure(B, K, p, sc) for sc in ['steady', 'burst', 'variable'] for B in [8, 16, 32]
                   for K in [1, 2, 4] for p in ['amount', 'timer']])
ex.to_csv(f'{OUT}/exposure.csv', index=False); print(ex.round(2).to_string(index=False))

# %% Plots
import matplotlib.pyplot as plt
fifo = df[df.policy == 'fifo'].groupby(['B', 'K']).mean(numeric_only=True).reset_index()
base = df[df.policy != 'fifo'].groupby(['policy', 'B']).mean(numeric_only=True).reset_index()
panels = [('acc_at_Req', f'Accuracy at {CFG["R_EQ"]} rounds'), ('acc', 'Accuracy (equal data)'),
          ('mia_single', 'MIA AUC – single update'), ('mia_agg', 'MIA AUC – aggregated'), ('inv_gain', 'Inversion gain over random')]
fig, ax = plt.subplots(1, len(panels), figsize=(22, 4))
for a, (col, title) in zip(ax, panels):
    for K in CFG['K_GRID']:
        d = fifo[fifo.K == K]; a.plot(d.B, d[col], 'o-', label=f'FIFO K={K}')
    for p, mk in [('reservoir', 's--'), ('keepall', '^:')]:
        d = base[base.policy == p]; a.plot(d.B, d[col], mk, label=f'{p} (C={CFG["BASE_C"]})')
    a.set_xscale('log', base=2); a.set_xlabel('Buffer size B'); a.set_title(title); a.grid(alpha=.3)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f'{OUT}/rq3_grid.png', dpi=200); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a, sc in zip(ax, ['steady', 'burst', 'variable']):
    for p, mk in [('amount', 'o-'), ('timer', 's--')]:
        d = ex[(ex.policy == p) & (ex.K == 1) & (ex.scenario == sc)]; a.plot(d.B, d.L_max, mk, label=f'{p}: max held')
    a.set_title(f'{sc} arrivals (K=1)'); a.set_xlabel('B'); a.grid(alpha=.3)
ax[0].set_ylabel('raw samples held'); ax[0].legend(); plt.tight_layout(); plt.savefig(f'{OUT}/rq2_exposure.png', dpi=200); plt.show()

# %% Pre-registered checks (PASS / FAIL)
F_ = df[df.policy == 'fifo']; checks = []
def chk(name, ok, detail): checks.append(dict(check=name, result='PASS' if ok else 'FAIL', detail=detail))

r = spearmanr(F_.B, F_.mia_single); chk('RQ3a: single-update MIA falls as B grows', r.correlation < 0 and r.pvalue < .05,
                                        f'Spearman={r.correlation:.2f}, p={r.pvalue:.1e}, n={len(F_)}')
for B in CFG['B_GRID']:
    k1 = F_[(F_.B == B) & (F_.K == 1)].set_index('seed').mia_agg; kx = F_[(F_.B == B) & (F_.K == max(CFG['K_GRID']))].set_index('seed').mia_agg
    d = (kx - k1).dropna(); lo, hi = boot_ci(d)
    chk(f'RQ3b B={B}: aggregated MIA higher at K={max(CFG["K_GRID"])} than K=1', lo > 0, f'diff={d.mean():.3f} [{lo:.3f}, {hi:.3f}]')
k1 = F_[F_.K == 1]; r = spearmanr(k1.B, k1.acc_at_Req)
chk(f'Utility: at equal rounds ({CFG["R_EQ"]}), accuracy rises with B (K=1)', r.correlation > 0 and r.pvalue < .05,
    f'Spearman={r.correlation:.2f}, p={r.pvalue:.1e}')
r = spearmanr(k1.B, k1.acc); chk('Utility: at equal data, accuracy rises with B (K=1)', r.correlation > 0 and r.pvalue < .05,
                                 f'Spearman={r.correlation:.2f}, p={r.pvalue:.1e}  (FAIL expected: fewer rounds)')
g_ = df.inv_gain.dropna(); lo, hi = boot_ci(g_); chk('Inversion beats random guessing', lo > 0, f'gain={g_.mean():.3f} [{lo:.3f}, {hi:.3f}]')
pv = ex.pivot_table(index=['scenario', 'B', 'K'], columns='policy', values='L_max')
for sc in ['burst', 'variable']:
    chk(f'RQ2 ({sc}): timer max exposure > amount max in every cell', bool((pv.loc[sc].timer > pv.loc[sc].amount).all()), '')
st = ex[(ex.scenario == 'steady') & (ex.policy == 'amount')]; err = float((abs(st.L_mean - st.L_model) / st.L_model).mean())
chk('RQ2: steady-state model error < 10 %', err < .10, f'mean relative error = {err:.3f}')
checks = pd.DataFrame(checks); checks.to_csv(f'{OUT}/checks.csv', index=False); print(checks.to_string(index=False))
