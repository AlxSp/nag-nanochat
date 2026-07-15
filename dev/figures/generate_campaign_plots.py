
import os, json, math, csv, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO / 'checkpoints'
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault('NANOCHAT_BASE_DIR', str(REPO / 'runs'))

plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 220,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.7,
})
COL = {'gpt':'#2f6f9f', 'old':'#a33d2e', 'nag':'#26734d', 'muted':'#737373', 'gold':'#b98516'}

def savefig(name):
    for ext in ['png', 'svg', 'pdf']:
        plt.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    plt.close()

# Data from completed W&B runs / summaries.
gpt = {
    250:1.101846, 500:0.971190, 750:0.923023, 1000:0.898349, 1250:0.883893, 1500:0.873386,
    2000:0.858655, 2500:0.849491, 3000:0.841648, 3500:0.833550, 4000:0.822263,
    4500:0.812509, 5000:0.804135, 5500:0.796149, 6000:0.788669, 6500:0.781476,
    7000:0.774802, 7500:0.768644, 7750:0.765596, 8000:0.762898, 8250:0.760201,
    8500:0.757825, 8750:0.755578, 9000:0.753464, 9250:0.751651, 9474:0.7502894891726846,
}
old_nag = {
    250:2.090872, 500:1.274642, 750:1.061777, 1000:1.001226, 1250:0.969619, 1500:0.947619,
    2000:0.919637, 2500:0.901402, 3000:0.887930, 3500:0.875177, 4000:0.862217,
    4500:0.851229, 5000:0.841569, 5500:0.832418, 7750:0.800917, 8000:0.798007,
    8250:0.795332, 8500:0.792838, 8750:0.790556, 9000:0.788504, 9250:0.786788, 9425:0.7857823233202547,
}
gatefix = {
    250:1.156194, 500:1.000615, 750:0.947493, 1000:0.920238, 1250:0.902128, 1500:0.889825,
    2000:0.872485, 2500:0.860833, 3000:0.851592, 3500:0.841874, 4000:0.830199,
    4500:0.820530, 5000:0.811856, 5500:0.803889, 6000:0.796283, 6500:0.789218,
    7000:0.782437, 7500:0.776273, 7750:0.773308, 8000:0.770662, 8250:0.768075,
    8500:0.765730, 8750:0.763560, 9000:0.761588, 9250:0.759901, 9425:0.7589225415915322,
}

# R1 headline.
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={'width_ratios':[1.65, 1.0]})
for name, data, c, lw in [('GPT', gpt, COL['gpt'], 2.2), ('old NAG', old_nag, COL['old'], 2.0), ('gatefix NAG', gatefix, COL['nag'], 2.4)]:
    xs = np.array([x for x in sorted(data) if x >= 500]); ys = np.array([data[x] for x in xs])
    ax.plot(xs/1000, ys, marker='o', ms=3.2, lw=lw, color=c, label=name)
ax.set_title('d64/w640 at 3e19 FLOPs')
ax.set_xlabel('training step (k)')
ax.set_ylabel('validation bits per byte')
ax.set_ylim(0.72, 1.32)
ax.legend(frameon=False)
ax.annotate('final 0.7503', xy=(9.474, gpt[9474]), xytext=(7.4, .742), color=COL['gpt'], arrowprops=dict(arrowstyle='-', color=COL['gpt']))
ax.annotate('final 0.7589', xy=(9.425, gatefix[9425]), xytext=(6.9, .765), color=COL['nag'], arrowprops=dict(arrowstyle='-', color=COL['nag']))
ax.annotate('old NAG 0.7858', xy=(9.425, old_nag[9425]), xytext=(6.7, .802), color=COL['old'], arrowprops=dict(arrowstyle='-', color=COL['old']))
common = sorted(set(gpt).intersection(gatefix))
gaps = np.array([gatefix[s]-gpt[s] for s in common])
ax2.plot(np.array(common)/1000, gaps, marker='o', color=COL['gold'], lw=2.2, ms=3.2)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_title('NAG entry fee vs GPT')
ax2.set_xlabel('training step (k)')
ax2.set_ylabel('gatefix NAG - GPT bpb')
for s in [500, 4000, 9250]:
    if s in common:
        ax2.annotate(f'+{gatefix[s]-gpt[s]:.3f}', xy=(s/1000, gatefix[s]-gpt[s]), xytext=(s/1000+.15, gatefix[s]-gpt[s]+.003), fontsize=9)
fig.suptitle('Gate fixes cut the d64 NAG validation gap by 76%', y=1.03, fontsize=14)
savefig('R1_headline_val_bpb')

# R2 gate health quartiles.
old_q = np.array([0.135, 0.062, 0.084, 0.210])
fix_q = np.array([0.1505518764, 0.1541485828, 0.1382104750, 0.1342359456])
labels = ['early\nQ0', 'mid\nQ1', 'mid\nQ2', 'late\nQ3']
x = np.arange(4); w = .36
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.bar(x-w/2, old_q, width=w, color=COL['old'], label='old NAG')
ax.bar(x+w/2, fix_q, width=w, color=COL['nag'], label='gatefix NAG')
ax.set_xticks(x, labels)
ax.set_ylabel('mean realized branch gain')
ax.set_title('Gate health before/after')
ax.legend(frameon=False)
ax.text(0.02, 0.96, 'old: 31/128 quasi-dead branches\ngatefix: 0 dead, floor_frac=0', transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=.35', fc='white', ec='#cccccc', alpha=.9))
ax.set_ylim(0, 0.25)
savefig('R2_gate_health_quartiles')

# R3 gap vs budget.
tokens = np.array([763e6, 2.29e9, 9.8828288e9])
gap = np.array([0.0384, 0.0160, 0.7589225415915322 - 0.7502894891726846])
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(tokens/1e9, gap, marker='o', lw=2.3, color=COL['gold'])
ax.set_xscale('log')
ax.set_xlabel('training tokens (B, log scale)')
ax.set_ylabel('NAG - GPT validation bpb')
ax.set_title('The NAG entry fee amortizes with budget')
ax.fill_between([8.5, 11.5], [0.002, 0.002], [0.010, 0.010], color=COL['nag'], alpha=.14, label='pre-registered d64 band')
for x0, y0, txt in zip(tokens/1e9, gap, ['d26\n1e18', 'd26\n3e18', 'd64\n3e19']):
    ax.annotate(f'{txt}\n+{y0:.4f}', (x0, y0), xytext=(6, 8), textcoords='offset points', fontsize=9)
ax.axhline(0, color='black', lw=.8)
ax.legend(frameon=False)
ax.set_ylim(0, 0.045)
savefig('R3_gap_vs_budget')

# R4 ablation ladder.
abl = [('m ≡ 1', 0.987), ('fixed-gate NAG', 0.968), ('temp-hot', 0.954), ('combo', 0.951), ('α-init ×0.25', 0.949), ('α warmup + gate LR', 0.938)]
gpt_ref = 0.899
fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(abl)); vals = np.array([v for _, v in abl])
ax.barh(y, vals, color=[COL['old']]*2 + [COL['gold']]*3 + [COL['nag']])
ax.axvline(gpt_ref, color=COL['gpt'], lw=2, ls='--', label='GPT reference 0.899')
ax.set_yticks(y, [n for n,_ in abl]); ax.invert_yaxis()
ax.set_xlabel('d26/w640 validation bpb at 1e18 FLOPs')
ax.set_title('Ablation ladder: fixing the training dynamics')
for yi, v in zip(y, vals): ax.text(v + 0.002, yi, f'{v:.3f}', va='center')
ax.set_xlim(0.89, 1.00); ax.legend(frameon=False, loc='lower right')
savefig('R4_ablation_ladder')

# R5 bpb vs CORE split.
fig, ax = plt.subplots(figsize=(6.6, 3.6))
metrics = ['validation bpb\n(lower better)', 'CORE\n(higher better)']
gpt_vals = [0.7502894891726846, 0.2225225749674352]
nag_vals = [0.7589225415915322, 0.2372803890739298]
x = np.arange(2); w=.34
ax.bar(x-w/2, gpt_vals, w, color=COL['gpt'], label='GPT')
ax.bar(x+w/2, nag_vals, w, color=COL['nag'], label='gatefix NAG')
ax.set_xticks(x, metrics); ax.set_title('Validation loss and CORE disagree slightly')
for xi, gv, nv in zip(x, gpt_vals, nag_vals):
    ax.text(xi-w/2, gv+.006, f'{gv:.3f}', ha='center', fontsize=9)
    ax.text(xi+w/2, nv+.006, f'{nv:.3f}', ha='center', fontsize=9)
ax.text(.5, .88, 'NAG: +0.0086 bpb, +0.0148 CORE', transform=ax.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=.35', fc='white', ec='#cccccc'))
ax.legend(frameon=False)
savefig('R5_bpb_core_split')

# P1 orthogonalization diagram.
fig, ax = plt.subplots(figsize=(5.2, 4.2)); ax.set_aspect('equal')
ax.axhline(0, color='#dddddd'); ax.axvline(0, color='#dddddd')
R = np.array([3.2, 0.0]); raw = np.array([2.25, 1.65]); proj = np.array([raw[0], 0.0]); orth = raw - proj
arrow_kw = dict(arrowstyle='-|>', mutation_scale=14, lw=2)
ax.add_patch(FancyArrowPatch((0,0), R, color=COL['gpt'], **arrow_kw))
ax.add_patch(FancyArrowPatch((0,0), raw, color=COL['old'], **arrow_kw))
ax.add_patch(FancyArrowPatch((0,0), proj, color=COL['muted'], arrowstyle='-|>', mutation_scale=12, lw=1.8, linestyle='--'))
ax.add_patch(FancyArrowPatch(tuple(proj), tuple(raw), color=COL['nag'], **arrow_kw))
ax.plot([raw[0], raw[0]], [0, raw[1]], color='#aaaaaa', lw=1, ls=':')
ax.text(R[0]+.08, .02, r'residual direction $\bar R$', color=COL['gpt'])
ax.text(raw[0]+.1, raw[1], 'raw branch output', color=COL['old'])
ax.text(proj[0]/2, -.25, r'projection onto $\bar R$', ha='center', color=COL['muted'])
ax.text(proj[0]+.12, orth[1]/2, 'orthogonal\nremainder', color=COL['nag'], va='center')
ax.add_patch(Arc((proj[0],0), .5, .5, theta1=90, theta2=180, color='#777777'))
ax.set_xlim(-.3, 3.8); ax.set_ylim(-.6, 2.25); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('NAG writes only the orthogonal component')
savefig('P1_orthogonalization_diagram')

# P3 beta curvature.
s = np.linspace(1e-4, 1, 500)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for beta, c in [(0.3, COL['old']), (1.0, COL['gpt']), (3.0, COL['nag'])]:
    ax.plot(s, s**beta, lw=2.2, color=c, label=fr'$\beta={beta:g}$')
ax.set_xlabel(r'base gate $s=\sum_i p_i\sigma_i$')
ax.set_ylabel(r'modulator $m=s^\beta$')
ax.set_title('β controls gate curvature')
ax.legend(frameon=False); ax.set_xlim(0,1); ax.set_ylim(0,1.03)
savefig('P3_beta_curvature')

# Model probes for P2/P4.
def load_state(path):
    data = torch.load(path, map_location='cpu')
    return {k.removeprefix('_orig_mod.'): v for k,v in data.items()}

def get_real_tokens(max_tokens=512):
    import pyarrow.parquet as pq
    from nanochat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    val_path = sorted((Path(os.environ.get('NANOCHAT_BASE_DIR', Path.home() / '.cache/nanochat')) / 'base_data_climbmix').glob('shard_*.parquet'))[-1]
    pf = pq.ParquetFile(val_path)
    text = ''
    ids = []
    for rg in range(min(8, pf.num_row_groups)):
        rows = pf.read_row_group(rg).column('text').to_pylist()
        for t in rows:
            if len(t) > len(text): text = t
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        if len(ids) >= max_tokens: break
    return torch.tensor([ids[:max_tokens]], dtype=torch.long), tok, text

@torch.no_grad()
def probe_gpt():
    from nanochat.gpt import GPT, GPTConfig, norm
    meta = json.loads((CHECKPOINTS/'gpt_d64_w640_3e19/meta_009474.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.device('meta'):
        model = GPT(GPTConfig(**meta['model_config']))
    model.to_empty(device=device); model.init_weights()
    state = load_state(CHECKPOINTS/'gpt_d64_w640_3e19/model_009474.pt')
    model.load_state_dict({k:v.to(device) if torch.is_tensor(v) else v for k,v in state.items()}, strict=True, assign=True)
    model.eval()
    idx, _, _ = get_real_tokens(512); idx = idx.to(device)
    B,T = idx.shape; cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx).to(next(model.parameters()).dtype); x = norm(x)
    residual_rms = [float(x.float().pow(2).mean(-1).sqrt().mean().cpu())]; rel_update = []
    for i, block in enumerate(model.transformer.h):
        x0 = x
        a = block.attn(norm(x), cos_sin, model.window_sizes[i], None); x = x + a
        m = block.mlp(norm(x)); x = x + m
        delta = x - x0
        rel_update.append(float((delta.float().pow(2).mean(-1).sqrt() / x0.float().pow(2).mean(-1).sqrt().clamp_min(1e-12)).mean().cpu()))
        residual_rms.append(float(x.float().pow(2).mean(-1).sqrt().mean().cpu()))
    return np.array(residual_rms), np.array(rel_update)

@torch.no_grad()
def probe_nag_token_conf():
    from nanochat.nag_gpt import GPT, GPTConfig, COMPUTE_DTYPE
    meta = json.loads((CHECKPOINTS/'nag_gpt_d64_w640_3e19_gatefix/meta_009425.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.device('meta'):
        model = GPT(GPTConfig(**meta['model_config']))
    model.to_empty(device=device); model.init_weights()
    state = load_state(CHECKPOINTS/'nag_gpt_d64_w640_3e19_gatefix/model_009425.pt')
    model.load_state_dict({k:v.to(device) if torch.is_tensor(v) else v for k,v in state.items()}, strict=True, assign=True)
    model.eval()
    idx, tok, text = get_real_tokens(192); idx = idx.to(device)
    B,T = idx.shape; cos_sin = model.cos[:, :T], model.sin[:, :T]
    emb = model.transformer.wte(idx)
    rho = emb.float().norm(dim=-1, keepdim=True) / (model.config.n_embd ** 0.5)
    res_log_norm = rho.log() + model.g_log_encode
    res_dir = emb.to(COMPUTE_DTYPE) / rho.to(COMPUTE_DTYPE)
    for i, block in enumerate(model.transformer.h):
        res_log_norm, res_dir = block(res_log_norm, res_dir, None, cos_sin, model.window_sizes[i], None)
    logits = model.lm_head(res_dir) / model.config.n_embd
    logits = logits[..., :model.config.vocab_size].float() * res_log_norm.exp()
    losses = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    scale = res_log_norm.exp()[:, :-1, 0]
    return scale.squeeze(0).float().cpu().numpy(), losses.squeeze(0).float().cpu().numpy()

try:
    residual_rms, rel_update = probe_gpt()
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    layers = np.arange(len(rel_update))
    ax1.plot(layers+1, rel_update, color=COL['gpt'], lw=2.0, label=r'relative block update $\|\Delta x\|/\|x\|$')
    ax1.set_xlabel('layer'); ax1.set_ylabel('relative block update', color=COL['gpt']); ax1.tick_params(axis='y', labelcolor=COL['gpt'])
    ax2 = ax1.twinx(); ax2.plot(np.arange(len(residual_rms)), residual_rms / residual_rms[0], color=COL['old'], lw=2.0, label='residual norm growth')
    ax2.set_yscale('log'); ax2.set_ylabel('residual RMS growth (log)', color=COL['old']); ax2.tick_params(axis='y', labelcolor=COL['old'])
    ax1.set_title('Baseline GPT: useful updates despite residual norm growth')
    ax1.text(.03, .95, f'final residual RMS: {residual_rms[-1]/residual_rms[0]:.0f}× init', transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#cccccc'))
    savefig('P2_gpt_layer_contributions')
except Exception as e:
    print('P2 probe failed:', repr(e))

try:
    scale, losses = probe_nag_token_conf()
    pos = np.arange(1, len(losses)+1)
    fig, ax1 = plt.subplots(figsize=(8.0, 4.0))
    ax1.plot(pos, losses, color=COL['old'], lw=1.4, alpha=.9, label='next-token loss')
    ax1.set_xlabel('token position in real validation sequence'); ax1.set_ylabel('next-token loss', color=COL['old']); ax1.tick_params(axis='y', labelcolor=COL['old'])
    ax2 = ax1.twinx(); ax2.plot(pos, scale, color=COL['nag'], lw=1.6, alpha=.9, label=r'NAG scale $\exp(s_L)$')
    ax2.set_ylabel(r'final norm scale $\exp(s_L)$', color=COL['nag']); ax2.tick_params(axis='y', labelcolor=COL['nag'])
    ax1.set_title('Gatefix NAG: final norm scale over a real sequence')
    corr = float(np.corrcoef(np.log(scale + 1e-9), losses)[0,1])
    ax1.text(.02, .95, f'corr(log scale, loss) = {corr:+.2f}', transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#cccccc'))
    savefig('P4_token_confidence_sequence')
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.scatter(np.log(scale+1e-9), losses, s=12, alpha=.5, color=COL['nag'])
    ax.set_xlabel(r'log final scale $s_L$'); ax.set_ylabel('next-token loss'); ax.set_title('Token scale vs loss')
    ax.text(.04, .95, f'r = {corr:+.2f}', transform=ax.transAxes, va='top')
    savefig('P4_token_confidence_scatter')
except Exception as e:
    print('P4 probe failed:', repr(e))

print('WROTE', OUT)
for p in sorted(OUT.glob('*')): print(p.name)
