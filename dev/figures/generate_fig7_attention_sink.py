
import os, sys, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO / 'checkpoints'
OUT = Path(__file__).resolve().parent / 'output/fig7_attention_sink'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault('NANOCHAT_BASE_DIR', str(REPO / 'runs'))

from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT as BaseGPT, GPTConfig as BaseConfig
from nanochat.nag_gpt import GPT as NAGGPT, GPTConfig as NAGConfig
from nanochat.gpt import apply_rotary_emb as base_rotary
from nanochat.nag_gpt import apply_rotary_emb as nag_rotary

DEVICE = torch.device('cpu')
T = 64
N_SEQS = 128

plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 220,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

def load_state(path):
    data = torch.load(path, map_location='cpu')
    out = {}
    for k, v in data.items():
        k = k.removeprefix('_orig_mod.')
        if torch.is_tensor(v) and v.is_floating_point():
            v = v.float()
        out[k] = v
    return out

def load_base():
    ckpt = CHECKPOINTS/'gpt_d64_w640_3e19'
    meta = json.loads((ckpt/'meta_009474.json').read_text())
    with torch.device('meta'):
        m = BaseGPT(BaseConfig(**meta['model_config']))
    m.to_empty(device=DEVICE); m.init_weights()
    m.load_state_dict(load_state(ckpt/'model_009474.pt'), strict=True, assign=True)
    m.eval()
    return m

def load_nag():
    ckpt = CHECKPOINTS/'nag_gpt_d64_w640_3e19_gatefix'
    meta = json.loads((ckpt/'meta_009425.json').read_text())
    with torch.device('meta'):
        m = NAGGPT(NAGConfig(**meta['model_config']))
    m.to_empty(device=DEVICE); m.init_weights()
    m.load_state_dict(load_state(ckpt/'model_009425.pt'), strict=True, assign=True)
    m.eval()
    return m

def make_eval_batch():
    import pyarrow.parquet as pq
    tok = get_tokenizer()
    val_path = sorted((Path(os.environ.get('NANOCHAT_BASE_DIR', Path.home() / '.cache/nanochat')) / 'base_data_climbmix').glob('shard_*.parquet'))[-1]
    pf = pq.ParquetFile(val_path)
    seqs = []
    bos = tok.get_bos_token_id()
    for rg in range(pf.num_row_groups):
        rows = pf.read_row_group(rg).column('text').to_pylist()
        for text in rows:
            ids = tok.encode(text, prepend=bos)
            if len(ids) >= T:
                seqs.append(ids[:T])
            if len(seqs) >= N_SEQS:
                return torch.tensor(seqs, dtype=torch.long, device=DEVICE)
    if len(seqs) < N_SEQS:
        raise RuntimeError(f'Only found {len(seqs)} sequences')
    return torch.tensor(seqs, dtype=torch.long, device=DEVICE)

def attention_probs_from_qk(q, k):
    # q/k: B,T,H,D, already rotary and qk-normalized.
    B, T_, H, D = q.shape
    scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(D)
    mask = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))
    return torch.softmax(scores.float(), dim=-1)

def base_layer_attn_and_step(block, x, cos_sin):
    x_norm = rms_norm(x)
    B, T_, C = x_norm.shape
    attn = block.attn
    q = attn.c_q(x_norm).view(B, T_, attn.n_head, attn.head_dim)
    k = attn.c_k(x_norm).view(B, T_, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(x_norm).view(B, T_, attn.n_kv_head, attn.head_dim)
    cos, sin = cos_sin
    q, k = base_rotary(q, cos, sin), base_rotary(k, cos, sin)
    q, k = rms_norm(q) * 1.2, rms_norm(k) * 1.2
    probs = attention_probs_from_qk(q, k)
    y = torch.einsum('bhts,bshd->bthd', probs, v).contiguous().view(B, T_, -1)
    y = attn.c_proj(y)
    x = x + y
    x = x + block.mlp(rms_norm(x))
    return probs, x

def nag_layer_attn_and_step(block, res_log_norm, res_dir, cos_sin):
    B, T_, C = res_dir.shape
    attn = block.attn
    q = attn.c_q(res_dir).view(B, T_, attn.n_head, attn.head_dim)
    k = attn.c_k(res_dir).view(B, T_, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(res_dir).view(B, T_, attn.n_kv_head, attn.head_dim)
    cos, sin = cos_sin
    q, k = nag_rotary(q, cos, sin), nag_rotary(k, cos, sin)
    q, k = rms_norm(q) * 1.2, rms_norm(k) * 1.2
    probs = attention_probs_from_qk(q, k)
    y = torch.einsum('bhts,bshd->bthd', probs, v).contiguous().view(B, T_, -1)
    y = attn.c_proj(y)
    res_log_norm, res_dir = block.attn_branch(res_log_norm, res_dir, y)
    res_log_norm, res_dir = block.mlp_branch(res_log_norm, res_dir, block.mlp(res_dir))
    return probs, res_log_norm, res_dir

@torch.no_grad()
def collect_base(model, idx):
    B, T_ = idx.shape
    cos_sin = model.cos[:, :T_].float(), model.sin[:, :T_].float()
    x = rms_norm(model.transformer.wte(idx).float())
    mats = []
    for i, block in enumerate(model.transformer.h):
        probs, x = base_layer_attn_and_step(block, x, cos_sin)
        mats.append(probs.mean(dim=(0,1)).cpu().numpy())
    return np.stack(mats)

@torch.no_grad()
def collect_nag(model, idx):
    B, T_ = idx.shape
    cos_sin = model.cos[:, :T_].float(), model.sin[:, :T_].float()
    emb = model.transformer.wte(idx).float()
    rho = emb.float().norm(dim=-1, keepdim=True) / (model.config.n_embd ** 0.5)
    res_log_norm = rho.log() + model.g_log_encode.float()
    res_dir = emb / rho
    mats = []
    for i, block in enumerate(model.transformer.h):
        probs, res_log_norm, res_dir = nag_layer_attn_and_step(block, res_log_norm, res_dir, cos_sin)
        mats.append(probs.mean(dim=(0,1)).cpu().numpy())
    return np.stack(mats)

def sink_metrics(m):
    avg = m.mean(axis=0) # T,T
    # Ignore row 0, where causal attention has no choice but key 0.
    rows = np.arange(1, avg.shape[0])
    key0_mass = float(avg[rows, 0].mean())
    early4_mass = float(avg[rows, :4].sum(axis=1).mean())
    diagonal_mass = float(np.mean([avg[i, i] for i in rows]))
    return {'key0_mass': key0_mass, 'early4_mass': early4_mass, 'diagonal_mass': diagonal_mass}

def save_heatmap(base_mats, nag_mats):
    base_avg = base_mats.mean(axis=0)
    nag_avg = nag_mats.mean(axis=0)
    vmax = max(base_avg.max(), nag_avg.max())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    for ax, mat, title in [(axes[0], base_avg, 'GPT baseline'), (axes[1], nag_avg, 'gatefix NAG')]:
        im = ax.imshow(mat / vmax, origin='upper', cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('key position')
        ax.set_ylabel('query position')
        ax.set_xlim(-0.5, T-0.5); ax.set_ylim(T-0.5, -0.5)
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label='attention / shared max')
    fig.suptitle('Figure 7 recreation: layer-averaged post-softmax attention')
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'fig7_attention_sink_recreation.{ext}', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    for ax, mat, title in [(axes[0], base_avg, 'GPT baseline'), (axes[1], nag_avg, 'gatefix NAG')]:
        im = ax.imshow(mat / mat.max(), origin='upper', cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('key position')
        ax.set_ylabel('query position')
        ax.set_xlim(-0.5, T-0.5); ax.set_ylim(T-0.5, -0.5)
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label='attention / per-panel max')
    fig.suptitle('Figure 7 recreation: per-heatmap normalized, as in paper')
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'fig7_attention_sink_recreation_per_panel_norm.{ext}', bbox_inches='tight')
    plt.close(fig)

    # Also save a key-position profile to quantify the vertical sink band.
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    rows = np.arange(1, T)
    ax.plot(base_avg[rows].mean(axis=0), color='#2f6f9f', lw=2.0, label='GPT baseline')
    ax.plot(nag_avg[rows].mean(axis=0), color='#26734d', lw=2.0, label='gatefix NAG')
    ax.set_xlabel('key position')
    ax.set_ylabel('mean attention mass\naveraged over query positions 1..63')
    ax.set_title('Attention sink profile')
    ax.legend(frameon=False)
    for ext in ['png', 'svg']:
        fig.savefig(OUT/f'fig7_attention_sink_profile.{ext}', bbox_inches='tight')
    plt.close(fig)

    metrics = {'baseline': sink_metrics(base_mats), 'nag': sink_metrics(nag_mats)}
    (OUT/'fig7_attention_sink_metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(OUT/'fig7_attention_sink_mats.npz', base=base_mats, nag=nag_mats)
    return metrics

def main():
    idx = make_eval_batch()
    print('batch', tuple(idx.shape))
    print('loading baseline')
    base = load_base()
    print('collecting baseline')
    base_mats = collect_base(base, idx)
    del base
    print('loading nag')
    nag = load_nag()
    print('collecting nag')
    nag_mats = collect_nag(nag, idx)
    metrics = save_heatmap(base_mats, nag_mats)
    print(json.dumps(metrics, indent=2))
    print('wrote', OUT)

if __name__ == '__main__':
    main()
