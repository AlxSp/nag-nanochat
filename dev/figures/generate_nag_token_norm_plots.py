
import os, sys, json, textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'output/token_norm_examples'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault('NANOCHAT_BASE_DIR', str(REPO / 'runs'))

from nanochat.nag_gpt import GPT, GPTConfig, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer

plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 220,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.7,
})
BLUE = '#2878b5'
ORANGE = '#f2a65a'
GREEN = '#26734d'

PARAGRAPHS = [
    (
        'easy_news',
        'Easy paragraph',
        'The city opened a new library on Monday. Families walked through the bright rooms, borrowed books, and listened as the mayor thanked the volunteers who helped finish the project.'
    ),
    (
        'medium_science',
        'Medium paragraph',
        'When sunlight reaches a leaf, chlorophyll absorbs part of the energy and helps convert carbon dioxide and water into sugars. The process is efficient enough to feed the plant, but it also depends on temperature, moisture, and the availability of minerals in the soil.'
    ),
    (
        'hard_technical',
        'Hard paragraph',
        'In a norm-agnostic residual network, each branch writes only the component orthogonal to the current representation while a separate scalar lane tracks the accumulated log norm. This decouples directional computation from logit temperature, but it makes early training sensitive to how aggressively branch magnitudes are introduced.'
    ),
]

def token_label(tok, token_id):
    s = tok.decode([int(token_id)])
    s = s.replace('\n', '\\n').replace('\t', '\\t')
    if s == ' ':
        s = '␠'
    elif s.startswith(' '):
        s = '▁' + s[1:]
    if len(s) > 12:
        s = s[:11] + '…'
    return s

@torch.no_grad()
def load_model_cpu():
    ckpt = REPO.parent / 'checkpoints/nag_gpt_d64_w640_3e19_gatefix'
    meta = json.loads((ckpt / 'meta_009425.json').read_text())
    state = torch.load(ckpt / 'model_009425.pt', map_location='cpu')
    state = {
        k.removeprefix('_orig_mod.'): (v.float() if torch.is_tensor(v) and v.dtype == torch.bfloat16 else v)
        for k, v in state.items()
    }
    with torch.device('meta'):
        model = GPT(GPTConfig(**meta['model_config']))
    model.to_empty(device='cpu')
    model.init_weights()
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model, meta

@torch.no_grad()
def forward_with_log_scale(model, idx):
    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    emb = model.transformer.wte(idx)
    rho = emb.float().norm(dim=-1, keepdim=True) / (model.config.n_embd ** 0.5)
    res_log_norm = rho.log() + model.g_log_encode
    res_dir = emb.to(COMPUTE_DTYPE) / rho.to(COMPUTE_DTYPE)
    for i, block in enumerate(model.transformer.h):
        res_log_norm, res_dir = block(res_log_norm, res_dir, None, cos_sin, model.window_sizes[i], None)
    logits = model.lm_head(res_dir) / model.config.n_embd
    logits = logits[..., :model.config.vocab_size].float() * res_log_norm.exp()
    nll = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return res_log_norm[:, :-1, 0].float(), nll.float(), logits

def choose_window(labels, nll, max_tokens=34):
    n = len(labels)
    if n <= max_tokens:
        return 0, n
    # Prefer a window with varied losses but not only BPE fragments; skip BOS-adjacent first tokens.
    scores = []
    for start in range(0, n - max_tokens + 1):
        seg = nll[start:start+max_tokens]
        scores.append((float(seg.std()) + 0.08 * float(seg.mean()), start))
    _, start = max(scores)
    return start, start + max_tokens

def plot_one(slug, title, text, model, tok):
    ids = tok.encode(text, prepend=tok.get_bos_token_id())
    idx = torch.tensor([ids], dtype=torch.long)
    log_scale, nll, _ = forward_with_log_scale(model, idx)
    log_scale = log_scale.squeeze(0).cpu().numpy()
    nll = nll.squeeze(0).cpu().numpy()
    target_ids = ids[1:]
    labels = [token_label(tok, t) for t in target_ids]
    start, end = choose_window(labels, nll, max_tokens=34)
    labels = labels[start:end]
    x = np.arange(len(labels))
    ls = log_scale[start:end]
    loss = nll[start:end]

    corr = float(np.corrcoef(ls, loss)[0, 1]) if len(ls) > 2 else float('nan')
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, len(labels) * 0.42), 5.4), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.05}
    )
    fig.suptitle(f'{title}: inverse-temperature signal vs next-token loss', y=0.98)

    ax_top.bar(x, ls, color=BLUE, width=0.82, label=r'NAG log inverse temp $s_L$')
    ax_top.axhline(float(np.mean(ls)), color=BLUE, lw=1.2, alpha=0.45, ls='--')
    ax_top.set_ylabel(r'log inverse temp $s_L$')
    ax_top.legend(loc='upper right', frameon=True)
    ax_top.text(0.01, 0.93, 'higher confidence / sharper logits', transform=ax_top.transAxes, va='top', color=BLUE)

    ax_bot.bar(x, loss, color=ORANGE, width=0.82, label='next-token NLL')
    ax_bot.axhline(float(np.mean(loss)), color=ORANGE, lw=1.2, alpha=0.55, ls='--')
    ax_bot.invert_yaxis()
    ax_bot.set_ylabel('NLL (nats)')
    ax_bot.text(0.01, 0.08, 'harder token / higher loss', transform=ax_bot.transAxes, va='bottom', color='#9a5a14')
    ax_bot.legend(loc='lower right', frameon=True)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=65, ha='right')
    ax_bot.set_xlabel('predicted token')
    fig.text(0.012, 0.5, f'corr(signal, loss) = {corr:+.2f}', va='center', rotation=90, color='#444444')

    # Put compact source text under the plot for provenance.
    wrapped = '\n'.join(textwrap.wrap(text, width=120))
    fig.text(0.5, -0.02, wrapped, ha='center', va='top', fontsize=8, color='#555555')

    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'nag_token_norm_{slug}.{ext}', bbox_inches='tight')
    plt.close(fig)

    # Mean-centered natural-unit version: closer to the original mirrored plot,
    # but still removes paragraph-level offsets.
    c_ls = ls - np.mean(ls)
    c_loss = loss - np.mean(loss)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, len(labels) * 0.42), 5.4), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.05}
    )
    fig.suptitle(f'{title}: above-mean inverse temperature vs above-mean token loss', y=0.98)
    ax_top.bar(x, c_ls, color=BLUE, width=0.82, label=r'$s_L - \bar{s}_L$')
    ax_top.axhline(0, color='#333333', lw=1.0)
    ax_top.set_ylabel(r'log inverse temp above mean')
    ax_top.legend(loc='upper right', frameon=True)
    ax_top.text(0.01, 0.93, 'above: sharper-than-paragraph-average logits', transform=ax_top.transAxes, va='top', color=BLUE)

    ax_bot.bar(x, -c_loss, color=ORANGE, width=0.82, label=r'mirrored $(NLL - \overline{NLL})$')
    ax_bot.axhline(0, color='#333333', lw=1.0)
    ax_bot.set_ylabel('NLL above mean (nats), mirrored')
    ax_bot.text(0.01, 0.08, 'below: harder-than-paragraph-average token', transform=ax_bot.transAxes, va='bottom', color='#9a5a14')
    ax_bot.legend(loc='lower right', frameon=True)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=65, ha='right')
    ax_bot.set_xlabel('predicted token')
    fig.text(0.012, 0.5, f'corr(signal, loss) = {corr:+.2f}', va='center', rotation=90, color='#444444')
    fig.text(0.5, -0.02, wrapped, ha='center', va='top', fontsize=8, color='#555555')
    ax_top.text(0.99, 0.06, f'mean $s_L$={np.mean(ls):.2f}', transform=ax_top.transAxes, ha='right', va='bottom', fontsize=8, color='#444444')
    ax_bot.text(0.99, 0.94, f'mean NLL={np.mean(loss):.2f} nats', transform=ax_bot.transAxes, ha='right', va='top', fontsize=8, color='#444444')
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'nag_token_norm_mean_centered_{slug}.{ext}', bbox_inches='tight')
    plt.close(fig)

    # Single-centerline version: both metrics originate from the same visual baseline.
    # The plotted heights are normalized by each paragraph's own range so the
    # centerline is visually honest. Raw unit ranges are annotated on the plot.
    ls_from_min = ls - np.min(ls)
    loss_from_min = loss - np.min(loss)
    max_ls = max(float(np.max(ls_from_min)), 1e-9)
    max_loss = max(float(np.max(loss_from_min)), 1e-9)
    blue_h = ls_from_min / max_ls
    orange_h = -(loss_from_min / max_loss)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.42), 4.8))
    fig.suptitle(f'{title}: inverse temperature and token loss from one centerline', y=0.98)
    ax.bar(x, blue_h, color=BLUE, width=0.76, label=r'$s_L$ above paragraph min')
    ax.bar(x, orange_h, color=ORANGE, width=0.50, alpha=0.88, label='NLL above paragraph min, mirrored')
    ax.axhline(0, color='#303030', lw=1.15)
    ax.set_ylim(-1.12, 1.12)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(['max harder', 'half', 'center', 'half', 'max sharper'])
    ax.set_ylabel('within-paragraph relative height')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha='right')
    ax.set_xlabel('predicted token')
    ax.grid(True, axis='y', alpha=0.25)
    ax.text(0.01, 0.93, 'blue grows upward: sharper logits', transform=ax.transAxes, va='top', color=BLUE)
    ax.text(0.01, 0.07, 'orange grows downward: higher token loss', transform=ax.transAxes, va='bottom', color='#9a5a14')
    ax.text(
        0.99, 0.07,
        f'raw ranges: $s_L$ +0..{max_ls:.2f}; NLL +0..{max_loss:.2f} nats',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#444444'
    )
    ax.legend(loc='upper right', frameon=True)
    fig.text(0.012, 0.5, f'corr(raw $s_L$, raw NLL) = {corr:+.2f}', va='center', rotation=90, color='#444444')
    fig.text(0.5, -0.05, wrapped, ha='center', va='top', fontsize=8, color='#555555')
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'nag_token_norm_centerline_{slug}.{ext}', bbox_inches='tight')
    plt.close(fig)

    # Centered/z-scored version for visualizing the small token-wise modulation.
    z_ls = (ls - np.mean(ls)) / (np.std(ls) + 1e-12)
    z_loss = (loss - np.mean(loss)) / (np.std(loss) + 1e-12)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, len(labels) * 0.42), 5.4), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.05}
    )
    fig.suptitle(f'{title}: centered inverse-temperature signal vs centered NLL', y=0.98)
    ax_top.bar(x, z_ls, color=BLUE, width=0.82, label=r'centered $s_L$ (z-score)')
    ax_top.axhline(0, color='#333333', lw=1.0)
    ax_top.set_ylabel(r'$s_L$ z-score')
    ax_top.legend(loc='upper right', frameon=True)
    ax_top.text(0.01, 0.93, 'above mean: sharper logits', transform=ax_top.transAxes, va='top', color=BLUE)

    ax_bot.bar(x, z_loss, color=ORANGE, width=0.82, label='centered NLL (z-score)')
    ax_bot.axhline(0, color='#333333', lw=1.0)
    ax_bot.invert_yaxis()
    ax_bot.set_ylabel('NLL z-score')
    ax_bot.text(0.01, 0.08, 'below: harder than paragraph mean', transform=ax_bot.transAxes, va='bottom', color='#9a5a14')
    ax_bot.legend(loc='lower right', frameon=True)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=65, ha='right')
    ax_bot.set_xlabel('predicted token')
    fig.text(0.012, 0.5, f'corr(signal, loss) = {corr:+.2f}', va='center', rotation=90, color='#444444')
    fig.text(0.5, -0.02, wrapped, ha='center', va='top', fontsize=8, color='#555555')
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'nag_token_norm_centered_{slug}.{ext}', bbox_inches='tight')
    plt.close(fig)
    return {
        'slug': slug,
        'tokens_total': len(target_ids),
        'tokens_plotted': len(labels),
        'window': [start, end],
        'corr_log_scale_loss': corr,
        'log_scale_mean': float(np.mean(ls)),
        'log_scale_std': float(np.std(ls)),
        'nll_mean_nats': float(np.mean(loss)),
        'nll_std_nats': float(np.std(loss)),
    }

def main():
    model, meta = load_model_cpu()
    tok = get_tokenizer()
    rows = []
    for slug, title, text in PARAGRAPHS:
        rows.append(plot_one(slug, title, text, model, tok))
    (OUT / 'nag_token_norm_stats.json').write_text(json.dumps(rows, indent=2))
    print('wrote', OUT)
    for row in rows:
        print(row)

if __name__ == '__main__':
    main()
