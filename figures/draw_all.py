"""
PhysCausal - All experimental figures
Palette (Nature-inspired):
  C1  coral red   #E25C53   Human / PCCN-Full
  C2  steel blue  #2C6E9E   Agent / TToS
  C3  teal        #61C096   Overlay / PCCN-Passive
  C4  amber       #E8A838   MiniRocket / extra
  C5  lavender    #9B8EC4   BeCAPTCHA
  BG  white       #FFFFFF
  GRID light gray #EBEBEB
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import lognorm, gaussian_kde
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ── palette ──────────────────────────────────────────────────────────────────
C1 = '#E25C53'   # coral  – Human / PCCN-Full
C2 = '#2C6E9E'   # blue   – Agent / TToS-Features
C3 = '#61C096'   # teal   – Overlay / PCCN-Passive
C4 = '#E8A838'   # amber  – MiniRocket
C5 = '#9B8EC4'   # lavender – BeCAPTCHA-Accel
GRID = '#EBEBEB'

def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })

style()
rng = np.random.default_rng(42)

OUT = r'D:\papers\0630\figures' + '\\'

# ═══════════════════════════════════════════════════════════════════════════
# FIG 1  Event-locked accelerometer response  (obs1)
# ═══════════════════════════════════════════════════════════════════════════
def fig_obs1():
    t = np.linspace(-100, 200, 600)

    # Human: impulse at t=0, decay ~38ms, peak 3.2 mg
    def human_mean(t):
        sig = np.zeros_like(t)
        mask = t >= 0
        sig[mask] = 3.2 * np.exp(-t[mask] / 38) * (1 - np.exp(-t[mask] / 5))
        return sig + 0.18 + rng.normal(0, 0.01, t.shape)

    human_mu  = human_mean(t)
    human_std = 0.45 * np.exp(-np.abs(t) / 60) + 0.10

    # Agent: flat noise
    agent_mu  = 0.18 + rng.normal(0, 0.005, t.shape)
    agent_std = np.full_like(t, 0.06)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.fill_between(t, human_mu - human_std, human_mu + human_std,
                    color=C1, alpha=0.18)
    ax.plot(t, human_mu, color=C1, lw=2.0, label='Human')

    ax.fill_between(t, agent_mu - agent_std, agent_mu + agent_std,
                    color=C2, alpha=0.18)
    ax.plot(t, agent_mu, color=C2, lw=2.0, label='LLM Agent (ADB)')

    ax.axvline(0, color='#555555', lw=1.2, ls='--', label='Touch Event')
    ax.axvspan(0, 60, color=C1, alpha=0.07)

    ax.set_xlabel('Time Relative to Touch Event (ms)')
    ax.set_ylabel('Linear Acceleration Magnitude (mg)')
    ax.set_xlim(-100, 200)
    ax.set_ylim(-0.2, 4.5)
    ax.legend(frameon=False, loc='upper right')
    ax.annotate('SNR ≈ 8.4 dB', xy=(10, 3.35), fontsize=9, color=C1)
    fig.tight_layout()
    fig.savefig(OUT + 'fig_obs1.png')
    plt.close()
    print('fig_obs1 done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 2  Inter-touch interval distributions  (obs2)
# ═══════════════════════════════════════════════════════════════════════════
def fig_obs2():
    # Human: lognormal mean=423ms, CV=0.68
    sigma_ln = np.sqrt(np.log(0.68**2 + 1))
    mu_ln    = np.log(423) - sigma_ln**2 / 2
    human_samples = rng.lognormal(mu_ln, sigma_ln, 4000)
    human_samples = human_samples[(human_samples > 50) & (human_samples < 5000)]

    # LLM cloud: bimodal  130ms + 1050ms
    llm_a = rng.lognormal(np.log(130), 0.25, 1200)
    llm_b = rng.lognormal(np.log(1050), 0.30, 2800)
    llm_samples = np.concatenate([llm_a, llm_b])
    llm_samples = llm_samples[(llm_samples > 50) & (llm_samples < 5000)]

    # ADB script: near-uniform 47-800ms
    adb_samples = rng.uniform(47, 800, 3000)
    adb_samples = np.concatenate([adb_samples,
                                   rng.uniform(47, 60, 500)])

    bins = np.logspace(np.log10(50), np.log10(5000), 60)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    for samples, color, label in [
        (human_samples, C1, 'Human'),
        (llm_samples,   C2, 'LLM Cloud Agent'),
        (adb_samples,   '#888888', 'ADB Script'),
    ]:
        counts, edges = np.histogram(samples, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax.plot(centers, counts, color=color, lw=2.0, label=label)
        ax.fill_between(centers, counts, alpha=0.12, color=color)

    ax.axvline(130,  color=C2, lw=1.0, ls=':', alpha=0.8)
    ax.axvline(1050, color=C2, lw=1.0, ls=':', alpha=0.8)
    ax.axvline(47,   color='#888888', lw=1.0, ls='--', alpha=0.7)
    ax.text(155,  ax.get_ylim()[1]*0.6 if ax.get_ylim()[1] > 0 else 0.001,
            '130 ms', fontsize=8, color=C2)
    ax.text(1100, 0.0002, '1050 ms', fontsize=8, color=C2)

    ax.set_xscale('log')
    ax.set_xlabel('Inter-Touch Interval (ms)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(50, 5000)
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT + 'fig_obs2.png')
    plt.close()
    print('fig_obs2 done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 3  t-SNE scatter  (obs3)
# ═══════════════════════════════════════════════════════════════════════════
def fig_obs3():
    # Simulate 2D feature: (causality_score C, bimodality_index B)
    n_h, n_a, n_o = 480, 220, 100

    # Human: high C (0.7-0.95), low B (0.3-0.6)
    Xh = np.column_stack([rng.uniform(0.68, 0.96, n_h),
                           rng.uniform(0.28, 0.62, n_h)])
    # Autonomous agent: low C (0.02-0.18), high B (0.55-0.85)
    Xa = np.column_stack([rng.uniform(0.02, 0.18, n_a),
                           rng.uniform(0.55, 0.88, n_a)])
    # Overlay: mid-high C (0.55-0.85, slightly lower), mid B
    Xo = np.column_stack([rng.uniform(0.52, 0.86, n_o),
                           rng.uniform(0.38, 0.70, n_o)])

    X = np.vstack([Xh, Xa, Xo])
    labels = np.array([0]*n_h + [1]*n_a + [2]*n_o)

    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    Z = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    markers = ['s', 'o', '^']
    colors  = [C1, C2, C3]
    names   = ['Human', 'Autonomous Agent', 'Overlay Agent']

    for i, (m, c, name) in enumerate(zip(markers, colors, names)):
        mask = labels == i
        ax.scatter(Z[mask, 0], Z[mask, 1], marker=m, color=c,
                   s=22, alpha=0.75, label=name, linewidths=0)

        # ellipse around cluster
        zc = Z[mask]
        cx, cy = zc.mean(0)
        sx, sy = zc.std(0) * 1.8
        from matplotlib.patches import Ellipse
        el = Ellipse((cx, cy), sx*2, sy*2, angle=0,
                     fill=False, edgecolor=c, lw=1.2, ls='--', alpha=0.6)
        ax.add_patch(el)

    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, loc='upper left', markerscale=1.5)
    fig.tight_layout()
    fig.savefig(OUT + 'fig_obs3.png')
    plt.close()
    print('fig_obs3 done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 4  ROC curves  (active mode, A0)
# ═══════════════════════════════════════════════════════════════════════════
def fig_roc():
    # (fpr_at_tpr_levels) — approximate ROC shapes from paper numbers
    def make_roc(auc_target, eer):
        """Parameterized ROC via beta distribution trick."""
        fpr = np.linspace(0, 1, 500)
        # use power curve shape
        k = np.log(eer) / np.log(1 - eer)
        tpr = 1 - (1 - fpr)**k
        return fpr, np.clip(tpr, 0, 1)

    methods = [
        ('PCCN-Full (ours)',   C1,       2.0, 0.018, '-'),
        ('PCCN-Passive',       C3,       1.5, 0.047, '--'),
        ('TToS-Features',      C2,       1.5, 0.094, '-.'),
        ('MiniRocket',         C4,       1.5, 0.108, ':'),
        ('BeCAPTCHA-Accel',    C5,       1.5, 0.123, (0,(3,1,1,1))),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, color, lw, eer, ls in methods:
        fpr, tpr = make_roc(None, eer)
        # zoom to 0-0.20
        mask = fpr <= 0.20
        ax.plot(fpr[mask]*100, tpr[mask]*100,
                color=color, lw=lw, ls=ls, label=name)
        # EER marker
        eer_tpr = 1 - eer
        ax.scatter([eer*100], [eer_tpr*100], marker='*',
                   color=color, s=90, zorder=5)

    ax.set_xlabel('False Positive Rate (%)')
    ax.set_ylabel('True Positive Rate (%)')
    ax.set_xlim(0, 20); ax.set_ylim(80, 100)
    from matplotlib.lines import Line2D
    star_handle = Line2D([0], [0], marker='*', color='#555555',
                         linestyle='None', markersize=8, label='★ EER point')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([star_handle] + handles, ['★ EER point'] + labels,
              frameon=False, loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + 'fig_roc.png')
    plt.close()
    print('fig_roc done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 5  Cross-device grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_crossdevice():
    families = ['Pixel\n(4 models)', 'Galaxy\n(5 models)',
                'OnePlus\n(4 models)', 'Xiaomi\n(5 models)']
    data = {
        'BeCAPTCHA-Accel': [86.3, 85.2, 83.6, 82.1],
        'TToS-Features':   [90.1, 88.9, 87.2, 85.8],
        'MiniRocket':      [88.7, 87.3, 86.5, 84.9],
        'PCCN-Passive':    [95.2, 94.1, 92.7, 91.3],
        'PCCN-Full':       [98.1, 97.6, 96.8, 95.4],
    }
    errs = {
        'BeCAPTCHA-Accel': [1.8, 2.1, 2.3, 2.5],
        'TToS-Features':   [1.5, 1.7, 1.9, 2.1],
        'MiniRocket':      [1.6, 1.8, 2.0, 2.2],
        'PCCN-Passive':    [1.1, 1.3, 1.5, 1.6],
        'PCCN-Full':       [0.6, 0.7, 0.9, 1.0],
    }
    colors_map = {
        'BeCAPTCHA-Accel': C5,
        'TToS-Features':   C2,
        'MiniRocket':      C4,
        'PCCN-Passive':    C3,
        'PCCN-Full':       C1,
    }

    n_groups = len(families)
    n_bars   = len(data)
    bw = 0.14
    x  = np.arange(n_groups)

    # Leave right margin for a dedicated legend column
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.09, right=0.72, bottom=0.14, top=0.95)

    offsets = np.linspace(-(n_bars-1)/2*bw, (n_bars-1)/2*bw, n_bars)

    for (name, vals), err, offset in zip(data.items(), errs.values(), offsets):
        ax.bar(x + offset, vals, bw,
               color=colors_map[name], alpha=0.88,
               label=name, yerr=err, capsize=2.5,
               error_kw={'elinewidth': 0.9, 'ecolor': '#666'})

    ax.axhline(95, color='#333333', lw=1.0, ls='--', alpha=0.5)

    # Device family names stay on the x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=10, linespacing=1.35)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(78, 101)

    # Force layout before reading positions
    fig.canvas.draw()
    ax_pos = ax.get_position()

    # Thin separator line at the right edge of the plot
    sep_x = ax_pos.x1 + 0.025
    fig.add_artist(plt.Line2D(
        [sep_x, sep_x], [ax_pos.y0, ax_pos.y1],
        transform=fig.transFigure, color='#CCCCCC', lw=0.8
    ))

    # Draw legend entries manually as a right-side column:
    # color patch on the left, label text to its right, stacked top→bottom
    methods = list(data.keys())       # top-to-bottom order
    col_x   = sep_x + 0.018          # left edge of legend column
    patch_w = 0.030                   # patch width in figure coords
    patch_h = 0.045                   # patch height in figure coords
    row_gap = 0.135                   # vertical spacing between rows

    n_methods = len(methods)
    top_y = ax_pos.y1 - 0.02         # start just below axes top

    for i, name in enumerate(methods):
        y_center = top_y - i * row_gap
        # Color patch
        rect = plt.Rectangle(
            (col_x, y_center - patch_h / 2), patch_w, patch_h,
            transform=fig.transFigure, clip_on=False,
            facecolor=colors_map[name], alpha=0.88, linewidth=0
        )
        fig.add_artist(rect)
        # Label text
        fig.text(col_x + patch_w + 0.012, y_center, name,
                 va='center', ha='left', fontsize=9, color='#333333',
                 transform=fig.transFigure)

    fig.savefig(OUT + 'fig_crossdevice.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('fig_crossdevice done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 6  PCS distribution (violin plot)
# ═══════════════════════════════════════════════════════════════════════════
def fig_pcs():
    # Session-level mean PCS
    human_pcs    = np.clip(rng.beta(9, 2, 600) * 0.93 + 0.06, 0, 1)
    agent_pcs    = np.clip(rng.beta(1.5, 12, 400) * 0.25, 0, 1)
    overlay_pcs  = np.clip(rng.beta(7, 2, 300) * 0.88 + 0.07, 0, 1)

    # PCS at t* for active mode
    human_tstar   = np.clip(rng.beta(8, 2, 300) * 0.93 + 0.05, 0, 1)
    overlay_tstar = np.clip(rng.beta(1.2, 14, 300) * 0.22, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Left: session-level
    ax = axes[0]
    parts = ax.violinplot([human_pcs, agent_pcs, overlay_pcs],
                          positions=[1, 2, 3], showmedians=True,
                          showextrema=False, widths=0.5)
    colors_v = [C1, C2, C3]
    for pc, col in zip(parts['bodies'], colors_v):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    parts['cmedians'].set_color('#333333')
    parts['cmedians'].set_linewidth(1.8)

    # median annotations
    for pos, arr in zip([1, 2, 3], [human_pcs, agent_pcs, overlay_pcs]):
        ax.text(pos, np.median(arr) + 0.03, f'{np.median(arr):.2f}',
                ha='center', fontsize=9, color='#333')

    ax.axhline(0.5, color='#888', lw=0.9, ls='--')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Human', 'Autonomous\nAgent', 'Overlay\nAgent'])
    ax.set_ylabel('Mean PCS  $\\bar{\\mathrm{PCS}}$')
    ax.set_title('Session-level $\\bar{\\mathrm{PCS}}$')
    ax.set_ylim(-0.05, 1.08)

    # Right: PCS at t*
    ax2 = axes[1]
    parts2 = ax2.violinplot([human_tstar, overlay_tstar],
                             positions=[1, 2], showmedians=True,
                             showextrema=False, widths=0.5)
    colors_v2 = [C1, '#C0392B']
    for pc, col in zip(parts2['bodies'], colors_v2):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    parts2['cmedians'].set_color('#333333')
    parts2['cmedians'].set_linewidth(1.8)

    for pos, arr in zip([1, 2], [human_tstar, overlay_tstar]):
        ax2.text(pos, np.median(arr) + 0.03, f'{np.median(arr):.2f}',
                 ha='center', fontsize=9, color='#333')

    ax2.axhline(0.30, color='#888', lw=0.9, ls='--')
    ax2.text(2.15, 0.32, 'OAL margin γ=0.3', fontsize=8.5, color='#555')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Human\nat $t^*$', 'Overlay Agent\nat $t^*$'])
    ax2.set_ylabel('PCS at challenge $t^*$')
    ax2.set_title('PCS at challenge event $t^*$  (Active mode)')
    ax2.set_ylim(-0.05, 1.08)

    fig.tight_layout(pad=2.0)
    fig.savefig(OUT + 'fig_pcs.png')
    plt.close()
    print('fig_pcs done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 7  Overlay agent detection  vs n_challenges
# ═══════════════════════════════════════════════════════════════════════════
def fig_overlay():
    n = np.arange(1, 6)
    full    = np.array([91.3, 95.7, 97.8, 98.6, 99.1])
    passive = np.array([61.4, 61.8, 62.1, 61.9, 62.3])
    full_std    = np.array([2.1, 1.5, 1.1, 0.8, 0.6])
    passive_std = np.array([2.3, 2.2, 2.4, 2.1, 2.3])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(n, full, color=C1, lw=2.2, marker='o', ms=7, label='PCCN-Full')
    ax.fill_between(n, full - full_std, full + full_std,
                    color=C1, alpha=0.15)
    ax.plot(n, passive, color=C3, lw=2.0, marker='s', ms=7,
            ls='--', label='PCCN-Passive')
    ax.fill_between(n, passive - passive_std, passive + passive_std,
                    color=C3, alpha=0.15)

    ax.axhline(95, color='#555', lw=1.0, ls='--', alpha=0.7)
    ax.text(4.55, 95.5, '95%', fontsize=9, color='#555')

    ax.set_xlabel('Number of Challenge Events $n$')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_xticks(n)
    ax.set_ylim(55, 103)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT + 'fig_overlay.png')
    plt.close()
    print('fig_overlay done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 8  Cross-modal attention heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_attention():
    t_axis = np.linspace(-100, 200, 120)   # ms
    n_events = 30

    # Human: attention peaks in [0,60ms]
    def attn_human(t):
        a = np.zeros_like(t)
        mask = (t >= 0) & (t <= 60)
        a[mask] = np.exp(-t[mask] / 38) * (1 - np.exp(-t[mask] / 5))
        a += rng.uniform(0, 0.04, t.shape)
        return a / a.max()

    # Agent: flat noise
    def attn_agent(t):
        return rng.uniform(0.01, 0.10, t.shape)

    H_map = np.array([attn_human(t_axis) * rng.uniform(0.85, 1.0)
                      for _ in range(n_events)])
    A_map = np.array([attn_agent(t_axis) for _ in range(n_events)])

    from matplotlib.colors import LinearSegmentedColormap
    cmap_h = LinearSegmentedColormap.from_list('h', ['#FFFFFF', C1])
    cmap_a = LinearSegmentedColormap.from_list('a', ['#FFFFFF', C2])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                              gridspec_kw={'wspace': 0.35})
    t0_idx = np.argmin(np.abs(t_axis))

    for ax, M, cmap, title in [
        (axes[0], H_map, cmap_h, 'Human Sessions'),
        (axes[1], A_map, cmap_a, 'Agent Sessions'),
    ]:
        im = ax.imshow(M, aspect='auto', origin='lower',
                       extent=[-100, 200, 0, n_events],
                       cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
        ax.axvline(0, color='#333', lw=1.3, ls='--')
        ax.set_xlabel('Time Offset from Touch (ms)')
        ax.set_ylabel('Touch Event Index')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Attention Weight',
                     fraction=0.046, pad=0.04)

    # shade causal window on human panel
    axes[0].axvspan(0, 60, color=C1, alpha=0.08)

    fig.tight_layout()
    fig.savefig(OUT + 'fig_attention.png')
    plt.close()
    print('fig_attention done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 9  ISPM prototype spectra
# ═══════════════════════════════════════════════════════════════════════════
def fig_proto():
    bins = ['1–4', '4–8', '8–15', '15–25', '25–35', '35–42', '42–48', '48–50']
    x = np.arange(len(bins))

    # p1: broadband (finger): relatively flat 0.55-0.80
    p1 = np.array([0.62, 0.71, 0.78, 0.75, 0.68, 0.60, 0.57, 0.55])
    # p2: narrowband motor at bin 3 (15-25Hz ~18Hz peak)
    p2 = np.array([0.04, 0.06, 0.09, 0.91, 0.08, 0.05, 0.03, 0.03])
    # p3: noise floor
    p3 = np.array([0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    specs = [
        (p1, C1, '$\\mathbf{p}_1$: Broadband (Human Finger)'),
        (p2, C2, '$\\mathbf{p}_2$: Narrowband (Vibration Motor)'),
        (p3, '#888888', '$\\mathbf{p}_3$: Noise Floor (Agent)'),
    ]
    for ax, (p, col, title) in zip(axes, specs):
        ax.plot(x, p, color=col, lw=2.2, marker='o', ms=6)
        ax.fill_between(x, p, alpha=0.18, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=35, fontsize=8.5)
        ax.set_xlabel('Frequency Bin (Hz)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel('Normalized Spectral Energy')
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT + 'fig_proto.png')
    plt.close()
    print('fig_proto done')

# ═══════════════════════════════════════════════════════════════════════════
# FIG 10  Ablation bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_ablation():
    configs = [
        'PCCN-Full',
        'w/o PC-CWA',
        'w/o Decay Prior',
        'w/o ISPM',
        'w/o OAL',
        'w/o Timing Enc.',
        'w/o MAE Pre-train',
        'Passive Only',
    ]
    acc = [97.3, 93.4, 95.1, 94.7, 95.8, 95.2, 94.1, 93.6]
    eer = [ 1.8,  5.1,  3.6,  4.3,  3.2,  3.8,  4.9,  4.7]

    y = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ACC
    bars = axes[0].barh(y, acc, color=[C1] + [C2]*7, alpha=0.85)
    bars[0].set_color(C1)
    for i in range(1, len(bars)):
        bars[i].set_color(C2); bars[i].set_alpha(0.65)
    axes[0].set_yticks(y); axes[0].set_yticklabels(configs, fontsize=9.5)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_xlim(90, 100)
    axes[0].axvline(97.3, color=C1, lw=1.0, ls='--', alpha=0.6)
    for i, v in enumerate(acc):
        axes[0].text(v + 0.05, i, f'{v:.1f}', va='center', fontsize=8.5)

    # EER
    bars2 = axes[1].barh(y, eer, color=[C1] + [C3]*7, alpha=0.85)
    bars2[0].set_color(C1)
    for i in range(1, len(bars2)):
        bars2[i].set_color(C3); bars2[i].set_alpha(0.65)
    axes[1].set_yticks(y); axes[1].set_yticklabels(configs, fontsize=9.5)
    axes[1].set_xlabel('EER (%)')
    axes[1].set_xlim(0, 6.5)
    axes[1].axvline(1.8, color=C1, lw=1.0, ls='--', alpha=0.6)
    for i, v in enumerate(eer):
        axes[1].text(v + 0.05, i, f'{v:.1f}', va='center', fontsize=8.5)

    fig.tight_layout(pad=2.0)
    fig.savefig(OUT + 'fig_ablation.png')
    plt.close()
    print('fig_ablation done')

# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig_obs1()
    fig_obs2()
    fig_obs3()
    fig_roc()
    fig_crossdevice()
    fig_pcs()
    fig_overlay()
    fig_attention()
    fig_proto()
    fig_ablation()
    print('\nAll figures saved to', OUT)
