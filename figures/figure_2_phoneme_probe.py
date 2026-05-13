import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

# ── Palette (Wong colourblind-safe) ───────────────────────────────────────────
C = {
    "de":    "#0072B2",   # blue   — ceiling
    "fr":    "#E69F00",   # amber  — floor
    "de2fr": "#009E73",   # teal   — adoptee
}
LABELS = {
    "de":    "$M_{post}$ (German only)",
    "fr":    "$M_{pre}$ (French only)",
    "de2fr": "$M_A$ (Adoptee model)",
}

# ── Load & aggregate across seeds ────────────────────────────────────────────
raw = pd.read_csv("figure_2_phoneme_probe.csv")
#LAYERS  = list(range(13))
LAYERS  = list(range(1, 13))
MODELS  = ["de", "fr", "de2fr"]
MIN_GAP = 0.01

def col(m, l): return f"{m}_layer_{l}"

# Mean and SD over seeds at each update step
grp  = raw.groupby("Updates")
mean = grp.mean(numeric_only=True)
sd   = grp.std(numeric_only=True)
updates = mean.index.values

# Final-step statistics
final_mean = mean.iloc[-1] * 100
final_sd   = sd.iloc[-1] * 100

fmean = {m: np.array([final_mean[col(m, l)] for l in LAYERS]) for m in MODELS}
fsd   = {m: np.array([final_sd  [col(m, l)] for l in LAYERS]) for m in MODELS}

# Relative recovery: (de2fr − fr) / (de − fr)
delta_final = fmean["de2fr"] - fmean["fr"]
gap_final   = fmean["de"]    - fmean["fr"]

with np.errstate(invalid="ignore", divide="ignore"):
    relative = np.where(gap_final >= MIN_GAP, delta_final / gap_final, np.nan)

# SD of relative recovery via error propagation (first-order)
# rel = (A - B) / (C - B), where A=de2fr, B=fr, C=de
A, B, Cg = fmean["de2fr"], fmean["fr"], fmean["de"]
sA, sB, sC = fsd["de2fr"], fsd["fr"], fsd["de"]
denom = gap_final
with np.errstate(invalid="ignore", divide="ignore"):
    drel_dA =  1.0 / denom
    drel_dB = -(Cg - A) / denom**2  # ∂rel/∂B = -(C-A)/denom^2; but simpler:
    # rel = (A-B)/(C-B), d/dA=1/denom, d/dB=-(C-A)/(denom^2)... let's just do numeric
    # Actually: rel = (A-B)/(C-B)
    # d/dA = 1/(C-B)
    # d/dB = [-(C-B) - (A-B)*(-1)] / (C-B)^2 = [-(C-B) + (A-B)] / (C-B)^2 = (A-C)/(C-B)^2
    # d/dC = -(A-B)/(C-B)^2
    drel_dA =  1.0 / denom
    drel_dB = (A - Cg) / denom**2
    drel_dC = -(A - B) / denom**2
    rel_sd  = np.sqrt((drel_dA*sA)**2 + (drel_dB*sB)**2 + (drel_dC*sC)**2)
    rel_sd  = np.where(gap_final >= MIN_GAP, rel_sd, np.nan)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(nrows=1, ncols=2, figsize=(9, 2))

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A — Convergence accuracy (mean ± 1 SD ribbon)
# ─────────────────────────────────────────────────────────────────────────────
for m in ("fr", "de", "de2fr"):        # draw fr first so ribbons don't overlap key lines
    ax_a.fill_between(LAYERS,
                      fmean[m] - fsd[m],
                      fmean[m] + fsd[m],
                      color=C[m], alpha=0.15, zorder=1)
    ax_a.plot(LAYERS, fmean[m], color=C[m], lw=1.6,
              marker="o", ms=4, markeredgewidth=0.5,
              markeredgecolor="white", zorder=3, label=LABELS[m])

# Shade layers where relative recovery ≥ 0.45
#savings_layers = [l for l in LAYERS if not np.isnan(relative[l]) and relative[l] >= 0.45]
#if savings_layers:
#    lo, hi = min(savings_layers) - 0.4, max(savings_layers) + 0.4
#    ax_a.axvspan(lo, hi, color=C["de2fr"], alpha=0.07, zorder=0)

ax_a.set_xlim(0.7, 12.3)
ax_a.set_xticks(LAYERS)
ax_a.set_xticklabels(LAYERS, fontsize=7.5)
ax_a.set_xlabel("Layer index", labelpad=3)
ax_a.set_ylabel("Probe Accuracy", labelpad=3)
#ax_a.set_title("(a) Convergence accuracy by layer", fontsize=9, pad=5, loc="left")
ax_a.legend(fontsize=9, loc="lower right", handlelength=1.6,
            handletextpad=0.5, borderpad=0.5, framealpha=0.5)
ax_a.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax_a.set_ylim(47, 83)
ax_a.grid(True, alpha=0.3)

# Seeds note
ax_a.text(0.02, 0.97, "mean ± 1 SD,  n = 3 seeds",
          transform=ax_a.transAxes, fontsize=6.2, va="top",
          color="#666666", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B — Relative recovery (de2fr − fr) / (de − fr) ± propagated SD
# ─────────────────────────────────────────────────────────────────────────────
valid   = ~np.isnan(relative)
invalid = ~valid
x_valid = np.array(LAYERS)[valid]

ax_b.fill_between(x_valid,
                  relative[valid] - rel_sd[valid],
                  relative[valid] + rel_sd[valid],
                  color=C["de2fr"], alpha=0.2, zorder=1)
#ax_b.fill_between(x_valid, 0, relative[valid],
#                  color=C["de2fr"], alpha=0.12, zorder=1)
ax_b.plot(x_valid, relative[valid],
          color=C["de2fr"], lw=1.8, marker="o", ms=4,
          markeredgewidth=0.5, markeredgecolor="white", zorder=3, label=LABELS["de2fr"])

# Masked points
#if invalid.any():
#    ax_b.scatter(np.array(LAYERS)[invalid],
#                 np.zeros(invalid.sum()),
#                 color="#aaaaaa", s=16, zorder=4,
#                 label=f"gap < {MIN_GAP} (masked)")

# Reference lines
ax_b.axhline(0,   color="#555555", lw=0.8, ls="--", zorder=2)
ax_b.axhline(1.0, color="#555555", lw=0.8, ls=":",  zorder=2)
ax_b.text(12.0, 0.97, "No forgetting\n(same as $M_{pre}$)", fontsize=8, va="top",
          ha="right", color="#555555", style="italic")
ax_b.text(12.0, 0.03, "No savings\n(same as $M_{post}$)",  fontsize=8, va="bottom",
          ha="right", color="#555555", style="italic")

ax_b.set_xlim(0.7, 12.3)
ax_b.set_xticks(LAYERS)
ax_b.set_xticklabels(LAYERS, fontsize=7.5)
ax_b.set_ylim(-0.15, 1.18)
ax_b.set_xlabel("Layer index", labelpad=3)
#ax_b.set_ylabel(r"Relative Acc. $(\Delta_{de2fr}\,/\,\Delta_{de})$", labelpad=3)
ax_b.set_ylabel(r"Relative Accuracy", labelpad=3)

ax_b.legend(fontsize=9, loc="right", handlelength=1.6,
            handletextpad=0.5, borderpad=0.5, framealpha=0.5)
ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax_b.grid(True, alpha=0.3)

ax_b.text(0.02, 0.97, "± propagated SD,  n = 3 seeds",
          transform=ax_b.transAxes, fontsize=6.2, va="top",
          color="#666666", style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
out = "journal_probe_de2fr.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")

# Quick diagnostics
print("\nRelative recovery by layer:")
for l in LAYERS:
    r = relative[l-1]
    s = rel_sd[l-1] if not np.isnan(relative[l-1]) else float("nan")
    g = gap_final[l-1]
    print(f"  layer {l:2d}:  gap={g:.4f}  rel={r:.3f}  ±{s:.3f}" if not np.isnan(r)
          else f"  layer {l:2d}:  gap={g:.4f}  MASKED")
