import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

STEPS_PER_EPOCH = 1680
RANDOM_CHANCE_STEP = 4000

Lpre = "de"
Lpost = "fr"
Lctrl = "en"
Ma = f"{Lpre}2{Lpost}"
Mctrl = f"{Lctrl}2{Lpost}"
Mpre = Lpre
Mpost = Lpost

# Model group definitions
pre_models      = ['de25_take2', 'de25_take3', 'de25_take6']
post_models     = ['fr25_take4', 'fr25_take6']
adoptee_models  = ['de5_fr25_take2', 'de5_fr25_take6', 'de5_fr25_take8']
control_models  = ['en5_fr25_take8', 'en5_fr25_take9', 'en5_fr25_take10']

a_pre_label     = "RSA($M_a$, $M_{pre}$) Layers 1-4"
a_post_label    = "RSA($M_a$, $M_{post}$) Layers 1-4"
ctrl_pre_label  = "RSA($M_{ctrl}$, $M_{pre}$) Layers 1-4"
ctrl_post_label = "RSA($M_{ctrl}$, $M_{post}$) Layers 1-4"
topline_label   = "RSA($M_{post}^{\\neq seed}$, $M_{post}^{\\neq seed}$)"
baseline_label  = "RSA($M_{pre}$, $M_{post}$)"
control_label   = "RSA($M_{ctrl}$, $M_{pre}$)"

FILENAME = 'figure_4_correlation_analysis.png'

C = {
    "mpre":  "#f47a00",
    "mpost": "#62d8c3",
    "madop": "#d31f11",
    "mctrl": "#007191",
}


def k_formatter(x, pos):
    if x == 0:
        return 0
    return str(int(x // 1000)) + "k" 

def k_point_one_formatter(x, pos):
    return f"{x/1000:.1f}k"

def categorize(row):
    M1 = row["model_a"]
    M2 = row["model_b"]
    if M1 == Ma and M2 == Mpre:
        return a_pre_label
    if M1 == Ma and M2 == Mpost:
        return a_post_label
    if M1 == Mctrl and M2 == Mpre:
        return ctrl_pre_label
    if M1 == Mctrl and M2 == Mpost:
        return ctrl_post_label
    if M1 == Mpost and M2 == Mpost:
        return topline_label
    if (M1 == Mpre and M2 == Mpost) or (M1 == Mpost and M2 == Mpre):
        return baseline_label
    return None  # explicit fallthrough

def window_bounds(t: int, lo: int = 4) -> tuple[int, int]:
    end = max(lo, t - 5)
    start = max(lo, end - 4)
    return start, end

def rolling_avg(df: pl.DataFrame, label: str, target_epochs: list[int], out_col: str) -> pl.DataFrame:
    src = df.filter(pl.col("category") == label).select(["epoch", "score"])
    bounds = pl.DataFrame(
        [{"target_epoch": t, "win_start": window_bounds(t)[0], "win_end": window_bounds(t)[1]}
         for t in target_epochs]
    )
    return (
        bounds.join(src, how="cross")
        .filter(pl.col("epoch").is_between(pl.col("win_start"), pl.col("win_end"), closed="both"))
        .group_by("target_epoch")
        .agg(**{out_col: pl.col("score").mean()})
    )

if __name__ == "__main__":
    # Load & label panel-A data
    df_a = (
        pl.read_csv('figure_4_example_trajectory.csv')
        .with_columns(
            category=pl.struct(["model_a", "model_b"]).map_elements(categorize, return_dtype=str),
            steps=(pl.col("epoch") - 4) * STEPS_PER_EPOCH,
        )
        .filter(pl.col("layer").is_in([1, 2, 3, 4]))
    )

    target_epochs = list(range(4, 30))  # 4..29

    df_top_win = rolling_avg(df_a, topline_label, target_epochs, "top")
    df_bot_win = rolling_avg(df_a, baseline_label, target_epochs, "bot")

    df_traj = (
        df_a.filter(
            pl.col("category").is_in([a_pre_label, a_post_label, ctrl_pre_label, ctrl_post_label])
        )
        .group_by(["steps", "epoch", "category", "seed_a"])
        .agg(mean_score=pl.col("score").mean())
        .join(df_top_win.rename({"target_epoch": "epoch"}), on="epoch")
        .join(df_bot_win.rename({"target_epoch": "epoch"}), on="epoch")
        .with_columns(
            score_norm=(pl.col("mean_score") - pl.col("bot")) / (pl.col("top") - pl.col("bot")),
        )
    )
    

    # Print 95% CI for final percentage value
    print("Panel A — stats for 36-42k steps:")
    for (cat,), grp in df_traj.filter(pl.col('steps').is_between(36_000, 42_000)).group_by('category'):
        vals = grp['score_norm'].to_numpy()
        mean, se = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        ci = 1.96 * se
        print(f"  {cat}  n={len(vals)}  mean={mean:.3f}  95% CI=[{mean-ci:.3f}, {mean+ci:.3f}]")


    # Load & reshape panel-B data
    df_b = (
        pl.read_csv('figure_4_pretrain_steps_rsa.csv')
        .group_by(["model_a", "model_b", "pre_steps", "eval_lang", "layer"]).agg(
            mean_score=pl.col("score").mean()
        )
    )
    df_base = (
        df_b.filter((pl.col("pre_steps") == 0) & (pl.col("model_a") != pl.col("model_b")))
        .rename({"mean_score": "baseline"})
        .with_columns(
            model_a=pl.col("eval_lang") + "2" + pl.when(pl.col("model_a") == pl.col("eval_lang")).then(pl.col("model_b")).otherwise(pl.col("model_a")),
        )
        .drop(["pre_steps", "model_b", "eval_lang"])
    )
    df_plot = (
        df_b.filter((pl.col("pre_steps") != 0) & (pl.col("eval_lang") == pl.col("model_b")))
        .join(df_base, on=["model_a", "layer"])
        .with_columns(
            norm_score=(pl.col("mean_score") - pl.col("baseline")) / (1 - pl.col("baseline"))
        )
        .group_by(["model_a", "pre_steps"]).agg(
            s14=pl.col("norm_score").filter(pl.col("layer").is_in([1, 2, 3, 4])).mean(),
            s512=pl.col("norm_score").filter(pl.col("layer") > 4).mean()
        )
    )

    ##############################
    # Figure
    ##############################
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5))


    # Panel A: RSA trajectory over training steps
    sns.lineplot(
        data=df_traj, x='steps', y='score_norm', hue='category', style='category',
        hue_order=[a_post_label, a_pre_label, ctrl_post_label, ctrl_pre_label],
        style_order=[a_post_label, a_pre_label, ctrl_post_label, ctrl_pre_label],
        palette=[C['madop'], C['madop'], C['mctrl'], C['mctrl']],
        dashes=[(4, 2), (), (4, 2), ()],
        linewidth=1.5, errorbar='ci', ax=ax_a
    )

    # Reference lines
    ax_a.axvline(RANDOM_CHANCE_STEP, color='#555', ls='-.', alpha=0.8)
    ax_a.text(RANDOM_CHANCE_STEP + 100, 0.45, " $L_{pre}$ Accuracy\n = Chance",  color='#555', alpha=0.8, va='center', ha='left', fontsize=9)
    ax_a.axhline(1, color='#555', ls='--', alpha=0.8)
    ax_a.axhline(0, color='#555', ls=':',  alpha=0.8)
    ax_a.text(11_000, 1.01, topline_label,  color='#555', alpha=0.8, va='bottom', ha='left', fontsize=9)
    ax_a.text(11_000, -0.03, baseline_label, color='#555', alpha=0.8, va='top', ha='left', fontsize=9)
    ax_a.set_xlabel('Number of updates on $L_{post}$ after switch', fontsize=11)
    ax_a.set_ylabel('Relative RSA (pre-phonemic)', fontsize=11)
    ax_a.set_ylim(-0.13, 1.15)
    ax_a.set_xlim(0, 42_000)
    ax_a.set_yticks([0, 0.5, 1.0])
    ax_a.set_xticks([0, 10_000, 20_000, 30_000, 40_000])
    ax_a.grid(True, alpha=0.2)
    ax_a.legend(loc='center right', handlelength=2.0, fontsize=9)
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))

    
    # Panel B: Neural traces vs. pre-training steps
    df_plot = df_plot.sort("pre_steps")
    for Lpre, Lpost in [('de', 'fr'), ('en', 'de'), ('en', 'fr'), ('fr', 'de')]:
        sub = df_plot.filter(pl.col("model_a") == f"{Lpre}2{Lpost}")
        ax_b.plot(sub["pre_steps"], sub["s14"], color="#bbb", lw=1.2, zorder=1, markersize=5)
        ax_b.text(
            x=sub["pre_steps"][-1] + 0.08 * STEPS_PER_EPOCH, y=sub["s14"][-1],
            s=f"{Lpre}→{Lpost}", color="#bbb", va="center", ha="left", fontsize=8,
        )

    # Aggregate trendlines
    #agg = df_b.groupby('steps')[['s14', 's512']].mean()
    agg = df_plot.group_by("pre_steps").mean().sort("pre_steps")
    ax_b.plot(agg["pre_steps"], agg['s14'],  color=C['madop'], lw=2.0, label='Pre-phonemic layers (1–4)',
        marker='o', markeredgecolor="white", markersize=10)
    ax_b.plot(agg["pre_steps"], agg['s512'], color='#555', lw=2.0, label='Other 8 encoder layers',
        ls='--', marker='o', markeredgecolor="white", markersize=10)

    ax_b.set_xlabel('Number of updates on $L_{pre}$ before switch', fontsize=11)
    ax_b.set_ylabel('Relative RSA', fontsize=11)
    step_vals = sorted(df_plot['pre_steps'].unique())
    ax_b.set_xticks(step_vals)
    ax_b.set_xticklabels([str(int(s)) for s in step_vals])
    ax_b.set_xlim(step_vals[0] - 0.2 * STEPS_PER_EPOCH, step_vals[-1] + 0.5 * STEPS_PER_EPOCH)
    ax_b.set_ylim(-0.02, 0.21)
    ax_b.set_yticks([0, 0.05, 0.1, 0.15, 0.20])
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(frameon=True, loc='upper left', handlelength=4.0, fontsize=9)
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(k_point_one_formatter))

    # Panel labels
    for ax, label in zip([ax_a, ax_b], ['a', 'b']):
        ax.text(0.05, 1.06, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout()
    plt.savefig(FILENAME, dpi=150)
    print(f"Saved {FILENAME}")
