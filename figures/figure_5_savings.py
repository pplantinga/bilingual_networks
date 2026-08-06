import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from matplotlib import ticker

# CONSTANTS
THRESHOLD_PCT = 0.7
MA_LABEL = '$M_a$'
MC_LABEL = '$M_{ctrl}$'
Mpre_LABEL = '$M_{pre}$'
Mpost_LABEL = '$M_{post}$'
FIGURE_FILENAME = 'figure_5_speed.png'
C = {
    "mpre":  "#f47a00",
    "mpost": "#62d8c3",
    "madop": "#d31f11",
    "mctrl": "#007191",
}
Lpre = "de"
Lpost = "fr"
Lctrl = "en"

def steps_to_threshold(df, threshold_pct):
    '''Interpolated steps to reach threshold_pct'''
    above = df[df['valid_acc'] >= threshold_pct]
    if above.empty:
        return None
    idx = above.index[0]
    if idx == 0:
        return float(df.loc[0, 'steps'])
    x0, y0 = df.loc[idx - 1, 'steps'], df.loc[idx - 1, 'valid_acc']
    x1, y1 = df.loc[idx,     'steps'], df.loc[idx,     'valid_acc']
    return x0 + (threshold_pct - y0) * (x1 - x0) / (y1 - y0) 


def plot_trendline(ax, subset, color, label=None, zorder=0):

    slope, intercept, r_val, p_val, std_err = linregress(subset['score'], subset['steps'])
    x_range = np.array([subset['score'].min(), subset['score'].max()])
    p, = ax.plot(x_range, intercept + slope * x_range, color=color, linestyle='--', alpha=0.6, label=label, zorder=zorder)
    return p, r_val, p_val

if __name__ == '__main__':

    # 1. Load Data
    acc_df = pd.read_csv('figure_5_learning_curve.csv')
    replace_df = pd.read_csv('figure_5_replacement.csv')

    # 2. Categorize
    def get_cat(name):
        if not name:
            return None
        if name.startswith(f'{Lpre}5_{Lpost}25'):
            return MA_LABEL
        if name.startswith(f'{Lctrl}5_{Lpost}25'):
            return MC_LABEL
        if name.startswith(f'{Lpost}25'):
            return Mpost_LABEL
        if name.startswith(f'{Lpre}25'):
            return Mpre_LABEL
        return None

    acc_df['category'] = acc_df['label'].apply(get_cat)

    # 3. Interpolate learning curve
    model_steps = {'label': [], 'steps': [], 'category': []}
    for label in acc_df['label'].unique():
        model_df = acc_df[acc_df.label == label]
        steps = steps_to_threshold(model_df, THRESHOLD_PCT)
        model_steps['label'].append(label)
        model_steps['steps'].append(steps)
        model_steps['category'].append(get_cat(label))
    model_steps = pd.DataFrame(model_steps)

    # 4. Compute speedups
    replace_steps = {'layers': [], 'seed': [], 'dir': [], 'base2targ_steps': [], 'targ2base_steps': []}
    for direction in replace_df.dir.unique():
        dir_df = replace_df[replace_df.dir == direction]
        base, target = sorted(dir_df.base.unique(), key=lambda x: len(x))
        for (seed, layers), df in dir_df.groupby(['seed', 'layers']):
            steps = steps_to_threshold(df[df.base == base], THRESHOLD_PCT)
            reverse = steps_to_threshold(df[df.base == target], THRESHOLD_PCT)
            replace_steps["layers"].append(layers)
            replace_steps["seed"].append(seed)
            replace_steps["dir"].append(direction)
            replace_steps["base2targ_steps"].append(steps)
            replace_steps["targ2base_steps"].append(reverse)

    replace_steps = pd.DataFrame(replace_steps)
    replace_steps["speedup"] = (replace_steps.base2targ_steps - replace_steps.targ2base_steps) / replace_steps.targ2base_steps

    #############
    # 5. Plotting
    #############
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    colors = {MA_LABEL: C['madop'], MC_LABEL: C['mctrl'], Mpost_LABEL: C['mpost']}
    shapes = {MA_LABEL: "o", MC_LABEL: "s", Mpost_LABEL: "^"}

    layer_order = ['1-4', '5-8', '9-12']
    layer_palette = dict(zip(layer_order, sns.color_palette("flare", 3)))
    dir_labels = {'de2fr': f'{Lpre}\u2192{Lpost}', 'fr2en': f'{Lpost}\u2192{Lctrl}', 'en2de': f'{Lctrl}\u2192{Lpre}'}

    # Panel 1: Learning Curves
    sns.lineplot(data=acc_df, x='steps', y='valid_acc', hue='category', palette=colors, ax=ax1, lw=2)
    ax1.set_ylabel('Next-Token Accuracy')
    ax1.set_xlabel('Number of updates on $L_{pre}$ (German)')
    ax1.axhline(THRESHOLD_PCT, color='black', linestyle=':', alpha=0.8)
    ax1.annotate(f'{THRESHOLD_PCT:.0%} accuracy\nthreshold', xy=(300, THRESHOLD_PCT), xytext=(50, THRESHOLD_PCT - 0.12),
                 color='black', va="top", arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=9)
    ax1.get_legend().set_title(None)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0, right=1050)
    ax1.set_ylim(bottom=0, top=0.9)

    # Plot intercepts
    cat_steps = model_steps.groupby(['category'])['steps'].mean().reset_index()
    for category, steps in zip(cat_steps['category'], cat_steps['steps']):
        ax1.axvline(steps, color=colors[category], linestyle="--")

    # Annotate the gap
    left = min(cat_steps['steps'])
    right = max(cat_steps['steps'])
    relative_speedup = (right - left) / left
    ax1.annotate(f'$M_a$ learns\n{relative_speedup:.0%} faster', xy=(left-10, 0.51), xytext=(right+20, 0.51),
                 arrowprops=dict(arrowstyle='<|-|>', color="black", lw=2), va="center", fontsize=9)
    ax1.fill_between([left, right], y1=0, y2=1.0, color='grey', alpha=0.2)

    # Print the CI on the gap
    #print(f"Relative speedup: {relative_speedup:.1%}")
    def bootstrap_ci(df, catA, catB, nsamples=10000, ci_range=0.95):
        stepsA = df[df.category == catA]["steps"].to_numpy()
        stepsB = df[df.category == catB]["steps"].to_numpy()

        # Vectorized bootstrap across nsamples iterations
        rng = np.random.default_rng()
        bootA = rng.choice(stepsA, size=(nsamples, len(stepsA)), replace=True).mean(axis=1)
        bootB = rng.choice(stepsB, size=(nsamples, len(stepsB)), replace=True).mean(axis=1)
        diffs = (bootA - bootB) / bootB

        # Convert CI bounds to 0-100 scale (e.g., 2.5 and 97.5)
        lower_pct = ((1.0 - ci_range) / 2) * 100
        upper_pct = (1.0 - (1.0 - ci_range) / 2) * 100

        return np.percentile(diffs, lower_pct), np.mean(diffs), np.percentile(diffs, upper_pct)

    for cat in ["$M_{ctrl}$", "$M_{post}$"]:
        low_ci, mean, high_ci = bootstrap_ci(model_steps, cat, "$M_a$") 
        print(f"Ma Advantage against {cat}: {mean:.2%} with 95% CI=[{low_ci:.2%}, {high_ci:.2%}]")

    # Panel 2: Swap layers effect on learning speed
    sns.barplot(
        replace_steps, x='dir', y='speedup', hue='layers',
        hue_order=layer_order, palette=layer_palette,
        order=list(dir_labels.keys()),
        edgecolor='white', linewidth=0.8,
        err_kws={'linewidth': 1.3, 'color': '#333333'}, capsize=0.08,
        ax=ax2,
    )
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_ylabel('Adoptee Advantage (%)')
    ax2.set_xlabel('$M_a$ Language Switch Direction')
    ax2.set_ylim(bottom=0, top=0.20)
    ax2.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax2.set_xticks(ax2.get_xticks())
    ax2.set_xticklabels([dir_labels[d.get_text()] for d in ax2.get_xticklabels()])
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)

    # Legend placed outside the axes so it never overlaps the bars,
    # regardless of which direction ends up tallest.
    ax2.legend(
        title='Layers\nspliced\n' r'$M_a \leftrightarrow M_{post}$',
        loc='upper left', bbox_to_anchor=(1.02, 1.02),
        frameon=False, borderaxespad=0, fontsize=8, title_fontsize=8,
    )

    for ax, label in zip([ax1, ax2], ['a', 'b']):
        ax.text(
            0.05, 1.06, label,
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            va='bottom', ha='right',
            #fontfamily='sans-serif'
        )

    plt.tight_layout()
    plt.savefig(FIGURE_FILENAME, dpi=300, bbox_inches='tight')
