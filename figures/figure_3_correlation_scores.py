import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('figure_3_correlation_scores.csv')

# Define model groups
de_models = ['de25', 'de25_take2', 'de25_take3', 'de25_take4']
fr_models = ['fr25', 'fr25_take2', 'fr25_take4']
adoptee_models = ['de5_fr25', 'de5_fr25_take2', 'de5_fr25_take3', 'de5_fr25_take4']
last_ckpts = [24, 25]
a_pre_label = "RSA($M_A$, $M_{pre=de}$)"
a_post_label = "RSA($M_A$, $M_{post=fr}$)"
topline_label = r"RSA($M_{post=fr}^{\neq seed}$, $M_{post=fr}^{\neq seed}$)"
baseline_label = "RSA($M_{pre=de}$, $M_{post=fr}$)"

def categorize_v2(row):
    la, lb = row['lang_a'], row['lang_b']
    # Adoptee vs Birth
    if (la in adoptee_models) and (lb in de_models):
        return a_pre_label
    # Adoptee vs Rearing
    if (la in adoptee_models) and (lb in fr_models):
        return a_post_label
    # Topline: FR vs FR (different seeds)
    if (la in fr_models) and (lb in fr_models) and (la != lb):
        return topline_label
    # Baseline: DE vs FR
    if ((la in de_models) and (lb in fr_models)) or ((la in fr_models) and (lb in de_models)):
        return baseline_label
    return 'Other'

df['category'] = df.apply(categorize_v2, axis=1)

# Filter relevant data
df_traj = df[df['category'].isin([a_pre_label, a_post_label]) & (df['epoch_b'].isin(last_ckpts)) & df['layer'].isin([1, 2, 3, 4])]
df_base = df[df['category'].isin([topline_label, baseline_label]) & (df['epoch_a'].isin(last_ckpts)) & (df['epoch_b'].isin(last_ckpts))]
df_layer = df[df['category'].isin([a_pre_label, a_post_label]) & (df['epoch_b'].isin(last_ckpts) & df['epoch_a'].isin(last_ckpts))]

# Prepare baseline stats per layer (unstacking makes it easier to work with)
baseline_stats = df_base.groupby(['category', 'layer'])['score'].mean().unstack('category')
baseline_stats.columns = ['top', 'bottom']
baseline_stats = baseline_stats.reset_index()

# Figure
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
palette = ["orange", "green"]
hue_order = [a_post_label, a_pre_label]

# Panel A: Trajectory (Mean over layers, Std Dev over repetitions)
# We want to see how the Adoptee evolves.
global_baselines = df_base.groupby('category')['score'].mean()

# Scale all layers to the same 0-1 range, using Topline and Baseline to compute the range
traj_data = df_traj.groupby(['category', 'layer', 'epoch_a', 'lang_a'])['score'].mean().reset_index()
traj_scaled = traj_data.merge(baseline_stats, on='layer')
traj_scaled['score_normalized'] = (traj_scaled['score'] - traj_scaled['bottom']) / (traj_scaled['top'] - traj_scaled['bottom'])
traj_avg = traj_scaled.groupby(['category', 'epoch_a', 'lang_a'])['score_normalized'].mean().reset_index()

#sns.lineplot(data=traj_scaled, x='epoch_a', y='score_normalized', hue='layer', color='#888888', linewidth=1.5, alpha=0.5, ax=axes[0])
for layer in range(12):
    data_subset = traj_scaled[(traj_scaled['category'] == a_pre_label) & (traj_scaled['layer'] == layer)]
    sns.lineplot(data_subset, x='epoch_a', y='score_normalized', errorbar=None,
            color='lightgrey', alpha=0.4, linewidth=1, zorder=1, ax=axes[0])
sns.lineplot(data=traj_avg, x='epoch_a', y='score_normalized', hue='category', ax=axes[0], 
             errorbar='ci', palette=palette, linewidth=2.5, hue_order=hue_order)
axes[0].axhline(1, color='#555555', linestyle='--', alpha=0.8)#, label='Topline (FR vs FR)')
axes[0].axhline(0, color='#555555', linestyle=':', alpha=0.8)#, label='Baseline (DE vs FR)')
axes[0].text(25, 0.98, topline_label, color='#555555', alpha=0.8, va="top", ha="right")
axes[0].text(25, -0.02, baseline_label, color='#555555', alpha=0.8, va="top", ha="right")

#axes[0].set_title('A. Representational Trajectory during Adoption\n(Mean over layers, SD across replicates)', fontsize=13)
axes[0].set_xlabel('Epochs of French Training', fontsize=11)
axes[0].set_ylabel('Relative Similarity', fontsize=11, labelpad=0)
axes[0].legend(frameon=True, loc='upper left', handlelength=1.0)
axes[0].grid(True, alpha=0.2)
axes[0].set_ylim(bottom=-0.2, top=1.2)

# Panel B: Layer-wise (at end of training, last 3 checkpoints)
layer_data = df_layer.groupby(['category', 'layer', 'lang_a'])['score'].mean().reset_index()

sns.lineplot(data=layer_data, x='layer', y='score', hue='category', ax=axes[1], 
             errorbar='ci', palette=palette, linewidth=2.5, marker="o", hue_order=hue_order)

# Layer-wise baselines
axes[1].plot(baseline_stats['layer'], baseline_stats['top'], color='#555555', linestyle='--', alpha=0.8)#, label='Topline (FR vs FR)', alpha=0.8)
axes[1].plot(baseline_stats['layer'], baseline_stats['bottom'], color='#555555', linestyle=':', alpha=0.8)# label='Baseline (DE vs FR)', alpha=0.8)

#axes[1].set_title('B. Layer-wise Similarity at Convergence\n(Epoch 25, SD across replicates)', fontsize=13)
axes[1].set_xlabel('Layer Index', fontsize=11)
axes[1].set_ylabel('Representational Similarity', fontsize=11)
axes[1].set_xticks(range(12))
axes[1].legend(frameon=True)
axes[1].grid(True, alpha=0.2)
axes[1].set_ylim(bottom=0.4, top=1.0)

plt.tight_layout()
plt.savefig('figure_3_correlation_analysis.png')
