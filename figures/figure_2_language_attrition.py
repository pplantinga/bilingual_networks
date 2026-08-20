import pathlib, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D


COLORS = {"ma": "#d31f11", "mpost": "#62d8c3"}
RENAME = {"FR": "$M_a$ ($L_{post}$)", "DE": "$M_a$ ($L_{pre}$)"}

def unpivot_and_split(df):
    df = df.unpivot(index="Steps", value_name="Accuracy")
    df = (
        df.with_columns(
            pl.col("variable").str.split_exact("-", 1)
        )
        .unnest("variable")
        .rename({"field_0": "lang", "field_1": "seed"})
        .with_columns(
            pl.col("lang").replace(RENAME)
        )
    )
    return df

def k_formatter(x, pos, n=0):
    if x == 0:
        return 0
    if n == 0:
        return str(int(x // 1000)) + "k"

if __name__ == "__main__":
    # Reorient two csvs from wide- to long-form
    attrition_df = unpivot_and_split(pl.read_csv("figure_2_left_panel.csv"))
    matching_df = unpivot_and_split(pl.read_csv("figure_2_right_panel.csv"))

    # Setup plotting
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9,3), sharey=True, gridspec_kw={'wspace': 0.05})

    # Part a: Adoptee Language Attrition
    ####################
    # compute last step accuracy
    last_step = max(attrition_df["Steps"])
    final_de_acc = (
        attrition_df
        .filter((pl.col("Steps") == last_step) & (pl.col("lang") == "DE"))
        .select(pl.col("Accuracy").mean())
        .item()
    )

    # Plot lines
    palette = {v: COLORS["ma"] for v in RENAME.values()}
    sns.lineplot(attrition_df, x="Steps", y="Accuracy", style="lang", hue="lang", palette=palette,
        style_order=RENAME.values(), hue_order=RENAME.values(), dashes=[(), (4, 2)], errorbar="ci", lw=2, ax=ax_a)
    ax_a.axhline(0.254, color="grey", linestyle="dotted", alpha=0.8, label="Random")

    # Chart stuff
    ax_a.set_xlabel("Number of updates on $L_{post}$ (after switch)", labelpad=1)
    ax_a.set_ylabel("Next-Token Accuracy")
    ax_a.legend(title=None, handlelength=2.0)
    ax_a.set_xlim(left=0, right=4_000)
    ax_a.set_ylim(top=1.0)
    ax_a.grid(True, alpha=0.3)
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))
    ax_a.set_xticks([0, 1_000, 2_000, 3_000, 4_000])

    # Part b: Matching Monolingual Performance
    ######################
    switch_update = 5 * 1680

    # compute last step accuracy
    last_step = max(matching_df["Steps"])
    final_fr_acc = (
        matching_df
        .filter((pl.col("Steps") == last_step) & (pl.col("lang") == "fr"))
        .select(pl.col("Accuracy").mean())
        .item()
    )
    final_de2fr_acc = (
        matching_df
        .filter((pl.col("Steps") == last_step) & (pl.col("lang") == "de2fr"))
        .select(pl.col("Accuracy").mean())
        .item()
    )
    sns.lineplot(matching_df, x="Steps", y="Accuracy", hue="lang", hue_order=['de2fr', 'fr'],
        palette=[COLORS['ma'], COLORS['mpost']], errorbar="ci", lw=2, ax=ax_b)
    ax_b.axvline(switch_update, color="grey", linestyle="dashed", alpha=0.8, label="Language switch")
    ax_b.set_xlabel("Number of updates since initialization")

    # Get all handles and filter to only Line2D objects + the axvline
    handles, _ = ax_b.get_legend_handles_labels()
    line_handles = [h for h in handles if isinstance(h, Line2D)]
    labels = ["$M_a$ ($L_{post}$)", "$M_{post}$ ($L_{post}$)", "Switch"]
    ax_b.legend(line_handles, labels, loc="best", title=None, handlelength=2.0)#, fontsize=11)
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(k_formatter))
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(0, 40_000)
    ax_b.set_xticks([0, 10_000, 20_000, 30_000, 40_000])

    for ax, label in zip([ax_a, ax_b], ['a', 'b']):
        ax.text(
            0.05, 1.06, label,
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            va='bottom', ha='right',
            fontfamily='sans-serif'
        )

    fig.savefig("figure_2_attrition.png", dpi=200, bbox_inches="tight")
