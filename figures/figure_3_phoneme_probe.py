import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_FILE = "figure_3_phoneme_probe.csv"
FILENAME = "figure_3_phoneme_probe.png"

ALL_LANGS = ["de", "fr", "en"]
LANG_NAME = {"en": "English", "fr": "French", "de": "German"}


def third_lang(l1, l2):
    """The one language of the three that is neither l1 nor l2."""
    return [l for l in ALL_LANGS if l not in (l1, l2)][0]


# ── Panel A config: single example trajectory (birth=de, new=fr) ───────────────
EX_BIRTH = "de"
EX_NEW = "fr"
EX_THIRD = third_lang(EX_BIRTH, EX_NEW)  # "en"

EX_CEIL = EX_BIRTH                        # M_pre   : monolingual birth-lang model
EX_FLOOR = EX_NEW                         # M_post  : monolingual new-lang model
EX_ADOPT = f"{EX_BIRTH}2{EX_NEW}"         # M_a     : adoptee model
EX_CTRL = f"{EX_THIRD}2{EX_NEW}"          # M_ctrl  : control model (diff. birth lang, same new lang)

C = {
    EX_CEIL: "#f47a00",
    EX_FLOOR: "#62d8c3",
    EX_ADOPT: "#d31f11",
    EX_CTRL: "#007191",
}
LABELS = {
    EX_CEIL: "$M_{pre}$ (German)", #"$M_{pre}$",
    EX_FLOOR: "$M_{post}$ (French)", #$M_{post}$",
    EX_ADOPT: "$M_{a}$ (German\u2192French)", #"$M_a$",
    EX_CTRL: "$M_{ctrl}$ (English\u2192French)", #"$M_{ctrl}$",
}
C_LABEL = {LABELS[k]: v for k, v in C.items()}

# ── Panel B config: three language-pair splices ─────────────────────────────────
PAIRS = [("de", "fr"), ("fr", "de"), ("fr", "en"), ("en", "fr"), ("en", "de"), ("de", "en")]
#LAYER_BINS = [("Layers 1\u20134", 1, 4), ("Layers 5\u20138", 5, 8), ("Layers 9\u201312", 9, 12)]
LAYER_BINS = [("Pre-phonemic", 1, 4), ("Phonemic", 5, 8), ("Post-phonemic", 9, 12)]

B_COLORS = [C[EX_ADOPT], C[EX_CTRL]]
B_ORDER = ["$M_a$", "$M_{ctrl}$"]


def pair_label(l1, l2):
    return f"{LANG_NAME[l1]}\u2192{LANG_NAME[l2]}"


def layer_bin_name(layer):
    for name, lo, hi in LAYER_BINS:
        if lo <= layer <= hi:
            return name
    return None


if __name__ == "__main__":
    raw = pl.read_csv(CSV_FILE).filter(pl.col("layer") > 0)

    # ── Panel A data: filter to the example probe language and the 4 relevant models ──
    panelA_df = (
        raw.filter(pl.col("phon_lang") == EX_BIRTH)
        .filter(pl.col("model_lang").is_in([EX_CEIL, EX_FLOOR, EX_ADOPT, EX_CTRL]))
        .with_columns(
            lang=pl.col("model_lang").replace(LABELS),
            layer_bin=pl.col("layer").map_elements(layer_bin_name, return_dtype=pl.Utf8),
        )
    )
    LAYERS = sorted(panelA_df["layer"].unique().to_list())

    # Test for significant differences at each layer bin
    # Test is conducted over model seeds (4-6)
    print(f"\nAcross Ma:{EX_ADOPT} and Mctrl:{EX_CTRL} models, one-sided Mann-Whitney U test")
    print(f"asking whether the model contains more Lpre:{EX_BIRTH} phonemic knowledge than the Mpost:{EX_NEW}")
    layerwise_mean = (
        panelA_df.filter(pl.col("model_lang").is_in([EX_ADOPT, EX_CTRL, EX_NEW]))
        .group_by(["layer"]).agg(layer_mean=pl.col("accuracy").mean())
    )
    for bin_name in ["Pre-phonemic", "Phonemic", "Post-phonemic"]:
        bin_df = (
            panelA_df.filter(pl.col("layer_bin") == bin_name)
            .join(layerwise_mean, on="layer")
            .with_columns(norm_acc=pl.col("accuracy") - pl.col("layer_mean"))
            .group_by(["model_seed", "model_lang"]).agg(pl.col("norm_acc").mean())
        )
        madop_vals = bin_df.filter(pl.col("model_lang") == EX_ADOPT)
        mctrl_vals = bin_df.filter(pl.col("model_lang") == EX_CTRL)
        mpost_vals = bin_df.filter(pl.col("model_lang") == EX_NEW)
        w, pa = stats.mannwhitneyu(madop_vals["norm_acc"], mpost_vals["norm_acc"], alternative="greater")
        w, pc = stats.mannwhitneyu(mctrl_vals["norm_acc"], mpost_vals["norm_acc"], alternative="greater")

        print("Bin:", bin_name)
        print(f"Ma    p-value: {pa:.4g}, n: {len(madop_vals)}")
        print(f"Mctrl p-value: {pc:.4g}, n: {len(mctrl_vals)}")
        print()

    # ── Panel B data: build norm_acc for adoptee & control, for each of the 3 pairs ──
    bar_records = []
    for birth, new in PAIRS:
        third = third_lang(birth, new)
        adopt_model = f"{birth}2{new}"
        ctrl_model = f"{third}2{new}"

        sub = raw.filter(pl.col("phon_lang") == birth)

        baselines = (
            sub.with_columns(
                layer_bin=pl.col("layer").map_elements(layer_bin_name, return_dtype=pl.Utf8)
            )
            .group_by("layer_bin").agg(
                ceil_mean=pl.col("accuracy").filter(pl.col("model_lang") == birth).mean(),
                floor_mean=pl.col("accuracy").filter(pl.col("model_lang") == new).mean(),
            )
        )

        test = (
            sub.filter(pl.col("model_lang").is_in([adopt_model, ctrl_model]))
            .with_columns(
                pair=pl.lit(pair_label(birth, new)),
                layer_bin=pl.col("layer").map_elements(layer_bin_name, return_dtype=pl.Utf8),
                model_type=pl.when(pl.col("model_lang") == adopt_model)
                    .then(pl.lit("$M_a$"))
                    .otherwise(pl.lit("$M_{ctrl}$")),
            )
            .with_columns(
                group=pl.col("layer_bin") + " " + pl.col("model_type")
            )
            #.group_by(["pair", "group", "layer_bin", "model_type", "model_seed"]).agg(
            .group_by(["pair", "group", "layer_bin", "model_type"]).agg(
                mean_acc=pl.col("accuracy").mean()
            )
            .join(baselines, on="layer_bin")
            .with_columns(
                baseline_gap=(pl.col("ceil_mean") - pl.col("floor_mean"))
            )
            # Exclude any points that have extremely small gaps leading to very high variance
            #.filter(
            #    pl.col("baseline_gap") > 0.01
            #)
            .with_columns(
                norm_acc=(pl.col("mean_acc") - pl.col("floor_mean")) / pl.col("baseline_gap"),
            )
        )
        bar_records.append(test)
        #if adopt_model == "fr2en":
        #    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=1000):
        #        print(baselines.sort("layer_bin"))
        #        print(test.sort("group"))

    bar_df = pl.concat(bar_records)
    bin_order = [b[0] for b in LAYER_BINS]

    # Print one-sided Wilcoxon signed-rank test results for part b
    print("\nAcross language pairs, one-sided Wilcoxon signed-rank test results for each bin (Ma vs Mctrl)\n")
    for bin_name in ["Pre-phonemic", "Phonemic", "Post-phonemic"]:
        bin_df = bar_df.filter(pl.col("layer_bin") == bin_name)
        ma_vals = bin_df.filter(pl.col("model_type") == "$M_a$")
        mc_vals = bin_df.filter(pl.col("model_type") == "$M_{ctrl}$")
        w, p = stats.wilcoxon(ma_vals["norm_acc"], mc_vals["norm_acc"], alternative="greater")

        print("Bin:", bin_name)
        print("p-value:", p)
        print("n:", len(ma_vals))
        print()

    #
    # ── Figure ────────────────────────────────────────────────────────────────
    #
    fig, (ax_a, ax_b) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    plt.subplots_adjust(wspace=0.33)

    # PANEL A — Convergence accuracy (example trajectory: de -> fr)
    order = [LABELS[m] for m in [EX_CEIL, EX_FLOOR, EX_CTRL, EX_ADOPT]]
    sns.lineplot(
        data=panelA_df, x="layer", y="accuracy", hue="lang", hue_order=order,
        style="lang", ax=ax_a, errorbar="ci", palette=C_LABEL,
    )
    ax_a.set_xlabel("Layer index")
    ax_a.set_ylabel("German phoneme accuracy")
    ax_a.legend(loc="lower right", fontsize=9)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xticks(LAYERS)
    ax_a.tick_params(axis="x", labelsize=8)
    ax_a.set_ylim(top=0.81)

    # Zone shading to align Panel A with Panel B's layer bins
    ax_a.set_xlim(0.5, 12.5)
    ZONE_BOUNDS = [(0.5, 4.5, "Pre-phonemic"), (4.5, 8.5, "Phonemic"), (8.5, 12.5, "Post-phonemic")]
    ZONE_SHADE = ["#f2f2f2", "white", "#f2f2f2"]  # alternating bands

    for (lo, hi, name), shade in zip(ZONE_BOUNDS, ZONE_SHADE):
        ax_a.axvspan(lo, hi, color=shade, zorder=-10)
        ax_a.text(
            (lo + hi) / 2, 0.98, name,
            transform=ax_a.get_xaxis_transform(),  # x in data coords, y in axes fraction
            fontsize=9, ha="center", va="top", color="#555555", style="italic",
        )

    for lo, hi, _ in ZONE_BOUNDS[:-1]:
        ax_a.axvline(hi, color="#CCCCCC", lw=0.6, ls="--", zorder=-5)


    # ── PANEL B — Paired Slope Plot ───────────────────────────────────────────

    # Half-width offset between M_ctrl and M_a within each category
    x_map = {name: i for i, name in enumerate(bin_order)}
    dx = 0.2

    # Iterate through each unique language pair to draw connecting lines
    for pair_name in bar_df["pair"].unique().to_list():
        pair_data = bar_df.filter(pl.col("pair") == pair_name)

        for b_name in bin_order:
            bin_data = pair_data.filter(pl.col("layer_bin") == b_name)

            # Extract values for M_a and M_ctrl
            m_a_val = bin_data.filter(pl.col("model_type") == "$M_a$")["norm_acc"].to_list()
            m_ctrl_val = bin_data.filter(pl.col("model_type") == "$M_{ctrl}$")["norm_acc"].to_list()

            if b_name == "Pre-phonemic":
                pair_name_short = pair_name.replace("French", "fr").replace("English", "en").replace("German", "de")
                ax_b.text(
                    x=x_map[b_name] - dx - 0.1, 
                    y=m_a_val[0],
                    s=pair_name_short,
                    ha="right",
                    va="center",
                    color="grey",
                    fontsize=8,
                )


            if m_a_val and m_ctrl_val:
                x_center = x_map[b_name]
                x_ctrl = x_center + dx
                x_a = x_center - dx

                # Draw gray connecting line for the pair
                ax_b.plot([x_ctrl, x_a], [m_ctrl_val[0], m_a_val[0]],
                        color="#aaaaaa", lw=1.2, zorder=1)

                # Draw dots on top
                ax_b.scatter(x_ctrl, m_ctrl_val[0], color=C[EX_CTRL], s=40, zorder=2)
                ax_b.scatter(x_a, m_a_val[0], color=C[EX_ADOPT], s=40, zorder=2)

    # Annotate the weirdly high control points
    oval = mpatches.Ellipse(xy=(0.2, 0.275), width=0.15, height = 0.08, edgecolor="black", lw=1, zorder=3, fc='None')
    ax_b.add_patch(oval)
    ax_b.annotate(text="$M_a$ and $M_{ctrl}$\nadopted from\nthe same\nlanguage family", xy=(0.21, 0.28), xytext=(0.5, 0.29),
        color='black', va="top", arrowprops=dict(arrowstyle='-|>', color='black', lw=2), fontsize=8)


    ax_b.set_xticks(range(len(bin_order)))
    ax_b.set_xticklabels(bin_order, fontsize=9)
    ax_b.set_xlim(-0.8, len(bin_order) - 0.5)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C[EX_ADOPT], markersize=8, label="$M_a$"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C[EX_CTRL], markersize=8, label="$M_{ctrl}$")
    ]
    ax_b.legend(handles=legend_handles, loc="upper right", fontsize=9)

    #ax_b.set_xlabel("Layer bin")
    ax_b.set_ylabel("Rel. $L_{pre}$ phoneme accuracy")
    ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Panel labels
    for ax, label in zip([ax_a, ax_b], ["a", "b"]):
        ax.text(
            0.05, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="right", fontfamily="sans-serif",
        )

    fig.savefig(FILENAME, dpi=150, bbox_inches="tight")
    print(f"Saved {FILENAME}")
