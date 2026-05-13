import pathlib, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import polars as pl


if __name__ == "__main__":
    attrition_df = pl.read_csv("figure_1_left_panel.csv")
    attrition_df = attrition_df.with_columns(
        pl.col("French Acc").str.strip_chars("%").cast(pl.Float32),
        pl.col("German Acc").str.strip_chars("%").cast(pl.Float32),
    ).drop(["Monolingual", "", "_duplicated_0"])

    matching_df = pl.read_csv("figure_1_right_panel.csv")
    matching_df = matching_df.with_columns(
        Updates=pl.col("Epoch") * 1859
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,2), sharey=True, gridspec_kw={'wspace': 0.05})

    # Adoptee Language Attrition
    final_fr_acc = attrition_df["French Acc"][-1]
    ax1.plot(attrition_df["Updates"], attrition_df["French Acc"], label="$M_A$ accuracy on $L_{pre}$", color="steelblue", lw=2)
    ax1.plot(attrition_df["Updates"], attrition_df["German Acc"], label="$M_A$ accuracy on $L_{post}$", color="firebrick", lw=2)
    ax1.annotate(f"{final_fr_acc:.1f}%", xy=(attrition_df["Updates"][-1], final_fr_acc), xytext=(11000, 10), color="steelblue", weight="bold")

    ax1.set_xlabel("Number of updates on $L_{post}$ (German)", labelpad=1)
    ax1.set_ylabel("Next-Token Accuracy")
    ax1.legend(loc="upper right", handlelength=1.5)
    ax1.set_xscale("log")
    ax1.set_xlim(left=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    def k_formatter(x, pos):
        if x == 0:
            return 0
        return str(int(x // 1000)) + "k"

    # Matching Monolingual Performance
    final_de_acc = matching_df["Fr5,De20"][-1]
    switch_update = 5 * 1859

    ax2.plot(matching_df["Updates"], matching_df["De-mono"], label="$M_{post}$ accuracy on $L_{post}$", color="goldenrod", lw=2)
    ax2.plot(matching_df["Updates"], matching_df["Fr5,De20"], label="$M_A$ accuracy on $L_{post}$", color="firebrick", lw=2)
    ax2.axvline(switch_update, color="grey", linestyle="dashed", alpha=0.8, label="Language switch")
    ax2.annotate(f"{final_de_acc:.1f}%", xy=(matching_df["Updates"][-1], final_de_acc), xytext=(32_000, 82), color="firebrick", weight="bold")

    ax2.set_xlabel("Number of updates since initialization")
    ax2.legend(handlelength=1.5)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    ax2.grid(True, alpha=0.3)

    fig.savefig("figure_1_attrition.png", dpi=200, bbox_inches="tight")
