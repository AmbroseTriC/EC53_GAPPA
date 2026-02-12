import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe

# ---------- Load ----------
df = pd.read_csv("/Users/ambrosewang/Library/CloudStorage/OneDrive-UniversityCollegeLondon/ECON0053 Economics of Tax Policy/Grp Project/Quantitative/Analysis/v4/ggppa_kakwani_indices_v4.csv")

# If your "indirect" measure has a different label, change it here:
DIRECT = "direct_tax"
INDIRECT = "gross_cost"   # <-- replace with "indirect_tax" if that’s your column label
COMBINED = "tax"

# ---------- Combine direct + indirect (identical by construction) ----------
measures = set(df["Measure"].dropna().unique())
if {DIRECT, INDIRECT}.issubset(measures):
    a = df[df["Measure"] == DIRECT][["Province", "Year", "Kakwani"]].rename(columns={"Kakwani": "k1"})
    b = df[df["Measure"] == INDIRECT][["Province", "Year", "Kakwani"]].rename(columns={"Kakwani": "k2"})
    m = a.merge(b, on=["Province", "Year"], how="inner")

    # If identical (or numerically indistinguishable), keep one and relabel.
    if np.allclose(m["k1"].to_numpy(), m["k2"].to_numpy(), atol=1e-10, rtol=0.0):
        df = df[df["Measure"] != INDIRECT].copy()
        df.loc[df["Measure"] == DIRECT, "Measure"] = COMBINED
    else:
        # Fallback: combine by averaging (should not trigger in your case).
        comb = (df[df["Measure"].isin([DIRECT, INDIRECT])]
                .groupby(["Province", "Year"], as_index=False)["Kakwani"].mean())
        comb["Measure"] = COMBINED
        df = df[~df["Measure"].isin([DIRECT, INDIRECT])]
        df = pd.concat([df, comb], ignore_index=True)

# ---------- Orders / labels ----------
province_order = sorted(df["Province"].dropna().unique())
year_order = sorted(df["Year"].dropna().unique())

measure_order = []
if COMBINED in df["Measure"].unique():
    measure_order.append(COMBINED)
# keep other measures (e.g., net_single_base)
for m in sorted(df["Measure"].unique()):
    if m not in measure_order:
        measure_order.append(m)

measure_labels = {
    COMBINED: "Tax (direct = indirect)",
    "net_single_base": "Net single base",
    "gross_cost": "Gross cost",
    "direct_tax": "Direct tax",
}

# ---------- Styling ----------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

absmax = np.nanmax(np.abs(df["Kakwani"].values))
absmax = np.ceil(absmax * 10) / 10
norm = mpl.colors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

try:
    cmap = mpl.colormaps["RdBu_r"]
except Exception:
    cmap = mpl.cm.get_cmap("RdBu_r")
cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
cmap.set_bad(color="0.95")

# ---------- Plot (now fewer facets) ----------
fig, axes = plt.subplots(
    1, len(measure_order),
    figsize=(5.8 * len(measure_order) + 1.2, 4.2),
    constrained_layout=True
)
axes = np.atleast_1d(axes)

im = None
for ax, measure in zip(axes, measure_order):
    pivot = (df[df["Measure"] == measure]
             .pivot(index="Province", columns="Year", values="Kakwani")
             .reindex(index=province_order, columns=year_order))

    data = np.ma.masked_invalid(pivot.values.astype(float))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(measure_labels.get(measure, measure))
    ax.set_xlabel("Year")

    ax.set_xticks(np.arange(len(year_order)))
    ax.set_xticklabels(year_order)

    ax.set_yticks(np.arange(len(province_order)))
    if ax is axes[0]:
        ax.set_ylabel("Province")
        ax.set_yticklabels(province_order)
    else:
        ax.set_yticklabels([])

    # tile gridlines
    ax.set_xticks(np.arange(-0.5, len(year_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(province_order), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # contrast-adaptive annotations with subtle outline
    for i in range(len(province_order)):
        for j in range(len(year_order)):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue

            rgba = cmap(norm(val))
            luma = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "black" if luma > 0.62 else "white"
            outline = (1, 1, 1, 0.30) if txt_color == "black" else (0, 0, 0, 0.30)

            t = ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color)
            t.set_path_effects([pe.withStroke(linewidth=0.9, foreground=outline)])

# shared colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Kakwani index (centered at 0)")

fig.suptitle("Kakwani index by province, year, and program component", y=1.05)

fig.savefig("kakwani_heatmap_facets_combined.png", dpi=300, bbox_inches="tight")
fig.savefig("kakwani_heatmap_facets_combined.pdf", bbox_inches="tight")
plt.show()
