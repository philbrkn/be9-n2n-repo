"""
Skeleton geometry figure for the Be multiplicity assembly.
Two panels: (a) XZ cross-section, (b) XY cross-section showing He-3 tubes.
Hatching patterns for grayscale compatibility.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,  # better for multi-panel
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "figure.dpi": 300,
    }
)

# --- Geometry parameters (cm) ---
BE_RADIUS = 9.0
BE_HALF_HEIGHT = 32.5
BEAM_HOLE_RADIUS = 1.0
CD_THICKNESS = 0.1
CD_RADIUS = BE_RADIUS + CD_THICKNESS

N_TUBES = 20
HE3_RADIUS = 1.5
HE3_RADIAL_POS = 15.0

OUTER_RADIUS = 20.0
OUTER_HALF_HEIGHT = 40.0

# --- Figure setup ---
fig, (ax_xz, ax_xy) = plt.subplots(1, 2, figsize=(6.3, 5), constrained_layout=True)


# ============================================================
# (a) XZ cross-section (side view, y=0 slice)
# ============================================================
# Convention: x-axis = radial (r), y-axis = z

# HDPE (background fill for the whole assembly)
hdpe = patches.Rectangle(
    (-OUTER_RADIUS, -OUTER_HALF_HEIGHT),
    2 * OUTER_RADIUS,
    2 * OUTER_HALF_HEIGHT,
    linewidth=1.2,
    edgecolor="k",
    facecolor="#d9d9d9",
    label="HDPE",
)
ax_xz.add_patch(hdpe)

# Cadmium shell (radial only, full height)
cd_left = patches.Rectangle(
    (-CD_RADIUS, -OUTER_HALF_HEIGHT),
    CD_THICKNESS,
    2 * OUTER_HALF_HEIGHT,
    linewidth=0.5,
    edgecolor="k",
    facecolor="k",
    label="Cd",
)
cd_right = patches.Rectangle(
    (BE_RADIUS, -OUTER_HALF_HEIGHT),
    CD_THICKNESS,
    2 * OUTER_HALF_HEIGHT,
    linewidth=0.5,
    edgecolor="k",
    facecolor="k",
)
ax_xz.add_patch(cd_left)
ax_xz.add_patch(cd_right)

# Beryllium (with central hole)
be_left = patches.Rectangle(
    (-BE_RADIUS, -BE_HALF_HEIGHT),
    BE_RADIUS - BEAM_HOLE_RADIUS,
    2 * BE_HALF_HEIGHT,
    linewidth=0.8,
    edgecolor="k",
    facecolor="white",
    hatch="///",
    label="Be",
)
be_right = patches.Rectangle(
    (BEAM_HOLE_RADIUS, -BE_HALF_HEIGHT),
    BE_RADIUS - BEAM_HOLE_RADIUS,
    2 * BE_HALF_HEIGHT,
    linewidth=0.8,
    edgecolor="k",
    facecolor="white",
    hatch="///",
)
ax_xz.add_patch(be_left)
ax_xz.add_patch(be_right)

# Beam hole (void/air)
beam = patches.Rectangle(
    (-BEAM_HOLE_RADIUS, -OUTER_HALF_HEIGHT),
    2 * BEAM_HOLE_RADIUS,
    2 * OUTER_HALF_HEIGHT,
    linewidth=0.5,
    edgecolor="k",
    facecolor="white",
    label="Void",
)
ax_xz.add_patch(beam)

# Air cavities above/below Be (inside the Cd shell)
air_top = patches.Rectangle(
    (-BE_RADIUS, BE_HALF_HEIGHT),
    2 * BE_RADIUS,
    OUTER_HALF_HEIGHT - BE_HALF_HEIGHT,
    linewidth=0.3,
    edgecolor="grey",
    facecolor="white",
    linestyle="--",
)
air_bot = patches.Rectangle(
    (-BE_RADIUS, -OUTER_HALF_HEIGHT),
    2 * BE_RADIUS,
    OUTER_HALF_HEIGHT - BE_HALF_HEIGHT,
    linewidth=0.3,
    edgecolor="grey",
    facecolor="white",
    linestyle="--",
)
ax_xz.add_patch(air_top)
ax_xz.add_patch(air_bot)

# He-3 tubes in XZ view (show the two tubes at y~0, i.e. at x = +/- HE3_RADIAL_POS)
for sign in [-1, 1]:
    x_center = sign * HE3_RADIAL_POS
    he3 = patches.Rectangle(
        (x_center - HE3_RADIUS, -OUTER_HALF_HEIGHT),
        2 * HE3_RADIUS,
        2 * OUTER_HALF_HEIGHT,
        linewidth=0.8,
        edgecolor="k",
        facecolor="white",
        hatch="ooo",
        label="He-3" if sign == -1 else None,
    )
    ax_xz.add_patch(he3)

# Source marker
ax_xz.plot(
    0,
    0,
    marker="*",
    color="red",
    markersize=10,
    zorder=5,
    linestyle="none",
    label="DT source",
)


# --- Dimension lines (XZ) ---
def dim_line_h(ax, y, x0, x1, text, offset=2):
    """Horizontal dimension line."""
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", color="k", lw=0.8),
    )
    ax.text((x0 + x1) / 2, y + offset, text, ha="center", va="bottom", fontsize=7)


def dim_line_v(ax, x, y0, y1, text, offset=2):
    """Vertical dimension line."""
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops=dict(arrowstyle="<->", color="k", lw=0.8),
    )
    ax.text(
        x + offset, (y0 + y1) / 2, text, ha="left", va="center", fontsize=7, rotation=90
    )


# Be radius
# dim_line_h(ax_xz, BE_HALF_HEIGHT + 3, 0, BE_RADIUS, f"Be R={BE_RADIUS}")
#
# # Outer radius
# dim_line_h(
#     ax_xz, -BE_HALF_HEIGHT - 9, 0, OUTER_RADIUS, f"Outer R={OUTER_RADIUS}", offset=-4
# )

# Be height
# dim_line_v(
#     ax_xz,
#     -OUTER_RADIUS - 5,
#     -BE_HALF_HEIGHT,
#     BE_HALF_HEIGHT,
#     f"Be H={2 * BE_HALF_HEIGHT}",
# )
#
# # Outer height
# dim_line_v(
#     ax_xz,
#     OUTER_RADIUS + 2,
#     -OUTER_HALF_HEIGHT,
#     OUTER_HALF_HEIGHT,
#     f"H={2 * OUTER_HALF_HEIGHT}",
# )

ax_xz.set_xlim(-OUTER_RADIUS - 8, OUTER_RADIUS + 8)
ax_xz.set_ylim(-OUTER_HALF_HEIGHT - 10, OUTER_HALF_HEIGHT + 10)
ax_xz.set_aspect("equal")
ax_xz.set_xlabel("x (cm)", fontsize=9)
ax_xz.set_ylabel("z (cm)", fontsize=9)

# Store the XZ y-limits so XY can match scale
xz_ylim = ax_xz.get_ylim()
xz_range = xz_ylim[1] - xz_ylim[0]

# ============================================================
# (b) XY cross-section (top view, z=0 slice)
# ============================================================

# HDPE background
hdpe_xy = patches.Circle(
    (0, 0),
    OUTER_RADIUS,
    linewidth=1.2,
    edgecolor="k",
    facecolor="#d9d9d9",
    label="HDPE",
)
ax_xy.add_patch(hdpe_xy)

# Cadmium shell
cd_xy = patches.Circle(
    (0, 0), CD_RADIUS, linewidth=0.8, edgecolor="k", facecolor="k", label="Cd"
)
ax_xy.add_patch(cd_xy)

# Beryllium
be_xy = patches.Circle(
    (0, 0),
    BE_RADIUS,
    linewidth=0.8,
    edgecolor="k",
    facecolor="white",
    hatch="///",
    label="Be",
)
ax_xy.add_patch(be_xy)

# Beam hole
beam_xy = patches.Circle(
    (0, 0),
    BEAM_HOLE_RADIUS,
    linewidth=0.5,
    edgecolor="k",
    facecolor="white",
    label="Void",
)
ax_xy.add_patch(beam_xy)

# He-3 tubes
for i in range(N_TUBES):
    theta = 2 * np.pi * i / N_TUBES
    x = HE3_RADIAL_POS * np.cos(theta)
    y = HE3_RADIAL_POS * np.sin(theta)
    he3 = patches.Circle(
        (x, y),
        HE3_RADIUS,
        linewidth=0.8,
        edgecolor="k",
        facecolor="white",
        hatch="ooo",
        label="He-3" if i == 0 else None,
    )
    ax_xy.add_patch(he3)

# Source marker
ax_xy.plot(
    0,
    0,
    marker="*",
    color="red",
    markersize=10,
    zorder=5,
    label="DT source",
    linestyle="none",
)

# Dimension: He-3 radial position
# ax_xy.annotate(
#     "",
#     xy=(HE3_RADIAL_POS, 0),
#     xytext=(0, 0),
#     arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
# )
# ax_xy.text(
#     HE3_RADIAL_POS / 2,
#     2,
#     f"r={HE3_RADIAL_POS}",
#     ha="center",
#     fontsize=7,
#     bbox=dict(facecolor="white", alpha=0.9),
# )

ax_xy.set_xlim(-OUTER_RADIUS - 8, OUTER_RADIUS + 8)
ax_xy.set_ylim(xz_ylim[0], xz_ylim[1])  # same vertical range as XZ
ax_xy.set_aspect("equal")
ax_xy.set_xlabel("x (cm)", fontsize=9)
ax_xy.set_ylabel("y (cm)", fontsize=9)

# --- Legend (shared) ---
handles, labels = ax_xz.get_legend_handles_labels()
h2, l2 = ax_xy.get_legend_handles_labels()
# Deduplicate
seen = set()
unique_handles, unique_labels = [], []
for h, l in list(zip(handles, labels)) + list(zip(h2, l2)):
    if l not in seen:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)

# fig.legend(
#     unique_handles,
#     unique_labels,
#     loc="lower center",
#     ncol=6,
#     fontsize=8,
#     frameon=True,
#     bbox_to_anchor=(0.5, -0.02),
# )

ax_xy.legend(
    handles=unique_handles,
    labels=unique_labels,
    loc="upper right",
    fontsize=8,
    framealpha=0.95,
    fancybox=False,
    edgecolor="gray",
)
ax_xz.text(
    0.5, -0.12, "(a)", transform=ax_xz.transAxes, ha="center", va="top", fontsize=9
)
ax_xy.text(
    0.5, -0.12, "(b)", transform=ax_xy.transAxes, ha="center", va="top", fontsize=9
)


# plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("geometries/geometry_figure.pdf", dpi=300, bbox_inches="tight")
plt.savefig("geometries/geometry_figure.png", dpi=300, bbox_inches="tight")
print("Saved geometry_figure.pdf and geometry_figure.png")
