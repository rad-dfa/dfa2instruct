import dill
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dfax import DataSampler, list2batch, batch2graph
from rad_embeddings import Encoder

# Increase font sizes globally.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 50,
})

dataset = {}
dfax_list = []
with open("datasets/dataset_R_n139_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())

with open("datasets/dataset_RA_n479_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())

with open("datasets/dataset_RAD_n102_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())

dfax = dfax_list[0]
encoder = Encoder(
    max_size=dfax.max_n_states,
    n_tokens=dfax.n_tokens,
    seed=0
)

rad = [encoder(dfax.to_graph()).squeeze() for dfax in dfax_list]
# rad = encoder(batch2graph(list2batch([dfax.expand(5).to_graph() for dfax in dfax_list])))
emb = [jnp.array(dataset[dfax][1]) for dfax in dfax_list]

N = 100

def distance(feat1, feat2):
    feat1 = feat1 / jnp.linalg.norm(feat1, ord=2, axis=-1, keepdims=True)
    feat2 = feat2 / jnp.linalg.norm(feat2, ord=2, axis=-1, keepdims=True)
    d = jnp.linalg.norm(feat1 - feat2, ord=2, axis=-1)
    return float(d)

# Compute the normalized L2 distance matrix between all encoder outputs.
def compute_distance_matrix(embeddings):
    n = len(embeddings)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = distance(embeddings[i], embeddings[j])
    return dists


rad_distance_matrix = compute_distance_matrix(rad)
emb_distance_matrix = compute_distance_matrix(emb)

# Find global vmin/vmax for consistent color scale
vmin = min(rad_distance_matrix.min(), emb_distance_matrix.min())
vmax = max(rad_distance_matrix.max(), emb_distance_matrix.max())

from matplotlib import gridspec
fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
axes = [ax0, ax1]

# Plot RAD heatmap
ax0 = axes[0]
sns.heatmap(rad_distance_matrix, cmap="rocket_r", ax=ax0, vmin=vmin, vmax=vmax, cbar=False)
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
rad_logo_img = mpimg.imread("rad.png")
ax0.set_title("")
ax0.set_xlabel("")
ax0.set_ylabel("")
rad_imagebox = OffsetImage(rad_logo_img, zoom=0.4)  # Adjust zoom as needed
rad_ab = AnnotationBbox(rad_imagebox, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
ax0.add_artist(rad_ab)

# Plot Embedding heatmap
hm = sns.heatmap(emb_distance_matrix, cmap="rocket_r", ax=ax1, vmin=vmin, vmax=vmax, cbar=False)
import matplotlib.image as mpimg
emb_img = mpimg.imread("qwen3.png")
# Remove the text title
ax1.set_title("")
ax1.set_xlabel("")
ax1.set_ylabel("")
# Add the logo image above the heatmap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
imagebox = OffsetImage(emb_img, zoom=0.4)  # Adjust zoom as needed
ab = AnnotationBbox(imagebox, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
ax1.add_artist(ab)

# Calculate tick positions as the midpoints of each group.
tick_positions = [ (N - 1) / 2,
                   N + (N - 1) / 2,
                   2 * N + (N - 1) / 2]
tick_labels = [r"$\mathtt{R}$", r"$\mathtt{RA}$", r"$\mathtt{RAD}$"]

for ax in axes:
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=90)
    ax.set_aspect('equal')


# Place a single colorbar further right using inset axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.5)
cbar = fig.colorbar(hm.collections[0], cax=cax, orientation='vertical')
cbar.set_label('Distance', rotation=270, labelpad=50)

plt.savefig("heatmap.png", bbox_inches="tight", pad_inches=0.1)
plt.close()
