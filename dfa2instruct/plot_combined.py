import dill
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from rad_embeddings import Encoder
from dfax import list2batch, batch2graph

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 40,
})

dataset = {}
dfax_list = []
labels = []
with open("datasets/dataset_R_n139_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())
    labels.extend([0] * len(data))
with open("datasets/dataset_RA_n479_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())
    labels.extend([1] * len(data))
with open("datasets/dataset_RAD_n102_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())
    labels.extend([2] * len(data))

labels = np.array(labels)
dfax = dfax_list[0]
encoder = Encoder(
    max_size=dfax.max_n_states,
    n_tokens=dfax.n_tokens,
    seed=0
)
rad = [encoder(dfax.to_graph()).squeeze() for dfax in dfax_list]
emb = [jnp.array(dataset[dfax][1]) for dfax in dfax_list]
rad = np.stack([np.asarray(x) for x in rad])
emb = np.stack([np.asarray(x) for x in emb])

# t-SNE projections
rad_tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(rad)
emb_tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(emb)

# Distance matrices
N = 100
def distance(feat1, feat2):
    feat1 = feat1 / jnp.linalg.norm(feat1, ord=2, axis=-1, keepdims=True)
    feat2 = feat2 / jnp.linalg.norm(feat2, ord=2, axis=-1, keepdims=True)
    d = jnp.linalg.norm(feat1 - feat2, ord=2, axis=-1)
    return float(d)
def compute_distance_matrix(embeddings):
    n = len(embeddings)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = distance(embeddings[i], embeddings[j])
    return dists
rad_distance_matrix = compute_distance_matrix(rad)
emb_distance_matrix = compute_distance_matrix(emb)
vmin = min(rad_distance_matrix.min(), emb_distance_matrix.min())
vmax = max(rad_distance_matrix.max(), emb_distance_matrix.max())

# Plotting
colors = ['#d55f29', '#624aea', '#2ca02c']
label_names = [r"$\mathtt{Reach}$ ($\mathtt{R}$)", r"$\mathtt{ReachAvoid}$ ($\mathtt{RA}$)", r"$\mathtt{ReachAvoidDerived}$ ($\mathtt{RAD}$)"]
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 4, figsize=(48, 10))

# --- t-SNE scatter plots ---
for i in range(3):
    axes[0].scatter(rad_tsne[labels==i,0], rad_tsne[labels==i,1], label=label_names[i], color=colors[i], alpha=0.7)
    axes[1].scatter(emb_tsne[labels==i,0], emb_tsne[labels==i,1], label=label_names[i], color=colors[i], alpha=0.7)

# Remove text titles and add logos as titles
axes[0].set_title("")
axes[1].set_title("")
rad_logo_img = mpimg.imread("rad.png")
rad_imagebox = OffsetImage(rad_logo_img, zoom=0.4)
rad_ab = AnnotationBbox(rad_imagebox, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
axes[0].add_artist(rad_ab)
qwen_logo_img = mpimg.imread("qwen3.png")
qwen_imagebox = OffsetImage(qwen_logo_img, zoom=0.4)
qwen_ab = AnnotationBbox(qwen_imagebox, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
axes[1].add_artist(qwen_ab)

# Set same axis limits for both scatter plots
tsne_all_x = np.concatenate([rad_tsne[:,0], emb_tsne[:,0]])
tsne_all_y = np.concatenate([rad_tsne[:,1], emb_tsne[:,1]])
tsne_xmin, tsne_xmax = tsne_all_x.min(), tsne_all_x.max()
tsne_ymin, tsne_ymax = tsne_all_y.min(), tsne_all_y.max()
for ax in axes[:2]:
    ax.set_xlim(tsne_xmin, tsne_xmax)
    ax.set_ylim(tsne_ymin, tsne_ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

# --- Heatmaps ---
sns.heatmap(rad_distance_matrix, cmap="rocket_r", ax=axes[2], vmin=vmin, vmax=vmax, cbar=False)
axes[2].set_title("")
axes[2].set_xlabel("")
axes[2].set_ylabel("")
rad_imagebox2 = OffsetImage(rad_logo_img, zoom=0.4)
rad_ab2 = AnnotationBbox(rad_imagebox2, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
axes[2].add_artist(rad_ab2)
sns.heatmap(emb_distance_matrix, cmap="rocket_r", ax=axes[3], vmin=vmin, vmax=vmax, cbar=False)
axes[3].set_title("")
axes[3].set_xlabel("")
axes[3].set_ylabel("")
qwen_imagebox2 = OffsetImage(qwen_logo_img, zoom=0.4)
qwen_ab2 = AnnotationBbox(qwen_imagebox2, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
axes[3].add_artist(qwen_ab2)

# Set ticks for heatmaps
heatmap_tick_positions = [ (N - 1) / 2, N + (N - 1) / 2, 2 * N + (N - 1) / 2]
heatmap_tick_labels = [r"$\mathtt{R}$", r"$\mathtt{RA}$", r"$\mathtt{RAD}$"]
for ax in axes[2:]:
    ax.set_xticks(heatmap_tick_positions)
    ax.set_xticklabels(heatmap_tick_labels, rotation=0)
    ax.set_yticks(heatmap_tick_positions)
    ax.set_yticklabels(heatmap_tick_labels, rotation=90)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)

# Colorbar for heatmaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axes[3])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(axes[3].collections[0], cax=cax, orientation='vertical')
cbar.set_label('Distance', rotation=270, labelpad=40)

# Shared legend for scatter plots
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', label=label_names[i], markerfacecolor=colors[i], markersize=15, alpha=0.7) for i in range(3)]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=True, fancybox=True, shadow=False)

plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.18, wspace=0.18)
plt.savefig('combined_tsne_heatmap.png', bbox_inches='tight', pad_inches=0.05)
plt.close()
