import dill
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

from rad_embeddings import Encoder
from dfax import list2batch, batch2graph

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

parser = argparse.ArgumentParser()
parser.add_argument('--no-legend', action='store_true', help='Disable the legend')
parser.add_argument('--no-logos', action='store_true', help='Disable the logos above the plots')
args_cli, _ = parser.parse_known_args()

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
labels = []

# with open("datasets/dataset_R_n139.pkl", "rb") as f:
with open("datasets/dataset_R_n139_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())
    labels.extend([0] * len(data))
# with open("datasets/dataset_RA_n479.pkl", "rb") as f:
with open("datasets/dataset_RA_n479_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())
    labels.extend([1] * len(data))

# with open("datasets/dataset_RAD_n102.pkl", "rb") as f:
with open("datasets/dataset_RAD_n102_n100.pkl", "rb") as f:
    data = dill.load(f)
    dataset.update(data)
    dfax_list.extend(data.keys())
    labels.extend([2] * len(data))

dfax = dfax_list[0]
encoder = Encoder(
    max_size=dfax.max_n_states,
    n_tokens=dfax.n_tokens,
    seed=0
)

rad = [encoder(dfax.to_graph()).squeeze() for dfax in dfax_list]
# rad = encoder(batch2graph(list2batch([dfax.expand(5).to_graph() for dfax in dfax_list])))
emb = [jnp.array(dataset[dfax][1]) for dfax in dfax_list]

rad = np.stack([np.asarray(x) for x in rad])
emb = np.stack([np.asarray(x) for x in emb])
labels = np.array(labels)

# t-SNE projections
rad_tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(rad)
emb_tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(emb)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
label_names = [r"$\mathtt{Reach}$ ($\mathtt{R}$)", r"$\mathtt{ReachAvoid}$ ($\mathtt{RA}$)", r"$\mathtt{ReachAvoidDerived}$ ($\mathtt{RAD}$)"]

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for i in range(3):
    axes[0].scatter(rad_tsne[labels==i,0], rad_tsne[labels==i,1], label=label_names[i], color=colors[i], alpha=0.7, s=180)
    axes[1].scatter(emb_tsne[labels==i,0], emb_tsne[labels==i,1], label=label_names[i], color=colors[i], alpha=0.7, s=180)

# Remove text titles and add logos as titles
axes[0].set_title("")
axes[1].set_title("")

if not args_cli.no_logos:
    rad_logo_img = mpimg.imread("rad.png")
    rad_imagebox = OffsetImage(rad_logo_img, zoom=0.4)
    rad_ab = AnnotationBbox(rad_imagebox, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
    axes[0].add_artist(rad_ab)

    qwen_logo_img = mpimg.imread("qwen3.png")
    qwen_imagebox = OffsetImage(qwen_logo_img, zoom=0.4)
    qwen_ab = AnnotationBbox(qwen_imagebox, (0.5, 1.02), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0))
    axes[1].add_artist(qwen_ab)


# Set same axis limits for both plots
all_x = np.concatenate([rad_tsne[:,0], emb_tsne[:,0]])
all_y = np.concatenate([rad_tsne[:,1], emb_tsne[:,1]])
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()



for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

if not args_cli.no_legend:
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', label=label_names[i],
                      markerfacecolor=colors[i], markersize=15, alpha=0.7) for i in range(3)]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=True, fancybox=True, shadow=False)

plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.13, wspace=0.05)
plt.savefig('tsne.png', bbox_inches='tight', pad_inches=0.05)
plt.close()
