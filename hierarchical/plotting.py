"""
plotting functions
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.patches as mpatches
from .category import category_to_indices


sns.set_theme(
    context="paper",
    style="white",  # 'whitegrid', 'dark', 'darkgrid', ...
    palette="colorblind",
    font="DejaVu Sans",  # 'serif'
    font_scale=1.75,  # 1.75, 2, ...
)

def cos_heatmap(mats, titles = None, figsize = (19, 8),
                labels = None,
                cmap=None, use_absvals=False, save_to = None):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(mats), wspace=0)

    vmin = -0.001 if use_absvals else -1.001
    vmax = 1.001

    if cmap is None:
        # darker defaults
        cmap = "mako" if use_absvals else "icefire"

    ims = []
    for i in range(len(mats)):
        ax = plt.subplot(gs[0, i])
        im = ax.imshow(mats[i], aspect = 'equal', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ims.append(im)
        if labels != None:
            ytick = list(range(len(labels)))
            ax.set_yticks(ytick)
            ax.set_xticks(ytick)
            if i == 0:
                ax.set_yticklabels(labels)
                # ax.set_xticklabels(labels, rotation = 60, ha = "right")
                ax.set_xticklabels([])
            else:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            
        if titles != None:
            ax.set_title(titles[i])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ims[-1], cax=cax, orientation='vertical')

    plt.tight_layout()
    if save_to != None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()

def proj_2d(dir1, dir2, unembed, vocab_list, ax,
              added_inds=None,
              normalize = True,
              orthogonal=False, k=10, fontsize=10,
              alpha=0.2, s=0.5,
              target_alpha = 0.9, target_s = 2,
              xlim = None,
              ylim = None,
              draw_arrows = False,
              arrow1_name = None,
              arrow2_name = None,
              right_topk = True,
              left_topk = True,
              top_topk = True,
              bottom_topk = True,
              xlabel="dir1",
              ylabel="dir2",
              title="2D projection plot"):
    original_dir1 = dir1
    original_dir2 = dir2
    
    if normalize:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 / dir2.norm()
    if orthogonal:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 - (dir2 @ dir1) * dir1
        dir2 = dir2 / dir2.norm()

        arrow1 = [(original_dir1 @ dir1).cpu().numpy(), 0]
        arrow2 = [(original_dir2 @ dir1).cpu().numpy(), (original_dir2 @ dir2).cpu().numpy()]

    proj1 = unembed @ dir1
    proj2 = unembed @ dir2
    
    ax.scatter(proj1.cpu().numpy(), proj2.cpu().numpy(),
               alpha=alpha, color="gray", s=s)
    
    def _add_labels_for_largest(proj, largest):
        indices = torch.topk(proj, k=k, largest=largest).indices
        for idx in indices:
            if "$" not in vocab_list[idx]:
                ax.text(proj1[idx], proj2[idx], vocab_list[idx], fontsize=fontsize)
    
    if right_topk:
        _add_labels_for_largest(proj1, largest=True)
    if left_topk:
        _add_labels_for_largest(proj1, largest=False)
    if top_topk:
        _add_labels_for_largest(proj2, largest=True)
    if bottom_topk:
        _add_labels_for_largest(proj2, largest=False)

    if added_inds:
        colors = iter(["b", "r", "green", "orange",
                       "skyblue", "pink",  "yellowgreen", "orange", "yellow",
                       "brown", "cyan", "olive", "purple", "lime"])
        legend_handles = []
        for label, indices in added_inds.items():
            color = next(colors)
            word_add = [vocab_list[i] for i in indices]
            for word, idx in zip(word_add, indices):
                # ax.text(proj1[idx], proj2[idx], word,
                #         fontsize=fontsize, bbox=dict(facecolor=color, alpha=0.2))
                ax.scatter(proj1[idx].cpu().numpy(), proj2[idx].cpu().numpy(),
                           alpha=target_alpha, color=color, s=target_s)
            # Create a patch for the legend
            legend_handles.append(mpatches.Patch(color=color, label=label))
        
        ax.legend(handles=legend_handles, loc = 'lower left')


    if xlim is not None:
        ax.set_xlim(xlim)
        ax.hlines(0, xmax=xlim[1], xmin=xlim[0],
                  colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.hlines(0, xmax=proj1.max().cpu().numpy(), xmin=proj1.min().cpu().numpy(),
              colors="black", alpha=0.3, linestyles="dashed")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.vlines(0, ymax=ylim[1], ymin=ylim[0],
                  colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.vlines(0, ymax=proj2.max().cpu().numpy(), ymin=proj2.min().cpu().numpy(),
              colors="black", alpha=0.3, linestyles="dashed")
        
    if draw_arrows:
        ax.arrow(0, 0, arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                 width=0.1, fc='blue', ec='blue',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if arrow1_name!=None:
            ax.text(arrow1[0]/2, arrow1[1]/2-1.5, arrow1_name, fontsize=fontsize,
                    bbox=dict(facecolor='blue', alpha=0.2))
        ax.arrow(0, 0, arrow2[0], arrow2[1], head_width=0.5, head_length=0.5,
                 width=0.1,  fc='red', ec='red',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if arrow2_name!=None:
            ax.text(arrow2[0]/2-1.5, arrow2[1]/2+1.5, arrow2_name, fontsize=fontsize,
                    bbox=dict(facecolor='red', alpha=0.2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def proj_2d_single_diff(higher, subcat1, subcat2, unembed, vocab_list, ax,
                        added_inds=None,
                        normalize = True,
                        orthogonal=False, k=10, fontsize=10,
                        alpha=0.2, s=0.5,
                        target_alpha = 0.9, target_s = 2,
                        xlim = None,
                        ylim = None,
                        draw_arrows = False,
                        arrow1_name = None,
                        arrow2_name = None,
                        right_topk = True,
                        left_topk = True,
                        top_topk = True,
                        bottom_topk = True,
                        xlabel="dir1",
                        ylabel="dir2",
                        title="2D projection plot"):
    dir1 = higher
    dir2 = subcat2 - subcat1

    original_higher = higher
    original_subcat1 = subcat1
    original_subcat2 = subcat2
    original_dir1 = dir1
    original_dir2 = dir2
    
    if normalize:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 / dir2.norm()
    if orthogonal:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 - (dir2 @ dir1) * dir1
        dir2 = dir2 / dir2.norm()

        arrow1 = [(original_dir1 @ dir1).cpu().numpy(), 0]
        arrow2 = [(original_dir2 @ dir1).cpu().numpy(), (original_dir2 @ dir2).cpu().numpy()]

    proj1 = unembed @ dir1
    proj2 = unembed @ dir2
    
    ax.scatter(proj1.cpu().numpy(), proj2.cpu().numpy(),
               alpha=alpha, color="gray", s=s)
    
    def _add_labels_for_largest(proj, largest):
        indices = torch.topk(proj, k=k, largest=largest).indices
        for idx in indices:
            if "$" not in vocab_list[idx]:
                ax.text(proj1[idx], proj2[idx], vocab_list[idx], fontsize=fontsize)
    
    if right_topk:
        _add_labels_for_largest(proj1, largest=True)
    if left_topk:
        _add_labels_for_largest(proj1, largest=False)
    if top_topk:
        _add_labels_for_largest(proj2, largest=True)
    if bottom_topk:
        _add_labels_for_largest(proj2, largest=False)

    if added_inds:
        colors = iter(["b",  "orange", "r", "green", 
                       "skyblue", "pink",  "yellowgreen", "orange", "yellow",
                       "brown", "cyan", "olive", "purple", "lime"])
        legend_handles = []
        for label, indices in added_inds.items():
            color = next(colors)
            word_add = [vocab_list[i] for i in indices]
            for word, idx in zip(word_add, indices):
                # ax.text(proj1[idx], proj2[idx], word,
                #         fontsize=fontsize, bbox=dict(facecolor=color, alpha=0.2))
                ax.scatter(proj1[idx].cpu().numpy(), proj2[idx].cpu().numpy(),
                            alpha=target_alpha, color=color, s=target_s)
            # Create a patch for the legend
            legend_handles.append(mpatches.Patch(color=color, label=label))
        
        ax.legend(handles=legend_handles, loc = 'lower left')


    if xlim is not None:
        ax.set_xlim(xlim)
        ax.hlines(0, xmax=xlim[1], xmin=xlim[0],
                  colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.hlines(0, xmax=proj1.max().cpu().numpy(), xmin=proj1.min().cpu().numpy(),
              colors="black", alpha=0.3, linestyles="dashed")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.vlines(0, ymax=ylim[1], ymin=ylim[0],
                  colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.vlines(0, ymax=proj2.max().cpu().numpy(), ymin=proj2.min().cpu().numpy(),
              colors="black", alpha=0.3, linestyles="dashed")
        
    if draw_arrows:
        ax.arrow(0, 0, arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                 width=0.1, fc='blue', ec='blue',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if arrow1_name!=None:
            ax.text(arrow1[0]/2, arrow1[1]/2-1.5, arrow1_name, fontsize=fontsize,
                    bbox=dict(facecolor='blue', alpha=0.2))
        ax.arrow((original_subcat1 @ dir1).cpu().numpy(),
                  (original_subcat1 @ dir2).cpu().numpy(),
                  arrow2[0], arrow2[1], head_width=0.5, head_length=0.5,
                 width=0.1,  fc='red', ec='red',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if arrow2_name!=None:
            ax.text((original_subcat1 @ dir1).cpu().numpy()+ 3*arrow2[0]/4-5,
                    (original_subcat1 @ dir2).cpu().numpy() + 3*arrow2[1]/4, arrow2_name, fontsize=fontsize,
                    bbox=dict(facecolor='red', alpha=0.2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def proj_2d_double_diff(higher1, higher2, subcat1, subcat2, unembed, vocab_list, ax,
                        added_inds=None,
                        normalize = True,
                        orthogonal=False, k=10, fontsize=10,
                        alpha=0.2, s=0.5,
                        target_alpha = 0.9, target_s = 2,
                        xlim = None,
                        ylim = None,
                        draw_arrows = False,
                        arrow1_name = None,
                        arrow2_name = None,
                        right_topk = True,
                        left_topk = True,
                        top_topk = True,
                        bottom_topk = True,
                        xlabel="dir1",
                        ylabel="dir2",
                        title="2D projection plot"):
    dir1 = higher2 - higher1
    dir2 = subcat2 - subcat1

    original_higher1 = higher1
    original_higher2 = higher2
    original_subcat1 = subcat1
    original_subcat2 = subcat2
    original_dir1 = dir1
    original_dir2 = dir2
    
    if normalize:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 / dir2.norm()
    if orthogonal:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 - (dir2 @ dir1) * dir1
        dir2 = dir2 / dir2.norm()

        arrow1 = [(original_dir1 @ dir1).cpu().numpy(), 0]
        arrow2 = [(original_dir2 @ dir1).cpu().numpy(), (original_dir2 @ dir2).cpu().numpy()]

    proj1 = unembed @ dir1
    proj2 = unembed @ dir2
    
    ax.scatter(proj1.cpu().numpy(), proj2.cpu().numpy(),
               alpha=alpha, color="gray", s=s)
    
    def _add_labels_for_largest(proj, largest):
        indices = torch.topk(proj, k=k, largest=largest).indices
        for idx in indices:
            if "$" not in vocab_list[idx]:
                ax.text(proj1[idx], proj2[idx], vocab_list[idx], fontsize=fontsize)
    
    if right_topk:
        _add_labels_for_largest(proj1, largest=True)
    if left_topk:
        _add_labels_for_largest(proj1, largest=False)
    if top_topk:
        _add_labels_for_largest(proj2, largest=True)
    if bottom_topk:
        _add_labels_for_largest(proj2, largest=False)

    if added_inds:
        colors = iter([ "green", "b",   "orange", "r",
                       "skyblue", "pink",  "yellowgreen", "orange", "yellow",
                       "brown", "cyan", "olive", "purple", "lime"])
        legend_handles = []
        for label, indices in added_inds.items():
            color = next(colors)
            word_add = [vocab_list[i] for i in indices]
            for word, idx in zip(word_add, indices):
                # ax.text(proj1[idx], proj2[idx], word,
                #         fontsize=fontsize, bbox=dict(facecolor=color, alpha=0.2))
                ax.scatter(proj1[idx].cpu().numpy(), proj2[idx].cpu().numpy(),
                            alpha=target_alpha, color=color, s=target_s)
            # Create a patch for the legend
            legend_handles.append(mpatches.Patch(color=color, label=label))
        
        ax.legend(handles=legend_handles, loc = 'lower left')


    if xlim is not None:
        ax.set_xlim(xlim)
        ax.hlines(0, xmax=xlim[1], xmin=xlim[0],
                  colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.hlines(0, xmax=proj1.max().cpu().numpy(), xmin=proj1.min().cpu().numpy(),
              colors="black", alpha=0.3, linestyles="dashed")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.vlines(0, ymax=ylim[1], ymin=ylim[0],
                  colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.vlines(0, ymax=proj2.max().cpu().numpy(), ymin=proj2.min().cpu().numpy(),
              colors="black", alpha=0.3, linestyles="dashed")
        
    if draw_arrows:
        ax.arrow((original_higher1 @ dir1).cpu().numpy(),
                  (original_higher1 @ dir2).cpu().numpy(),
                  arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                 width=0.1, fc='blue', ec='blue',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if arrow1_name!=None:
            ax.text((original_higher1 @ dir1).cpu().numpy()+ arrow1[0]*0.2, 
                    (original_higher1 @ dir2).cpu().numpy()+ arrow1[1]*0.2-1.5, arrow1_name, fontsize=fontsize,
                    bbox=dict(facecolor='blue', alpha=0.2))
        ax.arrow((original_subcat1 @ dir1).cpu().numpy(),
                  (original_subcat1 @ dir2).cpu().numpy(),
                  arrow2[0], arrow2[1], head_width=0.5, head_length=0.5,
                 width=0.1,  fc='red', ec='red',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
        if arrow2_name!=None:
            ax.text((original_subcat1 @ dir1).cpu().numpy()+ arrow2[0]/2+1,
                    (original_subcat1 @ dir2).cpu().numpy() + arrow2[1]/2, arrow2_name, fontsize=fontsize,
                    bbox=dict(facecolor='red', alpha=0.2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def show_evaluation(vec_reps_split, sorted_keys, vocab_dict, version = 'lda', save_to=None):
    ind, original, shuffled = ({'train': [], 'test': [], 'random': []} for _ in range(3))
    g = vec_reps_split['g']
    shuffled_g = vec_reps_split['shuffled_g']

    num_samples = 200000
    torch.random.manual_seed(100)
    all_indices = torch.randperm(g.shape[0])
    ind['random'] = all_indices[:num_samples]
    
    for node in sorted_keys:
        original_dir = vec_reps_split['original']['split'][node][version]
        original_dir = original_dir / original_dir.norm()**2
        shuffled_dir = vec_reps_split['shuffled']['split'][node][version]
        shuffled_dir = shuffled_dir / shuffled_dir.norm()**2

        ind['train'] = category_to_indices(vec_reps_split['train_lemmas'][node], vocab_dict)
        ind['test'] = category_to_indices(vec_reps_split['test_lemmas'][node], vocab_dict)

        for key in ind.keys():
            original[key].append((g[ind[key]] @ original_dir).cpu().numpy())
            shuffled[key].append((shuffled_g[ind[key]] @ shuffled_dir).cpu().numpy())

    
    inds = range(len(sorted_keys))
    colors = {'train': 'green', 'test': 'blue', 'random': 'orange'}
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    for key in ind.keys():
        proj_mean = [proj.mean() for proj in original[key]]
        proj_std = [proj.std() for proj in original[key]]

        ax[0].plot(inds, proj_mean, alpha=0.8, color = colors[key], linewidth = 1, label=key)
        # ax[0].scatter(inds, proj_mean, alpha=0.7, color = colors[key], s = 3, label=key)
        ax[0].errorbar(inds, proj_mean, yerr=proj_std, color = colors[key], capsize=3, ecolor = colors[key], alpha=0.1)
    # ax[0].set_xticks([])
    ax[0].set_title('Original Unembeddings')
    ax[0].set_ylim(-1,2)

    for key in ind.keys():
        proj_mean = [proj.mean() for proj in shuffled[key]]
        proj_std = [proj.std() for proj in shuffled[key]]

        ax[1].plot(inds, proj_mean, alpha=0.8, color = colors[key], linewidth = 1, label=key)
        # ax[1].scatter(inds, proj_mean, alpha=0.7, color = colors[key], s = 3, label=key)
        ax[1].errorbar(inds, proj_mean, yerr=proj_std, color = colors[key], capsize=3, ecolor = colors[key], alpha=0.1)
    # ax[1].set_xticks([])
    ax[1].set_title('Shuffled Unembeddings')
    ax[1].set_ylim(-1,2)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.supxlabel(rf'Binary Features in WordNet Hierarchy',x=0.5, y=0.08, fontsize=17)
    # fig.supylabel(rf'$(g(y)^\top\bar\ell_w) / \|\bar\ell_w\|^2$')
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()