from plot_cluster_grouped_results import plot_clustered_subfig, groups
from optimisation import plot_order, opt_names

import matplotlib.pyplot as plt
import pickle
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

experiment_ids = {
    'synth': '835893-840977',
    'strong': '837796',
    'selection': '838921'
}

rows = {
    'Direct': 1,
    'Prior': 2,
    'No transfer': 0
}

columns = {
    'synth': 0,
    'strong': 1,
    'selection': 2
}

titles = ['Synthetic', 'Strong', 'Selection']

def dict_key(dd, val):
    for kk, vv in dd.items():
        if vv == val:
            return kk
    return None


scale = 0.8

fig, axes = plt.subplots(3, 3, figsize=(scale*10,scale*6))

for exp_label in experiment_ids.keys():
    exp_id = experiment_ids[exp_label]
    results = pickle.load(open('Figures/results_plots/%s.p' %exp_id, 'rb'))
    N_tasks, N_its = np.shape(results['HBO_numpyro'])
    for subset_label in rows.keys():
        ax = axes[rows[subset_label], columns[exp_label]]
        # Plot groups separately to ease interpretation
        plot_subset = groups[subset_label]
        plot_clustered_subfig(ax, plot_subset, results, N_tasks, N_its, mute=True, alpha=0.1)
        
for ii in range(2):
    for jj in range(3):
        axes[ii,jj].set_xticks([])
    
for ii in range(3):
    for jj in range(3):
        axes[ii,jj].spines['top'].set_visible(False)
        axes[ii,jj].spines['right'].set_visible(False)
        
for ii in range(3):
    ymin, ymax = np.inf, -np.inf
    for jj in range(3):
        ymin = min(ymin, axes[ii,jj].get_ylim()[0])
        ymax = max(ymax, axes[ii,jj].get_ylim()[1])
        
    for jj in range(3):
        axes[ii,jj].set_ylim([ymin, ymax])
        
        if jj != 0:
            axes[ii,jj].set_yticks([])
            
axes[2,1].set_xlabel('Iteration')
axes[1,0].set_ylabel("Perf. compared to PLeBO")

for jj in range(3):
    axes[0,jj].set_title(titles[jj])

handles_all, labels_all = [], []
for jj in range(3):
    handles, labels = axes[jj,0].get_legend_handles_labels()
    handles_all += handles
    labels_all += labels
    
filtered_handles, filtered_labels = [], []
for jj in range(len(labels_all)):
    if labels_all[jj] not in filtered_labels:
        filtered_labels.append(labels_all[jj])
        filtered_handles.append(handles_all[jj])
        
order_list = [plot_order[dict_key(opt_names, ff)] for ff in filtered_labels]
        
sorted_handles = [xx for _, xx in sorted(zip(order_list, filtered_handles))]
sorted_labels = [xx for _, xx in sorted(zip(order_list, filtered_labels))]
    
fig.legend(sorted_handles, sorted_labels, loc='lower center', ncols=5, frameon=False,
          bbox_to_anchor=(0.5, -0.1))

plt.savefig('Figures/results_plots/subsets/overview_results.png', 
            bbox_inches='tight', dpi=400)