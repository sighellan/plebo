import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from optimisation import opt_colours, opt_names, line_styles

exp_id = '835893-840977' # synthetic
# exp_id = '837796' # strong
# exp_id = '838921' # selection


groups = {
    'Prior': ['Shared', 'Gamma', 'HBO_true', 'HBO_numpyro'],
    'Direct': ['HBO_numpyro', 'Initial', 'Dir_trans'],
    'No transfer': ['HBO_numpyro', 'EI', 'RS', 'UCB', 'BoTorch'],
    'Top': ['HBO_numpyro', 'EI', 'Shared']
}



def plot_clustered_subfig(ax, plot_subset, results, N_tasks, N_its, alpha=0.4, mute=False):
    for label in plot_subset:
        if results[label] != []:
            # Check that all results have the same shape
            assert np.shape(results[label]) == np.shape(results['HBO_numpyro'])
            # Compute the difference between HBO_numpyro performance and
            # other methods
            mean_diff = -np.mean(results['HBO_numpyro'] - results[label], 0)
            ste_diff = np.std(results['HBO_numpyro'] - results[label], 
                              0)/np.sqrt(N_tasks)
            if not mute:
                print(label, np.mean(mean_diff))
            ax.plot(mean_diff,
                    label=opt_names[label], c=opt_colours[label], linestyle=line_styles[label])
            ax.fill_between(list(range(N_its)),
                             mean_diff-ste_diff, 
                             mean_diff+ste_diff,
                             color=opt_colours[label],
                             alpha=0.4)

if __name__ == "__main__":
    
    results = pickle.load(open('Figures/results_plots/%s.p' %exp_id, 'rb'))
    N_tasks, N_its = np.shape(results['HBO_numpyro'])
    
    for subset_label in groups:
        # Plot groups separately to ease interpretation
        plot_subset = groups[subset_label]
        fig, ax = plt.subplots(figsize=(6,4))
        plot_clustered_subfig(ax, plot_subset, results, N_tasks, N_its)


        ax.set_xlabel('Iteration')
        ax.set_ylabel('Norm. perf comp. to %s' %opt_names['HBO_numpyro'])
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        os.makedirs('Figures/results_plots/subsets', exist_ok=True)
        fig.savefig(
            'Figures/results_plots/subsets/plebo_comp_%s_%s.png' %(exp_id, subset_label),
        bbox_inches='tight', dpi=400)