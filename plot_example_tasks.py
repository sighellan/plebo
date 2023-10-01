import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import GPy
plt.rcParams["font.family"] = "Times New Roman"

from synthetic import NOISE_VAR, gp_realisation

synth1 = pickle.load(open('problems/synth_2023-08-25-14-43-26-task-0.p', 'rb'))
synth2 = pickle.load(open('problems/synth_2023-08-25-14-43-26-task-1.p', 'rb'))

strong1 = pickle.load(open('problems/satellite_strong_0.p', 'rb'))
strong2 = pickle.load(open('problems/satellite_strong_1.p', 'rb'))

selection1 = pickle.load(open('problems/satellite_selection_23.p', 'rb'))
selection2 = pickle.load(open('problems/satellite_selection_74.p', 'rb'))

def make_direct_task(shared_locs, shared_obs, base_task):
    model = GPy.models.GPRegression(X=shared_locs, Y=shared_obs,
                                kernel=base_task['theta_true'].kernel,
                                noise_var=NOISE_VAR)

    sample = model.posterior_samples(base_task['domain'], size=1)

    return {'full_current_task': gp_realisation(
        base_task['domain'], sample)}

shared_locs = np.array([[0.25, 0.25],
                        [0.25, 0.28571429],
                        [0.25, 0.21428571],
                        [0.21428571, 0.25],
                        [0.28571429, 0.25],
                        [0.21428571, 0.21428571],
                        [0.28571429, 0.28571429],
                        [0.21428571, 0.28571429],
                        [0.28571429, 0.21428571],
                       ])

shared_obs = 3 + np.array([[6]+ [5]*8]).T

np.random.seed(42)
direct1 = make_direct_task(shared_locs, shared_obs, synth1)
direct2 = make_direct_task(shared_locs, shared_obs, synth2) 

def plot_problem(ax, task, version, fig):

    data_mat = np.ones((28,28))*np.nan
    
    mm = 28 if version == 'synth' else 1
    
    full_task = task['full_current_task']
    for ii in range(len(full_task.obs)):
        xx, yy = np.round(full_task.locs[ii,:] * mm).astype(int)
        data_mat[yy,xx] = full_task.obs[ii]
        
    # plot task
    cb = ax.imshow(data_mat, origin='lower', cmap='binary',
              extent=[-0.5/mm,27.5/mm,-0.5/mm,27.5/mm])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.05)
    plt.colorbar(cb, cax=cax)
    
    # indicate optimum
    max_idx = np.unravel_index(np.nanargmax(data_mat), data_mat.shape)
    ax.scatter(max_idx[1]/mm, max_idx[0]/mm, marker='*', c='red')
    
    # indicate missing values
    xx_nan, yy_nan = np.nonzero(np.isnan(data_mat))
    ax.scatter(yy_nan/mm, xx_nan/mm, marker='x', c='deepskyblue', s=5)
    
    
# Create plot
gridspec = dict(hspace=0.0, width_ratios=[1, 0.05, 1, 0.5, 1, 0.05, 1])
fig, axs = plt.subplots(nrows=2, ncols=7, gridspec_kw=gridspec,
                        figsize=(6,2.3))
for jj in range(2):
    for ii in [1, 3, 5]:
        # Some subplots are only there to get correct spacing
        axs[jj, ii].set_visible(False)

# Plot the tasks
version = 'synth'
for ii, task in enumerate([direct1, direct2]):
    plot_problem(axs[ii,0], task, version, fig)
for ii, task in enumerate([synth1, synth2]):
    plot_problem(axs[ii,2], task, version, fig)
version = 'satellite'
for ii, task in enumerate([strong1, strong2]):
    plot_problem(axs[ii, 4], task, version, fig)
for ii, task in enumerate([selection1, selection2]):
    plot_problem(axs[ii, 6], task, version, fig)
    
# Set titles and clean up plots
titles = ['Direct synth', '', 'Synth', '', 'Strong', '', 'Selection']
for ii in [0, 2, 4, 6]:
    axs[0, ii].set_xticks([])
    axs[0, ii].set_title(titles[ii])
for ii in [2, 6]:
    for jj in range(2):
        axs[jj, ii].set_yticks([])

plt.savefig('Figures/example_tasks.png', bbox_inches='tight', dpi=400)
