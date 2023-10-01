import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import scipy.stats as st

from jam import eta_jam, jam_scorer, prepare_plot

colours = {'train': 'gray',
          'test': 'purple'}

X = np.arange(200, 600, 1)

def make_plot(title, train, test, locs_dict, ident='', shift=0, ax=None,
             train_label='train', plot_both=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4*0.75,2.5*0.75))
    else:
        fig = None
    for fruit in [test, train]:
        if fruit.name == 'train':
            locs = locs_dict[fruit.name]
        else:
            if title == 'Direct':
                best_train_idx = np.argmax(train.score(locs_dict['train']))
                locs = np.hstack([locs_dict[fruit.name], locs_dict['train'][best_train_idx]])
            elif title == 'Prior':
                locs = np.hstack([locs_dict[fruit.name], np.max(locs_dict[fruit.name])+25*30/fruit.std()])
            elif title == 'Shift':
                locs = np.hstack([locs_dict[fruit.name], train.mean()-shift])
            else:
                locs = locs_dict[fruit.name]
        vals = fruit.score(locs)

        if fruit.name == 'test':
            zorder = 100
            ax.plot(X, fruit.scale*fruit.score(X), label='true '+fruit.name,
                    color=colours[fruit.name], linestyle=':', zorder=zorder)
            ax.scatter(locs, fruit.scale*vals, color=colours[fruit.name],
                       label=fruit.name + ' obs.',
                       marker='o', zorder=zorder)
            if title in ['Direct', 'Prior', 'Shift']:
                ax.scatter(locs[-1], fruit.scale*vals[-1], color=colours[fruit.name],
                           label='new obs.',
                           marker='D', s=50, zorder=zorder)
            
            
        else:
            zorder = 10
            if title != 'Shift':
                shift_list = [0]
            else:
                shift_list = [shift]
                if plot_both: shift_list = [0] + shift_list
            
            for shift in shift_list:
                ax.plot(locs - shift, fruit.scale*vals, marker='x',
                        color=colours[fruit.name] if shift == 0 else 'midnightblue',
                          label=train_label+ ' shifted' if shift != 0 else train_label,
                       zorder=zorder)
                ax.scatter(fruit.mean() - shift, fruit.score(fruit.mean())*fruit.scale,
                           label='%s max' %train_label if shift == shift_list[0] else None,
                              color=colours[fruit.name] if shift == 0 else 'midnightblue', marker="s", zorder=zorder)
            
    prepare_plot(ax)
    ax.legend(bbox_to_anchor=(0.9, 0.95), frameon=False)
    ax.set_xlim([180, 620])
    if fig is not None:
        plt.savefig('Figures/Presentation%s_%s.png' %(ident, title), bbox_inches='tight', dpi=1000)
    return ax

scale = 0.6
fig, axs = plt.subplots(2, 3, figsize=(13*scale,4*scale))

train = jam_scorer(450, 30, 'train')
test = jam_scorer(300, 30, 'test')
train.scale = 1000
test.scale = 1000

locs_dict = {'train': np.arange(train.mean()-200, train.mean()+150, 10),
            'test': np.array([test.mean()-50, test.mean()-25]),
            }

for ii, title in enumerate(['', 'Direct', 'Shift']):
    ax = make_plot(title, train, test, locs_dict, ident='_v1',
                   shift=150, ax=axs[1,ii], train_label='tune',
                  plot_both=True)

train = jam_scorer(450, 30, 'train')
test = jam_scorer(450, 10, 'test')
train.scale = 1000
test.scale = 350

locs_dict = {'train': np.arange(train.mean()-200, train.mean()+150, 10),
            'test': np.array([test.mean()-50, test.mean()-25]),
            }

for ii, title in enumerate(['', 'Direct', 'Shift']):
    make_plot(title, train, test, locs_dict, ident='_v2',
                   shift=-50, ax=axs[0,ii], train_label='tune',
             plot_both=True);
    
titles = ['Set-up', 'Direct transfer', 'Prior transfer']
for ii in range(3):
    axs[0,ii].set_xticks([])
    axs[0,ii].set_xlabel('')
    axs[0,ii].legend().set_visible(False)
    axs[0,ii].set_title(titles[ii])

for ii in range(1,3):
    for jj in range(2):
        axs[jj,ii].set_yticks([])
        axs[jj,ii].set_ylabel('')
        
for ii in [0, 1]:
    axs[1,ii].legend().set_visible(False)
    
axs[1,2].legend(bbox_to_anchor=(1.1, -0.45), 
                frameon=False, ncols=6)

plt.savefig('Figures/Presentation_together.png', bbox_inches='tight', dpi=400)