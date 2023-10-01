import pickle
import numpy as np

num_its = 50

for exp_ids in [
    [835893, 840977], # synthetic
    [837796], # strong
    [838921]  # selection
]:
    print('\n %s' %exp_ids)

    durations = pickle.load(open(
        'Figures/results_plots/durations-%s.p' %'-'.join(
            map(str, exp_ids)), 'rb'))

    for label in durations:
        if len(durations[label]) < 1:
            print('Not present: %s' %label)
            continue
        if label != 'RS':
            print(label, np.round(np.mean(durations[label]/num_its), 2))
        else:
            # Need more digits for RS
            print(label, np.round(np.mean(durations[label]/num_its), 3))
    print(np.mean(durations['HBO_numpyro']/durations['EI']))