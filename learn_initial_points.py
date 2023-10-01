import pickle
import numpy as np

problem_id = '2023-08-25-14-43-26'
task_id = '-task-0'

problems = pickle.load(open('problems/synth_%s.p' %(problem_id+task_id), 'rb'))

synth_locs = np.array([problems['related_tasks'][ii].locs
                       for ii in range(problems['N_related_tasks'])])
synth_obs = np.array([problems['related_tasks'][ii].obs
                      for ii in range(problems['N_related_tasks'])])[:,:,0]
J, N, F = np.shape(synth_locs)

best_locs = []
best_obs = []

for ii in range(problems['N_related_tasks']):
    best_idx = np.argmax(problems['related_tasks'][ii].obs)
    best_locs.append(problems['related_tasks'][ii].locs[best_idx])
    best_obs.append(problems['related_tasks'][ii].obs[best_idx])
    
best_locs = np.array(best_locs)
best_obs = np.array(best_obs)

D_best_historic = best_locs, best_obs

pickle.dump(D_best_historic, open(
    'candidates/synth-starting-points-%s.p' %(problem_id), 'wb'))