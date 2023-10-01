import pickle
import numpy as np
import datetime

from synthetic import gp_hypers, gp_realisation, eta_synthetic

### Settings
seed = 2
N_start = 10
N_seeds = 10
N_related_tasks = 10
N_samps_related = 20
true_eta = eta_synthetic(5, 0.01, 2, 2)
N_problems = 100
###

timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

domain = []
length=28
width=28
for ii in range(length):
    for jj in range(width):
        domain.append([ii/width,jj/width])
domain = np.array(domain)
N = len(domain)


np.random.seed(seed)
# sample thetas
## Related tasks
### Initialise related tasks
related_tasks = []
related_tasks_true = []
for _ in range(N_related_tasks):
    theta_rel = true_eta.sample_theta()
    related_tasks_true.append(theta_rel)
    loc = domain[np.random.randint(0, len(domain), size=N_samps_related)]
    _, obs = theta_rel.sample_data(locs=loc)
    related_tasks.append(gp_realisation(loc, obs))

pickle.dump(related_tasks, open('candidates/synth_%s_tune_subset.p' %timestamp, 'wb'))


# Generate list of starting locations
starting_loc = {}
for init_seed in range(N_seeds):
    np.random.seed(init_seed)
    starting_loc[init_seed] = domain[np.random.randint(0, len(domain), size=N_start)]
    
np.random.seed(2)
for pp in range(N_problems):
    theta_true = true_eta.sample_theta()
    # Initialise current task
    full_D = theta_true.sample_data(locs=domain)
    _, full_obs = full_D
    full_current_task = gp_realisation(domain, full_obs)

    # Initialise starting points
    starting_points = {}
    for init_seed in range(N_seeds):

        loc = starting_loc[init_seed]
        obs = np.array([full_current_task.score(xx) for xx in loc])
        D_test = loc, obs
        starting_points[init_seed] = D_test

    problem = {'seed': seed,
               'N_start': N_start,
               'N_seeds': N_seeds,
               'N_related_tasks': N_related_tasks,
               'true_lengthscale': theta_true.lengthscale(),
               'true_signal_var': theta_true.signal_var(),
               'domain': domain,
               'theta_true': theta_true,
               'full_current_task': full_current_task,
               'related_tasks': related_tasks,
               'starting_points': starting_points,
               'true_eta': true_eta,
               'related_tasks_true': related_tasks_true}


    file_name = 'problems/synth_%s-task-%s.p' %(timestamp, pp)
    pickle.dump(problem, open(file_name, 'wb'))
print('Generated set of problems with timestamp %s' %timestamp)