from jax import random
from numpyro.infer import MCMC, NUTS
import pickle
import numpy as np
import sys
from prior_learning import kernel, prior_learning
sys.path.append('../')
from synthetic import gp_hypers
import time
import json


version = 'satellite' # 'synth' or 'satellite'
problem_id = 'strong'
# problem_id = '2023-08-25-14-43-26'
task_id = '-task-0'

N_cand = 200

# num = 4
candidate_id = problem_id#+'_%s' %num


if version == 'synth':
    version_lbl = 'synth'
    version_lbl_2 = 'synthetic'
else:
    version_lbl = 'satellite'
    version_lbl_2 = version_lbl


start_t = time.time()

if version == 'synth':
    problems = pickle.load(open('../problems/synth_%s.p' %(
        problem_id+task_id), 'rb'))
    
    related_tasks = problems['related_tasks']

    locs = np.array([problems['related_tasks'][ii].locs
                           for ii in range(problems['N_related_tasks'])])
    obs = np.array([problems['related_tasks'][ii].obs
                          for ii in range(problems['N_related_tasks'])])[:,:,0]
else:
    related_tasks = pickle.load(open(
        '../candidates/satellite_%s_tune_subset.p' %problem_id, 'rb'))

    locs = np.array([related_tasks[ii].locs
                       for ii in range(len(related_tasks))])
    obs = np.array([related_tasks[ii].obs
                      for ii in range(len(related_tasks))])[:,:,0]


J, N, F = np.shape(locs)

nuts_kernel = NUTS(prior_learning, find_heuristic_step_size=True)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=N_cand, num_chains=5)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, J, N, locs, kernel, Y=obs, extra_fields=('potential_energy',))

print(mcmc.print_summary())


pickle.dump(mcmc, open('mcmc_%s.p' %candidate_id, 'wb'))

# Some of the threads might not have mixed well.
# Only use the samples which have non-zero likelihood on the tuning data
N_samples = len(mcmc.get_samples()['phi_l'])
ii_qualify = []
for ii in range(N_samples):
    score = 0
    for jj in range(J):
        D = (related_tasks[jj].locs,
             related_tasks[jj].obs)
        theta = gp_hypers(
            mcmc.get_samples()['l_scale'][ii,jj],
            mcmc.get_samples()['s_var'][ii,jj],
            '%s,%s' %(ii,jj)
        )
        score += np.log(theta.joint_score(D))
    if score > -np.inf:
        ii_qualify.append(ii)

print('%s/%s points have non-zero likelihood' %(len(ii_qualify), N_samples))
assert len(ii_qualify) >= N_cand, 'Not enough qualified samples.'
pickle.dump(ii_qualify, open('mcmc_%s_qualify.p' %candidate_id, 'wb'))

# Randomly choose N_cand of the qualified samples to use
np.random.shuffle(ii_qualify)
ii_to_use = np.array(ii_qualify[:N_cand])


candidates = []
for ii in ii_to_use:
    # for each candidate, use the eta parameters to generate a theta value
    phi_l = mcmc.get_samples()['phi_l'][ii]
    phi_s = mcmc.get_samples()['phi_s'][ii]
    psi_l = mcmc.get_samples()['psi_l'][ii]
    psi_s = mcmc.get_samples()['psi_s'][ii]
    ll = np.random.gamma(shape=np.exp(phi_l), scale=1/np.exp(psi_l))
    ss = np.random.gamma(shape=np.exp(phi_s), scale=1/np.exp(psi_s))
    
    candidates.append(gp_hypers(lengthscale=ll, signal_var=ss, name='cand %s' %ii))
    
duration = time.time() - start_t
    
cand_file = '../candidates/numpyro_%s_%s.p' %(
    version_lbl_2, candidate_id)
pickle.dump(candidates, open(cand_file, 'wb'))

with open('../candidates/numpyro_%s_%s.txt' %(
    version_lbl_2, candidate_id), 'w') as f:
    f.write('Duration: %s' %duration)
    
print('%s candidates learned and stored at %s. Duration %s.' %(N_cand, cand_file, np.round(duration, 4)))

# Only use the chosen samples to generate a gamma prior
ll_shape = float(np.exp(np.mean(mcmc.get_samples()['phi_l'][ii_to_use])))
ll_rate = float(np.exp(np.mean(mcmc.get_samples()['psi_l'][ii_to_use])))
ss_shape = float(np.exp(np.mean(mcmc.get_samples()['phi_s'][ii_to_use])))
ss_rate = float(np.exp(np.mean(mcmc.get_samples()['psi_s'][ii_to_use])))

gamma_prior = {'ll_shape': ll_shape,
               'll_rate': ll_rate,
               'ss_shape': ss_shape,
               'ss_rate': ss_rate}

json.dump(gamma_prior, open('../candidates/%s_%s_gamma_prior.json' %(
    version_lbl, candidate_id), 'w'))
