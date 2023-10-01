import copy
import numpy as np
import GPy
import scipy.stats as st
import time

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

from synthetic import gp_hypers, NOISE_VAR, gp_realisation

opt_colours = {
    'RS': 'maroon',
    'BoTorch': 'crimson',
    'EI': 'palevioletred',
    'UCB': 'darkorange',
    'Dir_trans': 'darkgreen',
    'Initial': 'olive',
    'HBO_numpyro': 'black',
    'HBO_true': 'grey',
    'Gamma': 'midnightblue',
    'Shared': 'purple',
}

opt_names = {
    'RS': 'RandomSearch',
    'BoTorch': 'BoTorch',
    'HBO_true': 'TruePLeBO',
    'EI': 'EI',
    'Dir_trans': 'DirectTrans',
    'HBO_numpyro': 'PLeBO',
    'Initial': 'Initial',
    'UCB': 'UCB',
    'Gamma': 'Gamma',
    'Shared': 'Shared',
}

DIRECT = '--'
PRIOR = '-'
NONE = '-.'

line_styles = {
    'RS': NONE,
    'BoTorch': NONE,
    'HBO_true': PRIOR,
    'EI': NONE,
    'Dir_trans': DIRECT,
    'HBO_numpyro': PRIOR,
    'Initial': DIRECT,
    'UCB': NONE,
    'Gamma': PRIOR,
    'Shared': PRIOR,
}

plot_order = {
    'RS': 0,
    'BoTorch': 1,
    'EI': 2,
    'UCB': 3,
    'Dir_trans': 4,
    'Initial': 5,
    'HBO_numpyro': 6,
    'HBO_true': 7,
    'Gamma': 8,
    'Shared': 9
}

def get_tensor_synth(arr):
    return torch.tensor(arr, dtype=torch.float64)

def get_tensor_domain_synth(arr):
    return torch.tensor(np.array([arr]).transpose(1,0,2))
    
def acq_botorch_template(D_cur, domain, scale_labels, get_tensor, get_tensor_domain):
    
    # Expected improvement
    # Discretise the search space
    
    scale_domain = np.max(domain)
    
    loc, obs = D_cur
    tensor_X = get_tensor(loc)/scale_domain
    tensor_Y = get_tensor(obs)/scale_labels
    
    gp = SingleTaskGP(tensor_X, tensor_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_model(mll, max_retries=3)
        EI = ExpectedImprovement(gp, best_f=np.max(obs)/scale_labels)

        tensor_domain = get_tensor_domain(domain)/scale_domain
        acq_values_torch = EI(tensor_domain)
        acq_values = np.array(acq_values_torch.detach())
        max_idx = np.argmax(acq_values)
    except:
        print('BoTorch fell over')
        max_idx = np.random.randint(len(domain))
    return domain[max_idx]

def acq_botorch_synth(D_cur, domain):
    return acq_botorch_template(
        D_cur, domain, 1, get_tensor_synth, get_tensor_domain_synth)

def acq_rs(D_cur, domain):
    return domain[np.random.randint(len(domain))]

def do_optimisation_template(acq, D_start, steps, scorer, domain, stacker):
    D_cur = copy.deepcopy(D_start)
    durations = []
    start_t = time.time()
    for _ in range(steps):
        new_x = acq(D_cur, domain)
        new_y = scorer(new_x)
        X, Y = D_cur
        X = stacker(X, new_x)
        Y = stacker(Y, new_y)
        D_cur = (X, Y)
        durations.append(time.time() - start_t)
        start_t = time.time()
    return D_cur, durations

def synth_stacker(M, new_m):
    return np.vstack([M, new_m])

def do_optimisation_synth(acq, D_start, steps, scorer, domain):
    return do_optimisation_template(acq, D_start, steps, scorer, domain, synth_stacker)

def define_model(D_cur, feat_dims=2, kernel=None):
    loc, obs = D_cur
    if kernel is None:
        kernel = GPy.kern.RBF(input_dim=feat_dims)
    model = GPy.models.GPRegression(X=loc, Y=obs, kernel=kernel,
                                   noise_var=NOISE_VAR)
    return model

def learn_model(D_cur, feat_dims=2, kern=None):
    model = define_model(D_cur, feat_dims=feat_dims, kernel=kern)
    model.likelihood.variance.constrain_fixed(NOISE_VAR)
    try:
        model.optimize(optimizer='lbfgsb')
    except:
        print('model optimisation failed')
        model = define_model(D_cur)
    return model

def learn_prior_model(D_cur, ll_shape, ll_rate, ss_shape, ss_rate):
    model = define_model(D_cur)
    model.likelihood.variance.constrain_fixed(NOISE_VAR)
    
    model.kern.lengthscale.set_prior(GPy.priors.Gamma(ll_shape, ll_rate))
    model.kern.variance.set_prior(GPy.priors.Gamma(ss_shape, ss_rate))
    try:
        model.optimize(optimizer='lbfgsb')
    except:
        print('model optimisation failed')
        model = define_model(D_cur)
    return model

def acq_ei_prior_synth(D_cur, domain, gamma_prior):
    loc, obs = D_cur
    model = learn_prior_model(
        D_cur, gamma_prior['ll_shape'], gamma_prior['ll_rate'], 
        gamma_prior['ss_shape'], gamma_prior['ss_rate']
    )
    ei = ei_from_model(model, domain, obs)
    idx = np.argmax(ei)
    return domain[idx]

def acq_ucb_synth(D_cur, domain):
    model = learn_model(D_cur)
    ucb = ucb_from_model(model, domain)
    idx = np.argmax(ucb)
    return domain[idx]

def acq_ei_synth(D_cur, domain):
    loc, obs = D_cur
    model = learn_model(D_cur)
    ei = ei_from_model(model, domain, obs)
    idx = np.argmax(ei)
    return domain[idx]

def acq_ei_transfer_rel(D_cur, domain, related_tasks, num_other=100):
    loc_ext, obs_ext, domain_ext = extend_features(D_cur, related_tasks, domain, num_other)

    kernel_spat = GPy.kern.RBF(input_dim=2, active_dims=[0,1])
    kernel_trans = GPy.kern.RBF(input_dim=1+len(related_tasks),
                                active_dims=list(range(2,len(related_tasks)+1+2)), ARD=True)
    kernel = kernel_spat * kernel_trans
    
    model = GPy.models.GPRegression(X=loc_ext, Y=obs_ext, kernel=kernel,
                                   noise_var=NOISE_VAR)
    model.likelihood.variance.constrain_fixed(NOISE_VAR)
    try:
        model.optimize(optimizer='lbfgsb')
    except:
        print('model optimisation failed')
        kernel_spat = GPy.kern.RBF(input_dim=2, active_dims=[0,1])
        kernel_trans = GPy.kern.RBF(input_dim=4, active_dims=[2,3,4,5], ARD=True)
        kernel = kernel_spat * kernel_trans
        model = GPy.models.GPRegression(X=loc_ext, Y=obs_ext, kernel=kernel,
                                       noise_var=NOISE_VAR)
        pass
    ei = ei_from_model(model, domain_ext, obs=D_cur[1])
    idx = np.argmax(ei)
    return domain[idx]

def extend_features(D_cur, related_tasks, domain, num_other):
    next_task = gp_realisation(D_cur[0], D_cur[1])
    
    N_tasks = 1 + len(related_tasks)
    all_tasks_ext = np.zeros((0,2+N_tasks))
    for ii, cur_task in enumerate([next_task] + related_tasks):
        N_cur = len(cur_task.obs)

        cur_task_ext = np.hstack([cur_task.locs,
                                  np.zeros((N_cur, ii)),
                                  np.ones((N_cur, 1)),
                                  np.zeros((N_cur, N_tasks-ii-1))])
        all_tasks_ext = np.vstack([all_tasks_ext, cur_task_ext])
    
    all_tasks_obs = np.vstack([cur_task.obs for cur_task in [next_task]+related_tasks])

    # Only use num_other of the related tasks
    # Want all indices from current task
    N_this = len(D_cur[1])
    N_other = len(all_tasks_obs)-N_this
    if N_other > num_other:
        idx_this = list(range(N_this))
        idx_all_other = list(range(N_this, N_this+N_other))
        np.random.shuffle(idx_all_other)
        idx_other_use = idx_all_other[:num_other]
        idx_to_use = np.array(idx_this+idx_other_use)
        
        all_tasks_obs = all_tasks_obs[idx_to_use]
        all_tasks_ext = all_tasks_ext[idx_to_use]
    
    
    N_dom = len(domain)
    domain_ext = np.hstack([domain, np.ones((N_dom, 1)), np.zeros((N_dom, N_tasks-1))])

    return all_tasks_ext, all_tasks_obs, domain_ext

def ei_from_model(model, domain, obs):
    mean, var = model.predict(domain)
    std = np.sqrt(var)
    best_yet = np.max(obs)
    ei = (mean-best_yet)*(1-st.norm.cdf(best_yet, loc=mean, scale=std)) \
        +var*st.norm.pdf(best_yet, loc=mean, scale=std)
    return ei

def ucb_from_model(model, domain, par=1):
    mean, var = model.predict(domain)
    std = np.sqrt(var)
    ucb = mean+par*std
    return ucb

def single_ei(cand, D_test, domain):
    loc, obs = D_test
    model = GPy.models.GPRegression(X=loc, Y=obs, kernel=cand.kernel,
                                   noise_var=NOISE_VAR)
    ei = ei_from_model(model, domain, obs)
    return ei

def weighted_ei(candidates, D_test, domain):
    ei = np.zeros((len(domain), 1))
    W = 0
    for cc in candidates:
        ww = cc.joint_score(D_test)
        ei += ww*single_ei(cc, D_test, domain)
        W += ww
    return ei/W

def acq_hbo_cand_synth(D_cur, domain, candidates):
    w_ei = weighted_ei(candidates, D_cur, domain)
    idx = np.argmax(w_ei)
    return domain[idx]