import datetime
import pickle
import numpy as np
import argparse
import os
import time
import json

from optimisation import (
    do_optimisation_synth, acq_rs, acq_botorch_synth, acq_hbo_cand_synth,
    acq_ei_synth, acq_ei_transfer_rel, acq_ucb_synth, acq_ei_prior_synth,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--benchmark", help="which type of experiment we're running", type=str, default=None)
    parser.add_argument("--res_folder", help="folder to put results in.", type=str, default=None)
    parser.add_argument("--user_id", help="user id, used for folders. Needed for cluster.", type=str, default=None)
    parser.add_argument("--acq_name", help="BO acquisition function.", type=str, default=None)
    parser.add_argument("--steps", help="BO iterations.", type=int, default=None)
    parser.add_argument("--num_seeds", help="BO replications.", type=int, default=None)
    parser.add_argument("--problem_id", help="ID for problem file.", type=str, default=None)
    parser.add_argument("--task_id", help="task ID for problem file.", type=str, default='')
    parser.add_argument("--starting_points", help="size of initial design.", type=int, default=10)
    
    args = parser.parse_args()
    
    return args


def do_BO_synth(args):
    candidates_file = 'numpyro_synthetic_%s.p' %args.problem_id
    initial_file = 'synth-starting-points-%s.p' %args.problem_id
    problem_file = 'synth_%s.p' %(args.problem_id+args.task_id)
    related_file = 'synth_%s_tune_subset.p' %args.problem_id
    prior_file = 'synth_%s_gamma_prior.json' %args.problem_id
    shared_file = 'synth_%s_supermodel.p' %args.problem_id

    out_label = 'synth'
    do_BO(candidates_file, initial_file, problem_file, related_file, prior_file, shared_file, out_label, args)


def do_BO_satellite(args):
    candidates_file = 'numpyro_satellite_%s.p' %args.problem_id
    initial_file = 'satellite-starting-points-%s.p' %args.problem_id
    problem_file = 'satellite_%s_%s.p' %(args.problem_id, args.task_id)
    related_file = 'satellite_%s_tune_subset.p' %args.problem_id
    prior_file = 'satellite_%s_gamma_prior.json' %args.problem_id
    shared_file = 'satellite_%s_supermodel.p' %args.problem_id

    out_label = 'satellite'
    do_BO(candidates_file, initial_file, problem_file, related_file, prior_file, shared_file, out_label, args)
    
    
def do_BO(candidates_file, initial_file, problem_file, related_file, prior_file, shared_file, out_label, args):
    start_t = time.time()

    # Preprocessing/loading
    uid = args.user_id
    steps = args.steps
    acq_name = args.acq_name
    problem_id = args.problem_id
    task_id = args.task_id
    num_seeds = args.num_seeds

    seeds = list(range(num_seeds))

    if args.res_folder:
        # Run on cluster
        assert uid is not None
        problem_dict = pickle.load(open('/disk/scratch/%s/%s' %(uid, problem_file), 'rb'))
        candidates_numpyro = pickle.load(open('/disk/scratch/%s/%s' %(uid, candidates_file), 'rb'))
        initial_points = pickle.load(open('/disk/scratch/%s/%s' %(uid, initial_file), 'rb'))
        related_tasks = pickle.load(open('/disk/scratch/%s/%s' %(uid, related_file), 'rb'))
        gamma_prior = json.load(open('/disk/scratch/%s/%s' %(uid, prior_file), 'r'))
        shared_hps = pickle.load(open('/disk/scratch/%s/%s' %(uid, shared_file), 'rb'))
    else:
        # Run locally
        problem_dict = pickle.load(open('problems/%s' %problem_file, 'rb'))
        candidates_numpyro = pickle.load(open('candidates/%s' %candidates_file, 'rb'))
        initial_points = pickle.load(open('candidates/%s' %initial_file, 'rb'))
        related_tasks = pickle.load(open('candidates/%s' %related_file, 'rb'))
        gamma_prior = json.load(open('candidates/%s' %prior_file, 'r'))
        shared_hps = pickle.load(open('candidates/%s' %shared_file, 'rb'))


    candidates_true = [problem_dict['theta_true']]
    domain = problem_dict['domain']
#     related_tasks = problem_dict['related_tasks']
    full_current_task = problem_dict['full_current_task']


    # acq_hbo = lambda D_cur, domain : acq_hbo_cand_synth(D_cur, domain, candidates)
    acq_hbo_true = lambda D_cur, domain : acq_hbo_cand_synth(D_cur, domain, candidates_true)
    acq_hbo_numpyro = lambda D_cur, domain : acq_hbo_cand_synth(D_cur, domain, candidates_numpyro)
    acq_transfer = lambda D_cur, domain : acq_ei_transfer_rel(D_cur, domain, related_tasks)
    acq_gamma = lambda D_cur, domain : acq_ei_prior_synth(D_cur, domain, gamma_prior)
    acq_shared = lambda D_cur, domain : acq_hbo_cand_synth(D_cur, domain, [shared_hps])
    
    acquisition_functions = {
        'EI': acq_ei_synth,
        'UCB': acq_ucb_synth,
        'HBO_true': acq_hbo_true,
        'Dir_trans': acq_transfer,
    #     'HBO': acq_hbo,
        'RS': acq_rs,
        'BoTorch': acq_botorch_synth,
        'HBO_numpyro': acq_hbo_numpyro,
        'Initial': acq_ei_synth,
        'Gamma': acq_gamma,
        'Shared': acq_shared,
    }

    results = {key: [] for key in acquisition_functions}
    results['args'] = args
    results['durations'] = {key: [] for key in acquisition_functions}

    # Do BO
    if acq_name == 'Initial':
        np.random.seed(seeds[0])
        D_hist = initial_points
        locs_start, _ = D_hist
        obs_start = np.array([full_current_task.score(xx) for xx in locs_start])
        print(np.shape(obs_start))
        D_test = (locs_start, obs_start)
        N_initial = len(obs_start)

        opt_res, opt_dur = do_optimisation_synth(
                acquisition_functions[acq_name], 
                D_test, steps+args.starting_points-N_initial, full_current_task.score, domain)
        results[acq_name].append(opt_res)
        results['durations'][acq_name].append(opt_dur)

    else:
        for seed in seeds:
            print(seed)
            if acq_name == 'RS' and args.benchmark == 'synth':
                # Otherwise the same seed is used for the initial points
                # as for the choices of RS, giving it an unfair disadvantage
                # Only relevant for the synth benchmark.
                np.random.seed(seed+100)
            else:
                np.random.seed(seed)
            D_start_full = problem_dict['starting_points'][seed]
            
            # Limit size of initial design
            locs_start_full, obs_start_full = D_start_full
            locs_start = locs_start_full[:args.starting_points]
            obs_start = obs_start_full[:args.starting_points]
            D_test = (locs_start, obs_start)

            opt_res, opt_dur = do_optimisation_synth(
                    acquisition_functions[acq_name], 
                    D_test, steps, full_current_task.score, domain)
            results[acq_name].append(opt_res)
            results['durations'][acq_name].append(opt_dur)

    duration = time.time() - start_t
    results['duration'] = duration

    # Store experiment
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_id = '%s_%s_%s_bo_%s' %(out_label, problem_id, task_id, timestamp)
    if args.res_folder:
        res_folder = '/disk/scratch/%s/%s' %(uid, args.res_folder)
        os.makedirs(res_folder, exist_ok=True)
        file_name = '%s/%s.p' %(res_folder, file_id)
    else:
        file_name = 'results/%s.p' %file_id
    print(file_name)
    pickle.dump(results, open(file_name, 'wb'))
    
if __name__ == "__main__":
    args = parse_arguments()
    if args.benchmark == 'synth':
        do_BO_synth(args)
    elif args.benchmark == 'satellite':
        do_BO_satellite(args)
    else:
        raise ValueError
