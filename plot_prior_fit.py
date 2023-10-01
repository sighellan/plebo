import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import sys

sys.path.append('mcmc/')

from synthetic import eta_synthetic, gp_hypers

candidate_id = '2023-08-25-14-43-26'
problem_id = candidate_id
J = 10

mcmc = pickle.load(open('mcmc/mcmc_%s.p' %candidate_id, 'rb'))

candidates = pickle.load(open(
    'candidates/numpyro_synthetic_%s.p' %candidate_id, 'rb'))

lengthscales = [cc.lengthscale() for cc in candidates]
sig_vars = [cc.signal_var() for cc in candidates]

problems = []
true_lengthscales = []
true_signal_vars = []
prob_new = pickle.load(open(
    'problems/synth_%s-task-0.p' %problem_id, 'rb'))

for ii in range(J):
    problems.append(prob_new['related_tasks_true'][ii])
    true_lengthscales.append(problems[ii].lengthscale())
    true_signal_vars.append(problems[ii].signal_var())
    
gamma_prior = json.load(open(
    'candidates/synth_%s_gamma_prior.json' %candidate_id, 'r'))


eta = eta_synthetic(gamma_prior['ll_shape'], 
                    1/gamma_prior['ll_rate'], 
                    gamma_prior['ss_shape'],
                    1/gamma_prior['ss_rate'])


# Check which mcmc samples are filtered out 
N_samples = len(mcmc.get_samples()['phi_l'])
ii_qualify = []
for ii in range(N_samples):
    score = 0
    for jj in range(J):
        D = (prob_new['related_tasks'][jj].locs,
             prob_new['related_tasks'][jj].obs)
        theta = gp_hypers(
            mcmc.get_samples()['l_scale'][ii,jj],
            mcmc.get_samples()['s_var'][ii,jj],
            '%s,%s' %(ii,jj)
        )
        score += np.log(theta.joint_score(D))
    if score > -np.inf:
        ii_qualify.append(ii)

print('%s/%s points have non-zero likelihood' %(len(ii_qualify), N_samples))
ii_qualify = np.array(ii_qualify)
###############################

col_dict = {
    'True': 'C0',
    'Learned': 'C1',
}

def plot_prior_fit(hyp_dict, bins, X, col_dict, true_eta, learned_eta):

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(hyp_dict['cand_vals'], density=True, color=col_dict['Learned'],
            bins=bins, alpha=0.5);
    ax.hist(hyp_dict['true_vals'], density=True, color=col_dict['True'],
            bins=bins, alpha=0.5);

    ax.plot(X, true_eta.score_param(hyp_dict['name'], X),
             label='True', c=col_dict['True'], linewidth=3)
    ax.plot(X, learned_eta.score_param(hyp_dict['name'], X),
            label='Learned', c=col_dict['Learned'], linewidth=3)
#     ax.set_title(hyp_dict['title'])

    ax.legend(frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.set_ylabel('Density')
    ax.set_xlabel(hyp_dict['title'])
    
    fig.savefig('Figures/Prior_fit_%s_%s.png' %(problem_id, hyp_dict['name']),
               bbox_inches='tight', dpi=400)

X = np.arange(0, 0.301, 0.001)
bins = np.arange(0, 0.31, 0.01)

lengthscale_dict = {
    'cand_vals': lengthscales,
    'true_vals': true_lengthscales,
    'name': 'length',
    'title': 'Lengthscale',
    'key': 'l_scale',
}

T = np.arange(0, 20, 0.01)
bins_sig = np.arange(0, 20, 0.7)

sigvar_dict = {
    'cand_vals': sig_vars,
    'true_vals': true_signal_vars,
    'name': 'signal',
    'title': 'Signal variance',
    'key': 's_var',
}

plot_prior_fit(lengthscale_dict, bins, X, col_dict,
               prob_new['true_eta'], eta)

plot_prior_fit(sigvar_dict, bins_sig, T, col_dict,
               prob_new['true_eta'], eta)



def plot_individual_fits(mcmc, hyp_dict, ii_qualify, bins, col_dict):
    fig, axs = plt.subplots(4, 3)
#     fig.suptitle(hyp_dict['title'])

    for ii in range(10):
        if ii == 0:
            ax = axs[0, 1]
        else:
            ax = axs[(ii+2)//3, (ii+2)%3]

        hyp_ii = np.array(mcmc.get_samples()
                          [hyp_dict['key']])[ii_qualify, ii]
        
        ax.hist(hyp_dict['true_vals'][ii], density=True,
                bins=bins, alpha=0.5,
               color=col_dict['True'], label='True');
        ax.hist(hyp_ii, density=True, bins=bins, alpha=0.5,
               color=col_dict['Learned'], label='Learned');

        ax.set_title('Tune %s' %(ii+1), fontsize=10, y=0.8)

    axs[0,1].legend(
        frameon=False, loc='center left', 
        bbox_to_anchor=(0.4, 0.5))
    
    for ii in range(4):
        for jj in range(3):
            ax = axs[ii,jj]
            ax.spines[['right', 'top']].set_visible(False)
            if ii != 3:
                ax.set_xticks([])
            if jj != 0 and ii  != 0:
                ax.set_yticks([])

    for ax in [axs[0,0], axs[0,2]]:
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
    
    axs[2, 0].set_ylabel('Density')
    axs[3, 1].set_xlabel(hyp_dict['title'])

    fig.savefig('Figures/Individual_fits_%s_%s.png' %(problem_id, hyp_dict['name']),
               bbox_inches='tight', dpi=400)
    
plot_individual_fits(mcmc, lengthscale_dict, ii_qualify, bins, col_dict)

plot_individual_fits(mcmc, sigvar_dict, ii_qualify, bins_sig, col_dict)

score_bins = np.logspace(np.log10(1e-25), np.log10(1e-10), 50)



def plot_individual_scores(mcmc, ii_qualify, score_bins, col_dict, prob_new):
    fig, axs = plt.subplots(4, 3)

    for ii in range(10):
        if ii == 0:
            ax = axs[0, 1]
        else:
            ax = axs[(ii+2)//3, (ii+2)%3]
        
        lscales = mcmc.get_samples()['l_scale'][ii_qualify,ii]
        sig_vars = mcmc.get_samples()['s_var'][ii_qualify,ii]
        
        locs = prob_new['related_tasks'][ii].locs
        obs = prob_new['related_tasks'][ii].obs
        D_ii = (locs, obs)
        
        true_score = prob_new['related_tasks_true'][ii].joint_score(D_ii)
        
        scores = [
            gp_hypers(lscales[cc], sig_vars[cc], name='mcmc').joint_score(D_ii)
            for cc in range(len(lscales))
        ]

        ax.hist(true_score, bins=score_bins, alpha=0.5,
                color=col_dict['True'], label='True');
        ax.hist(scores, weights=np.ones(len(scores)) / len(scores),
                bins=score_bins, alpha=0.5,
               color=col_dict['Learned'], label='Learned');
        ax.set_xscale('log')



        ax.set_title('Tune %s' %(ii+1), fontsize=10, y=0.8)

    axs[1,2].legend(
        frameon=False, loc='center left', 
        bbox_to_anchor=(0.05, 0.5))
    
    for ii in range(4):
        for jj in range(3):
            ax = axs[ii,jj]
            ax.spines[['right', 'top']].set_visible(False)
            if ii != 3:
                ax.set_xticks([])
            if jj != 0 and ii  != 0:
                ax.set_yticks([])

    for ax in [axs[0,0], axs[0,2]]:
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
    axs[2, 0].set_ylabel('Fraction')
    axs[3, 1].set_xlabel('Likelihood')
        
    fig.savefig('Figures/Individual_scores_%s.png' %problem_id,
               bbox_inches='tight', dpi=400)
    
plot_individual_scores(mcmc, ii_qualify, score_bins, col_dict, prob_new)