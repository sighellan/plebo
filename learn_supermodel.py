# Learn a shared set of GP hyperparameters for use as a baseline

import pickle
import numpy as np

from paramz.optimization import opt_lbfgsb

from synthetic import NOISE_VAR, gp_hypers
from optimisation import define_model, learn_model


problem_id = 'weak'
version = 'satellite'

# problem_id = '2023-08-25-14-43-26'
# version = 'synth'

if version == 'synth':
    task_id = '-task-0'
    problems = pickle.load(open('problems/synth_%s.p' %(
        problem_id+task_id), 'rb'))

    related = problems['related_tasks']

else:
    related = pickle.load(open(
        'candidates/satellite_%s_tune_subset.p' %problem_id, 'rb'))

class SuperModel():
    
    def __init__(self, model_list):
        self.model_list = model_list
        self.scaling = 1
        
    def _objective(self, xx):
        return self.scaling*np.mean([mm._objective(xx) for mm in self.model_list])
    
    def _grads(self, xx):
        return self.scaling*np.mean([mm._grads(xx) for mm in self.model_list], 0)

    def _objective_grads(self, xx):
        return (self._objective(xx), self._grads(xx))

    def optimize(self, xx_init):
        opt = opt_lbfgsb()
        opt.max_iters = 1000
        opt.opt(xx_init,
                f=self._objective,
                fp=self._grads,
                f_fp=self._objective_grads)
        return opt.x_opt
    
    def lengthscale(self):
        return self.model_list[0].rbf.lengthscale.values[0]
    
    def variance(self):
        return self.model_list[0].rbf.variance.values[0]
    
    def make_candidate(self, name='supermodel'):
        return gp_hypers(
            lengthscale=self.lengthscale(),
            signal_var=self.variance(),
            name=name)


models = []
for ii in range(len(related)):
    loc, obs = related[ii].locs, related[ii].obs
    model = define_model((loc, obs))
    model.likelihood.variance.constrain_fixed(NOISE_VAR)
    models.append(model)

super_model = SuperModel(models)
xx_init = models[0].optimizer_array
found = False
for tt in range(10):
    super_model.optimize(xx_init)
    if 0 == np.sum(np.isnan(super_model.model_list[0].param_array)):
        found = True
        break
    else:
        super_model.scaling *= 0.001 
        print(
            'Optimisation did not converge. Running again with scaled objective %s.' \
            %super_model.scaling)
        
if found: 
    cand = super_model.make_candidate()

    pickle.dump(cand, open('candidates/%s_%s_supermodel.p' %(version, problem_id), 
                           'wb'))
    print('Joint hp set found.')
    print(cand)
else:
    print('Did not find joint hp set.')