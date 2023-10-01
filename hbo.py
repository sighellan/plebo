from typing import Dict
import copy
import numpy as np
import scipy.stats as st

class Theta():
    def __init__(self, parameters: Dict, name):
        self.parameters = parameters
        self.name = name
        
    def __str__(self):
        return '%s: '%self.name+ ','.join(['(%s: %s)' %(key, self.parameters[key]) for key in self.parameters])
    
    def joint_score(self, D):
        raise NotImplementedError
        
    def update(self, key, value):
        self.parameters[key] = value


class gamma_struct():
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        
    def __str__(self):
        return '%s, %s' %(self.shape, self.scale)

class Eta():
    def __init__(self, parameters: Dict):
        self.parameters = parameters # dict of str to gamma_struct
        
    def __str__(self):
        return ','.join(['(%s: %s,%s)' %(key, self.parameters[key].shape, self.parameters[key].scale) for key in self.parameters])
        
    def valid(self):
        for key in self.parameters:
            if self.parameters[key].shape <= 0:
                return False
            if self.parameters[key].scale <= 0:
                return False
        return True
        
    def score(self, theta):
        if not self.valid():
            return 0
        
        contributions = [st.gamma.pdf(theta.parameters[key], loc=0,
                      a=self.parameters[key].shape,
                      scale=self.parameters[key].scale
                     )
            for key in self.parameters]
        return np.prod(contributions)

    
    def joint_score(self, theta_list):
        return np.prod([self.score(theta) for theta in theta_list])
    
    def sample_theta(self):
        raise NotImplementedError
    
    def score_param(self, par_name, T):
        return st.gamma.pdf(
            T, loc=0, a=self.parameters[par_name].shape,
            scale=self.parameters[par_name].scale
        )

    def copy(self):
        return Eta(copy.deepcopy(self.parameters))
