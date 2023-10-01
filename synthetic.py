from hbo import Theta, Eta, gamma_struct

import GPy
import numpy as np
import copy
import matplotlib.pyplot as plt

NOISE_VAR = 0.000001

class eta_synthetic(Eta):
    def __init__(self, shape_length, scale_length, shape_signal, scale_signal):
        
        parameters = {'length': gamma_struct(shape_length, scale_length),
                      'signal': gamma_struct(shape_signal, scale_signal)}
        
        super(eta_synthetic, self).__init__(
            parameters=parameters,
        )     
        
    def sample_theta(self, name='new'):
        length = np.random.gamma(shape=self.parameters['length'].shape, 
                               scale=self.parameters['length'].scale)
        signal = np.random.gamma(shape=self.parameters['signal'].shape,
                              scale=self.parameters['signal'].scale)
        return gp_hypers(length, signal, name)
    
class gp_hypers(Theta):
    def __init__(self, lengthscale, signal_var, name, feat_dims=2):
        
        # signal_var * np.exp(-0.5 * r**2)
        self.kernel = GPy.kern.RBF(
            input_dim=feat_dims, variance=signal_var, lengthscale=lengthscale)
        self.feat_dims = feat_dims
        
        super(gp_hypers, self).__init__(
            parameters={'length': lengthscale, 'signal': signal_var},
            name=name,
        )
    def __deepcopy__(self, memo):
        return gp_hypers(copy.deepcopy(self.lengthscale(), memo), 
                         copy.deepcopy(self.signal_var(), memo),
                         copy.deepcopy(self.name, memo))
            
    def lengthscale(self):
        return self.parameters['length']
    
    def signal_var(self):
        return self.parameters['signal']
    
    def sample_data(self, locs=None, N=5):
        if locs is None:
            locs = np.random.uniform(0, 1, (N,self.feat_dims))
        C = self.kernel.K(locs, locs)
        mu = np.zeros(len(locs))
        return (locs, np.random.multivariate_normal(mu, C, 1).T)

    def joint_score(self, D):
        loc, obs = D
        model = GPy.models.GPRegression(X=loc, Y=obs, kernel=self.kernel,
                                        noise_var=NOISE_VAR)
        return np.exp(model.log_likelihood())
    
    def predict(self, D, X):
        loc, obs = D
        model = GPy.models.GPRegression(X=loc, Y=obs, kernel=self.kernel,
                                        noise_var=NOISE_VAR)
        mean, var = model.predict(X)
        return mean, var

class gp_realisation():
    def __init__(self, locs, obs):
        assert np.shape(locs)[1] == 2
        assert np.shape(obs)[1] == 1
        self.locs = locs
        self.obs = obs
        
    def score(self, xx):
        # TODO: this doesn't verify that there is a match
        is_match = (np.sum(self.locs == xx, 1) == 2)
        idx = np.argmax(is_match)
        return self.obs[idx]

    def __equal__(self, other):
        if not (self.locs == other.locs).all():
            return False
        if not (self.obs == other.obs).all():
            return False
        return True

def my_imshow(vec_values, vmin=None, vmax=None, length=28, width=28):
    two_d = vec_values.reshape((length, width)).T
    cb = plt.imshow(two_d, origin='lower', extent=[0,1,0,1], vmin=vmin, vmax=vmax)
    return cb

def plot_status(eta, related_tasks, thetas, D, X, T):
    plt.figure()
    plt.plot(X, eta.score_param('length',X))
    plt.scatter([task.lengthscale() for task in related_tasks],
                [eta.score_param('length', task.lengthscale()) for task in related_tasks])
    
    plt.scatter([tt.lengthscale() for tt in thetas],
                [eta.score_param('length', tt.lengthscale()) for tt in thetas])
    plt.title('lengthscale')

    plt.figure()
    
    plt.plot(T, eta.score_param('signal', T))
    plt.scatter([task.signal_var() for task in related_tasks],
                [eta.score_param('signal', task.signal_var()) for task in related_tasks])
    
    plt.scatter([tt.signal_var() for tt in thetas],
                [eta.score_param('signal', tt.signal_var()) for tt in thetas])
    plt.title('signal_var')
    