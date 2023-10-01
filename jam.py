import scipy.stats as st
import numpy as np

from hbo import Theta, Eta, gamma_struct

def prepare_plot(ax):
    ax.set_xlabel('Sugar, g')
    ax.set_ylabel('Tastiness score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)

colours = {
    'Apples': 'limegreen',
    'Cherries': 'darkred',
    'Blueberries': 'darkslateblue',
    'Currants': 'rebeccapurple',
    'Cloudberries': 'orange',
    'Gooseberries': 'darkolivegreen',
    'Meta': 'black',
}


class eta_jam(Eta):
    def __init__(self, shape_mean, scale_mean, shape_std, scale_std):
        
        parameters = {'mean': gamma_struct(shape_mean, scale_mean),
                      'std': gamma_struct(shape_std, scale_std)}
        
        super(eta_jam, self).__init__(
            parameters=parameters,
        )     
        
    def sample_theta(self, name='new'):
        mean = np.random.gamma(shape=self.parameters['mean'].shape, 
                               scale=self.parameters['mean'].scale)
        std = np.random.gamma(shape=self.parameters['std'].shape,
                              scale=self.parameters['std'].scale)
        return jam_scorer(mean, std, name)
    
    
class jam_scorer(Theta):
    def __init__(self, mean, std, name):
        
        super(jam_scorer, self).__init__(
            parameters={'mean': mean, 'std':std},
            name=name,
        )
        
    def mean(self):
        return self.parameters['mean']
    
    def std(self):
        return self.parameters['std']
        
    def score(self, X):
        return st.norm.pdf(X, loc=self.mean(), scale=self.std())
    
    def best_score(self):
        return st.norm.pdf(self.mean(), loc=self.mean(), scale=self.std())
    
    def joint_score(self, D):
        X, Px = D
        Qx = self.score(X)
        KL_est = np.sum([pp*np.log(pp/Qx[ii]) for ii, pp in enumerate(Px)])
        if KL_est < 0:
            return 0 # TODO: be more principled
        else:
            return 1/KL_est
    
    def new_season(self):
        self.update('mean', self.mean() - 50)
        
    def get_observations(self, N, sigmas=5):
        if sigmas is not None:
            locs = np.random.uniform(self.mean()-sigmas*self.std(), 
                                     self.mean()+sigmas*self.std(), N)
        else:
            locs = np.random.uniform(0, 1000, N)
        vals = self.score(locs)
        return locs, vals
