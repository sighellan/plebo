import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import sys

sys.path.append('../')
from synthetic import NOISE_VAR

def kernel(N, X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    # based on # https://github.com/pyro-ppl/numpyro/blob/master/examples/gp.py

    diff = jnp.tile(X, (N, 1, 1)) - jnp.tile(Z, (N, 1, 1)).transpose(1, 0, 2)
    deltaXsq = jnp.sum(jnp.power(diff / length, 2.0), 2)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

def prior_learning(J, N, X, kernel, Y=None):
    # bottom layer: observations from J tasks
    # hyperparameters from J tasks
    # eta
    # we want to learn values of eta
    
    phi_l = numpyro.sample('phi_l', dist.Uniform(-10, 10))
    psi_l = numpyro.sample('psi_l', dist.Uniform(-10, 10))
    phi_s = numpyro.sample('phi_s', dist.Uniform(-10, 10))
    psi_s = numpyro.sample('psi_s', dist.Uniform(-10, 10))
    
    with numpyro.plate('J', J) as jj:
        l_scale = numpyro.sample('l_scale', dist.Gamma(jnp.exp(phi_l), jnp.exp(psi_l)),
                                 infer={"enumerate": "parallel"})

        s_var = numpyro.sample('s_var', dist.Gamma(jnp.exp(phi_s), jnp.exp(psi_s)),
                               infer={"enumerate": "parallel"})
        
        kk = jnp.array([kernel(N, X[jj], X[jj], s_var[jj], l_scale[jj], noise=NOISE_VAR) 
                        for jj in range(J)])
        
        numpyro.sample("Y", dist.MultivariateNormal(
            loc=jnp.zeros((J,N)),
            covariance_matrix=kk),
                       obs=Y)
