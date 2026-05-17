import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from jax import vmap
import numpy as np
from scipy.optimize import minimize



def gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((jnp.mean(x, axis=1) - jnp.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2

def compute_effective_sample_size_single_param(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    m_chains, n_iters = x.shape

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def compute_effective_sample_size(chains_):
    """ computes the effective sample size for each parameter in a MCMC simulation. 
        The function expects the argument chain to be a numpy array of shape (num_chains x num_samples x num_params)
        and it return a numpy of shape (num_params) containing the S_eff estimates for each parameter
    """

    # force numpy
    chains = np.array(chains_)

    # get dimensions
    num_chains, num_samples, num_params = chains.shape

    # estimate sample size for each parameter
    S_eff = np.array([compute_effective_sample_size_single_param(chains[:, :, idx_param]) for idx_param in range(num_params)])

    # return
    return S_eff


def compute_Rhat(chains):
    """ Computes the Rhat convergence diagnostic for each parameter in a MCMC simulation. 
        The function expects the argument chain to be a numpy array of shape (num_chains x num_samples x num_params)
        and it return a numpy of shape (num_params) containing the Rhat estimates for each parameter
    """

    # get dimensions
    num_chains, num_samples, num_params = chains.shape

    # make subchains by splitting each chains in half
    sub_chains = []
    half_num_samples = int(0.5*num_samples)
    for idx_chain in range(num_chains):
        sub_chains.append(chains[idx_chain, :half_num_samples, :])
        sub_chains.append(chains[idx_chain, half_num_samples:, :])

    # count number of sub chains
    num_sub_chains = len(sub_chains)
        
    # compute mean and variance of each subchain
    chain_means = np.array([np.mean(s, axis=0) for s in sub_chains])                                             # dim: num_sub_chains x num_params
    chain_vars = np.array([1/(num_samples-1)*np.sum((s-m)**2, 0) for (s, m) in zip(sub_chains, chain_means)])    # dim: num_sub_chains x num_params

    # compute between chain variance
    global_mean = np.mean(chain_means, axis=0)                                                                   # dim: num_params
    B = num_samples/(num_sub_chains-1)*np.sum((chain_means - global_mean)**2, axis=0)                            # dim: num_params

    # compute within chain variance
    W = np.mean(chain_vars, 0)                                                                                   # dim: num_params                                                          

    # compute estimator and return
    var_estimator = (num_samples-1)/num_samples*W + (1/num_samples)*B                                            # dim: num_params 
    Rhat = np.sqrt(var_estimator/W)
    return Rhat



def combine_chains(chains):
    return chains.flatten()


def eval_density_grid(density_fun, x_min=-5, x_max=5, P=100):
    """ evaluate density_fun on 2D grid from x_min to x_max. Return grid and density values evaluate on grid """
    x_grid = jnp.linspace(x_min, x_max, 100)
    X1, X2 = jnp.meshgrid(x_grid, x_grid)
    XX = jnp.column_stack((X1.ravel(), X2.ravel()))
    return x_grid, density_fun(XX).reshape((P, P))


def check_leapfrog_implementation(theta0, nu0, theta, nu, theta_trajectory, nu_trajectory, hamiltonian_fun, num_leapfrog_steps):

    # check dimensions of theta and nu
    assert theta0.shape == theta.shape, f"Shape of theta must be the same as theta0. The expected shape of theta is {theta0.shape}, but the actual shape is {theta.shape}"
    assert nu0.shape == nu.shape, f"Shape of nu must be the same as nu0. The expected shape of nu is {nu0.shape}, but the actual shape is {nu.shape}"
    print('Dimensions of theta and nu: OK.')

    # check shapes of trajectories
    assert theta_trajectory.shape == (num_leapfrog_steps+1, 2), f"Shape of theta_trajectory was expected to be ({num_leapfrog_steps}, 2), but the actual shape is {theta_trajectory.shape}"
    assert nu_trajectory.shape == (num_leapfrog_steps+1, 2), f"Shape of nu_trajectory was expected to be ({num_leapfrog_steps}, 2), but the actual shape is {nu_trajectory.shape}"
    print('Shapes of trajectories: OK.')

    assert jnp.abs(hamiltonian_fun(theta0, nu0) - hamiltonian_fun(theta, nu)) < 0.1, f"The value of the Hamiltonian should be approximately conserved (|diff| < 0.1), the actual difference was H(theta0, nu0) - H(theta, nu) = {hamiltonian_fun(theta0, nu0) - hamiltonian_fun(theta, nu):.3f}" 
    print('Conservation of Hamiltonian: OK')

    print('Sanity check of leapfrog implementation: OK')



def plot_summary(ax, x, s, interval=95, num_samples=100, sample_color='k', sample_alpha=0.4, interval_alpha=0.25, color='r', legend=True, title="", plot_mean=True, plot_median=False, label="", seed=0):
    
    b = 0.5*(100 - interval)
    
    lower = np.percentile(s, b, axis=0).T
    upper = np.percentile(s, 100-b, axis=0).T
    
    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        lab = 'Median'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), median, label=lab, color=color, linewidth=4)
        
    if plot_mean:
        mean = np.mean(s, axis=0).T
        lab = 'Mean'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), mean, '--', label=lab, color=color, linewidth=4)
    ax.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=interval_alpha, label='%d%% Interval' % interval)    
    
    if num_samples > 0:
        np.random.seed(seed)
        idx_samples = np.random.choice(range(len(s)), size=num_samples, replace=False)
        ax.plot(x, s[idx_samples, :].T, color=sample_color, alpha=sample_alpha);
    
    if legend:
        ax.legend(loc='best')
        
    if len(title) > 0:
        ax.title(title, fontweight='bold')
        
def plot_predictions(ax, x, s, num_samples=100, sample_color='k', sample_alpha=0.4, color='r', legend=False, plot_median=False, plot_mean=True, seed=123, title=''):

    plot_summary(ax, x, s, color=color, interval=99, num_samples=0, interval_alpha=0.25, plot_mean=False, plot_median=False, legend=legend, seed=seed)
    plot_summary(ax, x, s, color=color, interval=95, num_samples=0, sample_alpha=0.1, interval_alpha=0.35, plot_mean=False, plot_median=False, legend=legend, seed=seed)
    plot_summary(ax, x, s, color=color, interval=75, interval_alpha=0.6, num_samples=num_samples, sample_alpha=sample_alpha, plot_mean=False, plot_median=False, legend=legend, seed=seed, sample_color=sample_color)
    
    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        ax.plot(x.ravel(), median, label='Median', color='k', linewidth=4, alpha=0.7)
        
    if plot_mean:
        mean = np.mean(s, axis=0).T
        ax.plot(x.ravel(), mean, '-', label='Mean', color='k', linewidth=4, alpha=0.7)
        
    if title:
        ax.set_title(title, fontweight='bold')

    if legend:
        ax.legend(loc='best')



class BayesianLinearRegression(object):
    
    def __init__(self, Phi, y, alpha=1., beta=1.):
        
        # store data and hyperparameters
        self.Phi, self.y = Phi, y
        self.N, self.D = Phi.shape
        self.alpha, self.beta = alpha, beta
        
        # compute posterior distribution
        self.m, self.S = self.compute_posterior(alpha, beta)
        self.log_marginal_likelihood = self.compute_marginal_likelihood(alpha, beta)

        # perform sanity check of shapes/dimensions
        self.check_dimensions()

    def check_dimensions(self):
        D = self.D
        assert self.y.shape == (self.N, 1), f"Wrong shape for data vector y.\n For N = {N}, the shape of y must be ({N}, 1), but the actual shape is {self.y.shape}"
        assert self.m.shape == (D, 1), f"Wrong shape for posterior mean.\nFor D = {D}, the shape of the posterior mean must be ({D}, 1), but the actual shape is {self.m.shape}"
        assert self.S.shape == (D, D), f"Wrong shape for posterior covariance.\nFor D = {D}, the shape of the posterior mean must be ({D}, {D}), , but the actual shape is {self.S.shape}"

    def compute_posterior(self, alpha, beta):
        """ computes the posterior N(w|m, S) and return m, S.
            Shape of m and S must be (D, 1) and (D, D), respectively  """
        
        #############################################
        # Insert your solution here
        #############################################
        
        # compute prior and posterior precision 
        inv_S0 = alpha*jnp.identity(self.D)
        A = inv_S0 + beta*(self.Phi.T@self.Phi)
        
        # compute mean and covariance 
        m = beta*jnp.linalg.solve(A, self.Phi.T)@self.y   # eq. (2) above
        S = jnp.linalg.inv(A)                             # eq. (1) above
        
        #############################################
        # End of solution
        #############################################
        return m, S
      
    def generate_prior_samples(self, key, num_samples):
        """ generate samples from the prior  """
        return random.multivariate_normal(key, jnp.zeros(len(self.m)), (1/self.alpha)*jnp.identity(len(self.m)), shape=(num_samples, ))
    
    def generate_posterior_samples(self, key, num_samples):
        """ generate samples from the posterior  """
        return random.multivariate_normal(key, self.m.ravel(), self.S, shape=(num_samples, ))
    
    def predict_f(self, Phi):
        """ computes posterior mean (mu_f) and variance (var_f) of f(phi(x)) for each row in Phi-matrix.
            If Phi is a [N, D]-matrix, then the shapes of both mu_f and var_f must be (N,)
            The function returns (mu_f, var_f)
        """
        mu_f = (Phi@self.m).ravel()   
        var_f = jnp.diag(Phi@self.S@Phi.T)   
        
        # check dimensions before returning values
        assert mu_f.shape == (Phi.shape[0],), "Shape of mu_f seems wrong. Check your implementation"
        assert var_f.shape == (Phi.shape[0],), "Shape of var_f seems wrong. Check your implementation"
        return mu_f, var_f
        
    def predict_y(self, Phi):
        """ returns posterior predictive mean (mu_y) and variance (var_y) of y = f(phi(x)) + e for each row in Phi-matrix.
            If Phi is a [N, D]-matrix, then the shapes of both mu_y and var_y must be (N,).
            The function returns (mu_y, var_y)
        """
        mu_f, var_f = self.predict_f(Phi)
        mu_y = mu_f                  
        var_y = var_f + 1/self.beta  

        # check dimensions before returning values
        assert mu_y.shape == (Phi.shape[0],), "Shape of mu_y seems wrong. Check your implementation"
        assert var_y.shape == (Phi.shape[0],), "Shape of var_y seems wrong. Check your implementation"
        return mu_y, var_y
        
    
    def compute_marginal_likelihood(self, alpha, beta):
        """ computes and returns log marginal likelihood p(y|alpha, beta) """
        inv_S0 = alpha*jnp.identity(self.D)
        A = inv_S0 + beta*(self.Phi.T@self.Phi)
        m = beta*jnp.linalg.solve(A, self.Phi.T)@self.y   # (eq. 3.53 in Bishop)
        S = jnp.linalg.inv(A)                             # (eq. 3.54 in Bishop)
        Em = beta/2*jnp.sum((self.y - self.Phi@m)**2) + alpha/2*jnp.sum(m**2)
        return self.D/2*jnp.log(alpha) + self.N/2*jnp.log(beta) - Em - 0.5*jnp.linalg.slogdet(A)[1] - self.N/2*jnp.log(2*jnp.pi)
         

    def optimize_hyperparameters(self):
        # optimizes hyperparameters using marginal likelihood
        theta0 = jnp.array((jnp.log(self.alpha), jnp.log(self.beta)))
        def negative_marginal_likelihood(theta):
            alpha, beta = jnp.exp(theta[0]), jnp.exp(theta[1])
            return -self.compute_marginal_likelihood(alpha, beta)

        result = minimize(value_and_grad(negative_marginal_likelihood), theta0, jac=True)

        # store new hyperparameters and recompute posterior
        theta_opt = result.x
        self.alpha, self.beta = jnp.exp(theta_opt[0]), jnp.exp(theta_opt[1])
        self.m, self.S = self.compute_posterior(self.alpha, self.beta)
        self.log_marginal_likelihood = self.compute_marginal_likelihood(self.alpha, self.beta)

