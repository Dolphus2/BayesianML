import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax import random, hessian, value_and_grad, vmap
import jax.scipy.stats as jstats
from jax.scipy.special import gammaln
import seaborn as snb
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm

from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax.scipy.stats import multivariate_normal as mvn

from jax import config
config.update("jax_enable_x64", True)

snb.set_style('darkgrid')
snb.set_theme(font_scale=1.25)
colors = snb.color_palette()

# ====================================== Activation Functions ==========================================

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def softplus(x):
    return jnp.log(1.0 + jnp.exp(x))

def relu(x):
    return jnp.maximum(0.0, x)

def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum()

# ====================================== Probability Distributions =====================================

def probit(x):
    """Gaussian CDF — used as a link function for binary classification."""
    return norm.cdf(x)

def gaussian_logpdf(x, mu, sigma):
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

def gaussian_pdf(x, mu, sigma):
    return jnp.exp(gaussian_logpdf(x, mu, sigma))

def bernoulli_logpmf(y, p):
    """Log PMF of Bernoulli: y in {0, 1}, p in (0, 1)."""
    return y * jnp.log(p) + (1 - y) * jnp.log(1 - p)

# Multivariate Normal
def mvn_logpdf(x, mu, Sigma):
    """Log PDF of multivariate normal N(mu, Sigma).

    arguments:
        x     -- D-vector
        mu    -- D-vector mean
        Sigma -- DxD covariance matrix
    """
    return mvn.logpdf(x, mu, Sigma)

def mvn_pdf(x, mu, Sigma):
    return jnp.exp(mvn_logpdf(x, mu, Sigma))

def mvn_sample(key, mu, Sigma, shape=()):
    """Sample from N(mu, Sigma). shape gives the batch shape of samples."""
    return random.multivariate_normal(key, mu, Sigma, shape=shape)

# Gamma — parameterised by shape (a) and scale. Mean = a * scale.
def gamma_logpdf(x, a, scale=1.0):
    """Log PDF of Gamma(a, scale). Mean = a * scale, variance = a * scale**2."""
    return jstats.gamma.logpdf(x, a=a, scale=scale)

def gamma_pdf(x, a, scale=1.0):
    return jnp.exp(gamma_logpdf(x, a, scale))

def gamma_sample(key, a, scale=1.0, shape=()):
    return random.gamma(key, a, shape=shape) * scale

# Beta
def beta_logpdf(x, a, b):
    """Log PDF of Beta(a, b). Support x in (0, 1)."""
    return jstats.beta.logpdf(x, a=a, b=b)

def beta_pdf(x, a, b):
    return jnp.exp(beta_logpdf(x, a, b))

def beta_sample(key, a, b, shape=()):
    return random.beta(key, a, b, shape=shape)

# Poisson
def poisson_logpmf(k, lam):
    """Log PMF of Poisson(lam). k must be a non-negative integer."""
    return jstats.poisson.logpmf(k, mu=lam)

def poisson_pmf(k, lam):
    return jnp.exp(poisson_logpmf(k, lam))

def poisson_sample(key, lam, shape=()):
    return random.poisson(key, lam=lam, shape=shape)

# Binomial
def binomial_logpmf(k, n, p):
    """Log PMF of Binomial(n, p).

    arguments:
        k -- number of successes (non-negative integer)
        n -- number of trials (positive integer)
        p -- success probability in (0, 1)
    """
    log_coeff = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return log_coeff + k * jnp.log(p) + (n - k) * jnp.log(1 - p)

def binomial_pmf(k, n, p):
    return jnp.exp(binomial_logpmf(k, n, p))

def binomial_sample(key, n, p, shape=()):
    """Sample from Binomial(n, p) by summing n Bernoulli trials."""
    return jnp.sum(random.bernoulli(key, p, shape=(*shape, n)), axis=-1)

# Dirichlet
def dirichlet_logpdf(x, alpha):
    """Log PDF of Dirichlet(alpha). x must be a probability vector summing to 1.

    arguments:
        x     -- K-vector in the probability simplex
        alpha -- K-vector of concentration parameters (all positive)
    """
    return jstats.dirichlet.logpdf(x, alpha=alpha)

def dirichlet_pdf(x, alpha):
    return jnp.exp(dirichlet_logpdf(x, alpha))

def dirichlet_sample(key, alpha, shape=()):
    return random.dirichlet(key, alpha, shape=shape)

# ====================================== Bayesian Linear Regression =====================================

def compute_posterior_w(Phi, t, alpha, beta):
    """
    Computes posterior p(w|t) of a linear Gaussian system

    Arguments:
        Phi:    NxM matrix of N observations with M features
        t:      N-vector of targets
        alpha:  prior precision (scalar)
        beta:   likelihood precision (scalar)

    Returns:
        m:      M-vector posterior mean
        S:      MxM posterior covariance
    """
    N, M = Phi.shape
    A = alpha * jnp.identity(M) + beta * Phi.T @ Phi
    m = beta * jnp.linalg.solve(A, Phi.T) @ t
    S = jnp.linalg.inv(A)
    return m, S

def marginal_likelihood(Phi, t, alpha, beta):
    """Computes log marginal likelihood of a linear Gaussian system.

    Arguments:
        Phi:    NxM design matrix
        t:      N-vector of targets
        alpha:  prior precision (scalar)
        beta:   likelihood precision (scalar)

    Returns:
        log_Z:  log marginal likelihood (scalar)
    """
    N, M = Phi.shape
    m, S = compute_posterior_w(Phi, t, alpha, beta)
    Em = beta / 2 * jnp.sum((t - Phi @ m) ** 2) + alpha / 2 * jnp.sum(m ** 2)
    A = alpha * jnp.identity(M) + beta * Phi.T @ Phi
    return M / 2 * jnp.log(alpha) + N / 2 * jnp.log(beta) - Em - 0.5 * jnp.linalg.slogdet(A)[1] - N / 2 * jnp.log(2 * jnp.pi)

# ====================================== Laplace Approximation =========================================

def laplace_approximation(log_target, w0):
    """Computes the Laplace approximation of a log-target density.

    Arguments:
        log_target:  callable, log of the (unnormalized) target density
        w0:          initial parameter vector (1D array)

    Returns:
        m:  MAP estimate
        S:  covariance matrix from the Hessian at the MAP
    """
    obj = lambda w: -log_target(w)
    result = minimize(value_and_grad(obj), w0, jac=True)
    if not result.success:
        print('Warning: Laplace approximation optimization failed!')
    m = jnp.array(result.x)
    S = jnp.linalg.inv(hessian(obj)(m))
    return m, S

# ==================================== General Minimal Plotting Functions ======================================================

def _plot_data(ax, Xtrain, ytrain):
    ax.plot(Xtrain, ytrain, 'k.', markersize=12, label='Data')
    ax.grid(True)
    ax.set_xlabel('Input $x$')
    ax.set_ylabel('Response $y$')
    ax.legend()

def plot_data(Xtrain, ytrain):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    _plot_data(ax, Xtrain, ytrain)

def plot_contour(ax, f, x1s, x2s, num_contours=10, transform=lambda x: x,
                 color='b', alpha=1.0, xlabel='$x_1$', ylabel='$x_2$', title=''):
    """Plot contour lines of a 2D function f(X1, X2).

    arguments:
        ax           -- matplotlib axes
        f            -- callable f(X1, X2) where X1, X2 are NxM meshgrids; returns NxM array
        x1s          -- 1D array of values along the first axis (N points)
        x2s          -- 1D array of values along the second axis (M points)
        num_contours -- number of contour levels
        transform    -- optional transform applied to f values before plotting (e.g. jnp.exp)
        color        -- contour line color
        alpha        -- contour line opacity
        xlabel       -- x-axis label
        ylabel       -- y-axis label
        title        -- axes title
    """
    X1, X2 = jnp.meshgrid(x1s, x2s, indexing='ij')
    Z = transform(f(X1, X2))
    ax.contour(x1s, x2s, Z.T, num_contours, colors=color, alpha=alpha)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title, fontweight='bold')

def plot_heatmap(fig, ax, f, x1s, x2s, transform=lambda x: x,
                 xlabel='$x_1$', ylabel='$x_2$', title='', cmap='viridis'):
    """Plot a 2D function f(X1, X2) as a heatmap with a colorbar on the right.

    arguments:
        fig          -- matplotlib figure (needed for the colorbar)
        ax           -- matplotlib axes
        f            -- callable f(X1, X2) where X1, X2 are NxM meshgrids; returns NxM array
        x1s          -- 1D array of values along the first axis (N points)
        x2s          -- 1D array of values along the second axis (M points)
        transform    -- optional transform applied to f values before plotting (e.g. jnp.exp)
        xlabel       -- x-axis label
        ylabel       -- y-axis label
        title        -- axes title
        cmap         -- matplotlib colormap name
    """
    X1, X2 = jnp.meshgrid(x1s, x2s, indexing='ij')
    Z = transform(f(X1, X2))
    im = ax.pcolormesh(x1s, x2s, Z.T, shading='auto', cmap=cmap)
    add_colorbar(im, fig, ax)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title, fontweight='bold')

# ====================================== Gaussian Processes ============================================

def generate_samples(key, m, K, num_samples, jitter=0):
    """ returns M samples from an Gaussian process with mean m and kernel matrix K. The function generates num_samples of z ~ N(0, I) and transforms them into f  ~ N(m, K) via the Cholesky factorization.


    arguments:
        key              -- jax random key for controlling the random number generator
        m                -- mean vector (shape (N,))
        K                -- kernel matrix (shape NxN)
        num_samples      -- number of samples to generate (positive integer)
        jitter           -- amount of jitter (non-negative scalar)

    returns
        f_samples        -- a numpy matrix containing the samples of f (shape N x num_samples)
    """

    zs = random.normal(key, shape=(len(K), num_samples))
    N = len(K)
    L = jnp.linalg.cholesky(K + jitter*jnp.identity(N))
    f_samples = m[:, None] + jnp.dot(L, zs)
    assert f_samples.shape == (len(K), num_samples), f"The shape of f_samples appears wrong. Expected shape ({len(K)}, {num_samples}), but the actual shape was {f_samples.shape}. Please check your code. "
    return f_samples

@dataclass
class Hyperparameters(object):
    kappa:          float = 1.0 # magnitude, positive scalar (default=1.0)
    lengthscale:    float = 1.0 # characteristic lengthscale, positive scalar (default=1.0)
    sigma:          float = 1.0 # noise std. dev., positive scalar (default=1.0)

    def to_array(self):
        """ return hyperparameters as flat JaX-array (to be used later) """
        return jnp.array([self.kappa, self.lengthscale, self.sigma])

    @staticmethod
    def from_array(hyper_array):
        """ instantiates Hyperparameter object from flat JaX-array (or list) of hyperparameters (to be used later) """
        kappa, lengthscale, sigma = hyper_array
        return Hyperparameters(kappa, lengthscale, sigma)

    def __repr__(self):
        """ for reporting hyperparameter values """
        return f'Hyperparameters(kappa={self.kappa:3.2f}, lengthscale={self.lengthscale:3.2f}, sigma={self.sigma:3.2f})'

# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(tau, hyperparameters):
    return hyperparameters.kappa**2*jnp.exp(-0.5*tau**2/hyperparameters.lengthscale**2)

def matern12(tau, hyperparameters):
    return hyperparameters.kappa**2*jnp.exp(-tau/hyperparameters.lengthscale)

def matern32(tau, hyperparameters):
    return hyperparameters.kappa**2*(1 + jnp.sqrt(3)*tau/hyperparameters.lengthscale)*jnp.exp(-jnp.sqrt(3)*tau/hyperparameters.lengthscale)

class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun):
        """
            the argument kernel_fun must be a function of two arguments kernel_fun(||tau||, hyperparameters), e.g.
            squared_exponential = lambda tau, hyper: hyper.kappa**2*np.exp(-0.5*tau**2/hyper.lengthscale**2).
        """
        self.kernel_fun = kernel_fun

    def construct_kernel(self, X1, X2, hyperparameters, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun

        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix or None
            hyperparameters -- Hyperparameter object compatible with self.kernel_fun function
            jitter          -- non-negative scalar

        returns
            K               -- NxM matrix
        """

        N, M = X1.shape[0], X2.shape[0]
        dists = jnp.sqrt(jnp.sum((jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0))**2, axis=-1))
        K = self.kernel_fun(dists, hyperparameters)

        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter*jnp.identity(len(X1))
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K

class Kernel(object):
    """General kernel, not restricted to stationary or isotropic functions.

    The kernel_fun must be a function of (x1, x2, hyperparameters) where x1 and x2
    are individual D-dimensional input vectors and the return value is a scalar.

    Example — linear kernel:
        Kernel(lambda x1, x2, hyper: hyper.kappa**2 * jnp.dot(x1, x2))

    Example — polynomial kernel of degree p:
        Kernel(lambda x1, x2, hyper: (hyper.kappa + jnp.dot(x1, x2))**hyper.lengthscale)
    """

    def __init__(self, kernel_fun):
        """
        arguments:
            kernel_fun  -- function kernel_fun(x1, x2, hyperparameters) where
                           x1 (shape (D,)) and x2 (shape (D,)) are input vectors.
                           Must return a scalar kernel value k(x1, x2).
        """
        self.kernel_fun = kernel_fun

    def construct_kernel(self, X1, X2, hyperparameters, jitter=1e-8):
        """Compute and return the NxM kernel matrix between X1 (NxD) and X2 (MxD).

        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix
            hyperparameters -- Hyperparameter object compatible with self.kernel_fun
            jitter          -- non-negative scalar, added to diagonal when X1 == X2

        returns
            K               -- NxM matrix
        """
        N, M = X1.shape[0], X2.shape[0]

        K = vmap(lambda x1: vmap(lambda x2: self.kernel_fun(x1, x2, hyperparameters))(X2))(X1)

        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter * jnp.identity(N)
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K


class GaussianProcessRegression(object):

    def __init__(self, X, y, kernel, hyperparameters, jitter=1e-8):
        """
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values
            kernel           -- must be instance of StationaryIsotropicKernel or Kernel
            jitter           -- non-negative scaler
            hyperparameters  -- Hyperparameter object containing kernel hyperparameters and noise std. dev.
        """
        self.X = X
        self.y = y
        self.N = len(X)
        self.kernel = kernel
        self.jitter = jitter
        self.set_hyperparameters(hyperparameters)
        self.check_dimensions()

    def check_dimensions(self):
        N, D = self.X.shape
        assert self.X.ndim == 2, f"The variable X must be of shape (N, D), however, the current shape is: {self.X.shape}"
        assert self.y.ndim == 2, f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        assert self.y.shape == (N, 1), f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"

    def set_hyperparameters(self, hyper):
        self.hyperparameters = hyper

    def posterior_samples(self, key, Xstar, num_samples):
        """
            generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar

            Arguments:
                key              -- jax random key for controlling the random number generator
                Xstar            -- PxD prediction points

            returns:
                f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """

        mu, Sigma = self.predict_f(Xstar)
        f_samples = generate_samples(key, mu.ravel(), Sigma, num_samples)

        assert (f_samples.shape == (len(Xstar), num_samples)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, {num_samples}), but actual shape was {f_samples.shape}. Please check implementation"
        return f_samples

    def predict_y(self, Xstar):
        """ returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)

        Arguments:
        Xstar            -- PxD prediction points

        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        mu, Sigma = self.predict_f(Xstar)
        Sigma = Sigma + self.hyperparameters.sigma**2 * jnp.identity(len(mu))

        return mu, Sigma

    def predict_f(self, Xstar):
        """ returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)

        Arguments:
        Xstar            -- PxD prediction points

        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        k = self.kernel.construct_kernel(Xstar, self.X, self.hyperparameters, jitter=self.jitter)
        K = self.kernel.construct_kernel(self.X, self.X, self.hyperparameters, jitter=self.jitter)
        Kstar = self.kernel.construct_kernel(Xstar, Xstar, self.hyperparameters, jitter=self.jitter)

        C = K + self.hyperparameters.sigma**2*jnp.identity(len(self.X))

        mu = jnp.dot(k, jnp.linalg.solve(C, self.y))
        Sigma = Kstar - jnp.dot(k, jnp.linalg.solve(C, k.T))

        assert (mu.shape == (len(Xstar), 1)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, 1), but actual shape was {mu.shape}. Please check implementation"
        assert (Sigma.shape == (len(Xstar), len(Xstar))), f"The shape of the posterior Sigma seems wrong. Expected ({len(Xstar)}, {len(Xstar)}), but actual shape was {Sigma.shape}. Please check implementation"

        return mu, Sigma

    def log_marginal_likelihood(self, hyperparameters):
        """
            evaluate the log marginal likelihood p(y) given the hyperparaemters

            Arguments:
                hyperparameters  -- Hyperparameter object containing kernel hyperparameters and noise std. dev.
            """

        K = self.kernel.construct_kernel(self.X, self.X, hyperparameters)
        C = K + hyperparameters.sigma**2*jnp.identity(self.N)

        L = jnp.linalg.cholesky(C)
        v = jnp.linalg.solve(L, self.y)

        logdet_term = jnp.sum(jnp.log(jnp.diag(L)))
        quad_term =  0.5*jnp.sum(v**2)
        const_term = -0.5*self.N*jnp.log(2*jnp.pi)

        return const_term - logdet_term - quad_term

def optimize_marginal_likelihood(gp, hyperparameters_init, verbose=True):
    """ Optimize log marginal likelihood with gradient-based methods """

    def objective(log_hyperparam_array):
        hyperparams_array = jnp.exp(log_hyperparam_array)
        hyper = Hyperparameters.from_array(hyperparams_array)
        return -gp.log_marginal_likelihood(hyper)

    log_hyper_array = jnp.log(hyperparameters_init.to_array())
    res = minimize(value_and_grad(objective), log_hyper_array, jac=True)

    if not res.success:
        print('Warning: optimization failed!')

    log_hyper_hat = res.x
    hyper = Hyperparameters.from_array(jnp.exp(log_hyper_hat))

    if verbose:
        print('Result of optimization:', hyper)

    return hyper

# ==================================== Gaussian Process Plotting Functions ======================================================

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

def plot_kernel(X, K, hyper, key, num_samples):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    im = ax[0].pcolormesh(X.flatten(), X.flatten(), K, shading='auto')
    ax[0].set(xlabel='Input $x$', ylabel="Input $x'$", title=f"Kernel function $k(x, x')$ for $\\kappa = {hyper.kappa:2.1f}$ and $\\ell$ = {hyper.lengthscale:2.1f}")
    ax[0].grid(False)
    ax[0].set_aspect('equal')
    add_colorbar(im, fig, ax[0])

    m = jnp.zeros(len(X))
    f_samples = generate_samples(key, m, K, num_samples=num_samples, jitter=1e-8)
    ax[1].plot(X, f_samples, alpha=0.75, linewidth=3);
    ax[1].grid(True)
    ax[1].set(xlabel='$x$', ylabel='$f(x)$', title='Samples from the Gaussian process');


def _plot_with_uncertainty(ax, Xp, gp, color='r', color_samples='b', title="", num_samples=0, seed=0):

    mu, Sigma = gp.predict_y(Xp)
    mean, std = mu.ravel(), jnp.sqrt(jnp.diag(Sigma))

    key = random.PRNGKey(seed)

    ax.plot(Xp, mean, color=color, label='Mean')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')

    if num_samples > 0:
        fs = gp.posterior_samples(key, Xp, num_samples)
        ax.plot(Xp, fs[:,0], color=color_samples, alpha=.25, label="$f(x)$ samples")
        ax.plot(Xp, fs[:, 1:], color=color_samples, alpha=.25)
    ax.set_title(title)

def plot_with_uncertainty(kernel, hyper, Xtrain, ytrain, Xstar):
    gp_prior = GaussianProcessRegression(jnp.zeros((0, 1)), jnp.zeros((0, 1)), kernel, hyper)
    gp_post = GaussianProcessRegression(Xtrain, ytrain, kernel, hyper)
    fig, ax = plt.subplots(1, 2, figsize=(25, 6))
    _plot_with_uncertainty(ax[0], Xstar, gp_prior, title='Prior predictive distribution', num_samples=30)
    _plot_with_uncertainty(ax[1], Xstar, gp_post, title='Posterior predictive distribution', num_samples=30)
    for i in range(2):
        _plot_data(ax[i], Xtrain, ytrain)
        ax[i].legend(loc='lower center', ncol=4)
