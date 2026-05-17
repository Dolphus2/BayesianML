from __future__ import annotations

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, hessian, value_and_grad, vmap, jit, grad
from time import time
import jax.scipy.stats as jstats
from jax.scipy.special import gammaln
import seaborn as snb
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Any, Callable, Optional, Sequence, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax.scipy.stats import multivariate_normal as mvn

from jax import config
config.update("jax_enable_x64", True)

snb.set_style('darkgrid')
snb.set_theme(font_scale=1.25)
colors = snb.color_palette()

# ====================================== Utility Functions =============================================

# Calculate empirical confidence interval for a list of quantiles.
#jnp.percentile(random.normal(random.key(0), (1000,)), jnp.array([2.5, 97.5]))

class Grid2D(object):
    """ helper class for evaluating the function func on the grid defined by (alpha, beta) """

    def __init__(self, alphas: jax.Array, betas: jax.Array,
                 func: Callable[[jax.Array, jax.Array], jax.Array], name: str = "Grid2D") -> None:
        self.alphas = alphas
        self.betas = betas
        self.grid_size = (len(self.alphas), len(self.betas))
        self.alpha_grid, self.beta_grid = jnp.meshgrid(alphas, betas, indexing='ij')
        self.func = func
        self.name = name

        # evaluate function on each grid point
        self.values = self.func(self.alpha_grid[:, :, None], self.beta_grid[:, :, None]).squeeze()

    def plot_contours(self, ax: Axes, color: str = 'b', num_contours: int = 10,
                     f: Callable = lambda x: x, alpha: float = 1.0, title: Optional[str] = None) -> None:
        ax.contour(self.alphas, self.betas, f(self.values).T, num_contours, colors=color, alpha=alpha)
        ax.set(xlabel='$\\alpha$', ylabel='$\\beta$')
        ax.set_title(self.name, fontweight='bold')

    @property
    def argmax(self) -> tuple[jax.Array, jax.Array]: # approximation for small 2D problem. For larger use gradient based optimization.
        idx = jnp.argmax(self.values)
        alpha_idx, beta_idx = jnp.unravel_index(idx, self.grid_size)
        return self.alphas[alpha_idx], self.betas[beta_idx]

# ====================================== Activation Functions ==========================================

def sigmoid(x: jax.Array) -> jax.Array:
    return 1.0 / (1.0 + jnp.exp(-x))

def softplus(x: jax.Array) -> jax.Array:
    return jnp.log(1.0 + jnp.exp(x))

def relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(0.0, x)

def softmax(x: jax.Array) -> jax.Array:
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum()

# ====================================== Probability Distributions =====================================

def probit(x: jax.Array) -> np.ndarray:
    """Gaussian CDF — used as a link function for binary classification."""
    return norm.cdf(x)

def gaussian_logpdf(x: jax.Array, mu: float, sigma: float) -> jax.Array:
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

def gaussian_pdf(x: jax.Array, mu: float, sigma: float) -> jax.Array:
    return jnp.exp(gaussian_logpdf(x, mu, sigma))

def bernoulli_logpmf(y: jax.Array, p: jax.Array) -> jax.Array:
    """Log PMF of Bernoulli: y in {0, 1}, p in (0, 1)."""
    return y * jnp.log(p) + (1 - y) * jnp.log(1 - p)

# Multivariate Normal
def mvn_logpdf(x: jax.Array, mu: jax.Array, Sigma: jax.Array) -> jax.Array:
    """Log PDF of multivariate normal N(mu, Sigma).

    arguments:
        x     -- D-vector
        mu    -- D-vector mean
        Sigma -- DxD covariance matrix
    """
    return mvn.logpdf(x, mu, Sigma)

def mvn_pdf(x: jax.Array, mu: jax.Array, Sigma: jax.Array) -> jax.Array:
    return jnp.exp(mvn_logpdf(x, mu, Sigma))

def mvn_sample(key: jax.Array, mu: jax.Array, Sigma: jax.Array,
               shape: tuple[int, ...] = ()) -> jax.Array:
    """Sample from N(mu, Sigma). shape gives the batch shape of samples."""
    return random.multivariate_normal(key, mu, Sigma, shape=shape)

# Gamma — parameterised by shape (a) and scale. Mean = a * scale.
def gamma_logpdf(x: jax.Array, a: float, scale: float = 1.0) -> jax.Array:
    """Log PDF of Gamma(a, scale). Mean = a * scale, variance = a * scale**2."""
    return jstats.gamma.logpdf(x, a=a, scale=scale)

def gamma_pdf(x: jax.Array, a: float, scale: float = 1.0) -> jax.Array:
    return jnp.exp(gamma_logpdf(x, a, scale))

def gamma_sample(key: jax.Array, a: float, scale: float = 1.0,
                 shape: tuple[int, ...] = ()) -> jax.Array:
    return random.gamma(key, a, shape=shape) * scale

# Beta
def beta_logpdf(x: jax.Array, a: float, b: float) -> jax.Array:
    """Log PDF of Beta(a, b). Support x in (0, 1)."""
    return jstats.beta.logpdf(x, a=a, b=b)

def beta_pdf(x: jax.Array, a: float, b: float) -> jax.Array:
    return jnp.exp(beta_logpdf(x, a, b))

def beta_sample(key: jax.Array, a: float, b: float,
                shape: tuple[int, ...] = ()) -> jax.Array:
    return random.beta(key, a, b, shape=shape)

# Poisson
def poisson_logpmf(k: jax.Array, lam: float) -> jax.Array:
    """Log PMF of Poisson(lam). k must be a non-negative integer."""
    return jstats.poisson.logpmf(k, mu=lam)

def poisson_pmf(k: jax.Array, lam: float) -> jax.Array:
    return jnp.exp(poisson_logpmf(k, lam))

def poisson_sample(key: jax.Array, lam: float, shape: tuple[int, ...] = ()) -> jax.Array:
    return random.poisson(key, lam=lam, shape=shape)

# Binomial
def binomial_logpmf(k: jax.Array, n: int, p: float) -> jax.Array:
    """Log PMF of Binomial(n, p).

    arguments:
        k -- number of successes (non-negative integer)
        n -- number of trials (positive integer)
        p -- success probability in (0, 1)
    """
    log_coeff = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return log_coeff + k * jnp.log(p) + (n - k) * jnp.log(1 - p)

def binomial_pmf(k: jax.Array, n: int, p: float) -> jax.Array:
    return jnp.exp(binomial_logpmf(k, n, p))

def binomial_sample(key: jax.Array, n: int, p: float,
                    shape: tuple[int, ...] = ()) -> jax.Array:
    """Sample from Binomial(n, p) by summing n Bernoulli trials."""
    return jnp.sum(random.bernoulli(key, p, shape=(*shape, n)), axis=-1)

# Dirichlet
def dirichlet_logpdf(x: jax.Array, alpha: jax.Array) -> jax.Array:
    """Log PDF of Dirichlet(alpha). x must be a probability vector summing to 1.

    arguments:
        x     -- K-vector in the probability simplex
        alpha -- K-vector of concentration parameters (all positive)
    """
    return jstats.dirichlet.logpdf(x, alpha=alpha)

def dirichlet_pdf(x: jax.Array, alpha: jax.Array) -> jax.Array:
    return jnp.exp(dirichlet_logpdf(x, alpha))

def dirichlet_sample(key: jax.Array, alpha: jax.Array,
                     shape: tuple[int, ...] = ()) -> jax.Array:
    return random.dirichlet(key, alpha, shape=shape)

# ====================================== Bayesian Linear Regression =====================================

def compute_posterior_w(Phi: jax.Array, t: jax.Array,
                        alpha: float, beta: float) -> tuple[jax.Array, jax.Array]:
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

def marginal_likelihood(Phi: jax.Array, t: jax.Array,
                        alpha: float, beta: float) -> jax.Array:
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

def laplace_approximation(log_target: Callable[[jax.Array], jax.Array],
                          w0: jax.Array) -> tuple[jax.Array, jax.Array]:
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

# ====================================== MCMC and HMC =================================================

def metropolis(log_target: Callable[[jax.Array], jax.Array],
               num_params: int,
               tau: float,
               num_iter: int,
               theta_init: Optional[jax.Array] = None,
               seed: int = 0) -> jax.Array:
    """Metropolis-Hastings sampler with isotropic Gaussian proposal.

    arguments:
        log_target:  callable log_target(theta) where theta has shape (num_params,). Returns scalar.
        num_params:  dimension of the parameter space
        tau:         std. dev. of the Gaussian proposal distribution
        num_iter:    number of iterations
        theta_init:  initial parameter vector, shape (num_params,). Defaults to zeros.
        seed:        random seed

    returns:
        thetas:      MCMC samples, shape (num_iter+1, num_params)
    """
    key = random.PRNGKey(seed)
    theta = jnp.zeros(num_params) if theta_init is None else jnp.asarray(theta_init, dtype=float)
    log_p = log_target(theta)
    thetas = [theta]

    for _ in range(num_iter):
        key, key_prop, key_acc = random.split(key, 3)
        theta_star = theta + tau * random.normal(key_prop, shape=(num_params,))
        log_p_star = log_target(theta_star)
        if jnp.log(random.uniform(key_acc)) < jnp.minimum(0.0, log_p_star - log_p):
            theta, log_p = theta_star, log_p_star
        thetas.append(theta)

    return jnp.array(thetas)


def _build_hmc_fns(log_target: Callable[[jax.Array], jax.Array],
                   ) -> tuple[Callable, Callable, Callable]:
    """Build jit-compiled Hamiltonian, potential energy and its gradient from log_target.

    arguments:
        log_target:  callable log_target(theta) where theta has shape (D,). Returns scalar.

    returns:
        hamiltonian:    H(theta, nu) = -log_target(theta) + 0.5*||nu||^2
        potential:      U(theta)     = -log_target(theta)
        grad_potential: ∇U(theta)
    """
    @jit
    def potential(theta: jax.Array) -> jax.Array:
        return -log_target(theta)

    grad_potential = jit(grad(potential))

    @jit
    def hamiltonian(theta: jax.Array, nu: jax.Array) -> jax.Array:
        return potential(theta) + 0.5 * jnp.dot(nu, nu)

    return hamiltonian, potential, grad_potential


def leapfrog(theta: jax.Array,
             nu: jax.Array,
             grad_potential: Callable[[jax.Array], jax.Array],
             num_steps: int,
             step_size: float) -> tuple[jax.Array, jax.Array]:
    """Leapfrog integrator for Hamiltonian dynamics.

    arguments:
        theta:          current position, shape (D,)
        nu:             current momentum, shape (D,)
        grad_potential: callable ∇U(theta), shape (D,) → (D,)
        num_steps:      number of leapfrog steps L
        step_size:      step size ε

    returns:
        theta:  updated position, shape (D,)
        nu:     updated momentum, shape (D,)
    """
    for _ in range(num_steps):
        nu    = nu    - 0.5 * step_size * grad_potential(theta)
        theta = theta + step_size * nu
        nu    = nu    - 0.5 * step_size * grad_potential(theta)
    return theta, nu


def HMC(log_target: Callable[[jax.Array], jax.Array],
        num_iterations: int,
        theta0: jax.Array,
        num_leapfrog_steps: int = 10,
        step_size: float = 0.1,
        seed: int = 0,
        verbose: bool = True) -> jax.Array:
    """Hamiltonian Monte Carlo sampler.

    arguments:
        log_target:         callable log_target(theta) where theta has shape (D,). Returns scalar.
        num_iterations:     number of HMC iterations
        theta0:             initial position, shape (D,) or (1, D)
        num_leapfrog_steps: number of leapfrog steps L per iteration
        step_size:          leapfrog step size ε
        seed:               random seed
        verbose:            print progress and acceptance rate

    returns:
        thetas:             samples, shape (num_iterations+1, D)
    """
    hamiltonian, _, grad_potential = _build_hmc_fns(log_target)
    key = random.PRNGKey(seed)
    theta = jnp.asarray(theta0, dtype=float).ravel()
    D = len(theta)
    thetas = [theta]
    accept_count = 0
    t0 = time()

    for i in range(num_iterations):
        key, key_nu, key_acc = random.split(key, 3)
        nu = random.normal(key_nu, shape=(D,))
        theta_star, nu_star = leapfrog(theta, nu, grad_potential, num_leapfrog_steps, step_size)
        log_accept = jnp.minimum(0.0, -hamiltonian(theta_star, nu_star) + hamiltonian(theta, nu))
        if jnp.log(random.uniform(key_acc)) < log_accept:
            theta = theta_star
            accept_count += 1
        thetas.append(theta)
        if verbose and (i + 1) % max(1, num_iterations // 10) == 0:
            print(f'  {i+1:5d}/{num_iterations}  accept_rate={accept_count/(i+1):.3f}  t={time()-t0:.1f}s')

    if verbose:
        print(f'HMC done: accept_rate={accept_count/num_iterations:.3f}  t={time()-t0:.1f}s')
    return jnp.array(thetas)


# --- Convergence diagnostics ---

def _gelman_rubin_variance(x: np.ndarray) -> float:
    """Marginal posterior variance estimate (Gelman-Rubin). x shape: (num_chains, num_iters)."""
    m, n = x.shape
    B_over_n = ((jnp.mean(x, axis=1) - jnp.mean(x)) ** 2).sum() / (m - 1)
    W = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (m * (n - 1))
    return W * (n - 1) / n + B_over_n


def compute_Rhat(chains: jax.Array | np.ndarray) -> np.ndarray:
    """Gelman-Rubin R-hat convergence diagnostic. Values near 1.0 indicate convergence.

    arguments:
        chains:  array of shape (num_chains, num_samples, num_params)

    returns:
        Rhat:    array of shape (num_params,)
    """
    chains = np.array(chains)
    num_chains, num_samples, num_params = chains.shape

    # split each chain in half to double the number of sub-chains
    half = num_samples // 2
    sub = np.concatenate([chains[:, :half, :], chains[:, half:2*half, :]], axis=0)  # (2*num_chains, half, num_params)
    m, n = sub.shape[0], sub.shape[1]

    chain_means = sub.mean(axis=1)                                              # (m, num_params)
    chain_vars  = ((sub - chain_means[:, None, :]) ** 2).sum(axis=1) / (n - 1) # (m, num_params)

    global_mean = chain_means.mean(axis=0)
    B = n / (m - 1) * ((chain_means - global_mean) ** 2).sum(axis=0)
    W = chain_vars.mean(axis=0)

    var_est = (n - 1) / n * W + B / n
    return np.sqrt(var_est / W)


def compute_effective_sample_size(chains: jax.Array | np.ndarray) -> np.ndarray:
    """Effective sample size (ESS) for each parameter.

    arguments:
        chains:  array of shape (num_chains, num_samples, num_params)

    returns:
        ESS:     array of shape (num_params,)
    """
    chains = np.array(chains)
    num_chains, num_samples, num_params = chains.shape

    def _ess_single(x: np.ndarray) -> int:
        m, n = x.shape
        post_var = _gelman_rubin_variance(x)
        variogram = lambda t: ((x[:, t:] - x[:, :(n - t)]) ** 2).sum() / (m * (n - t))
        rho = np.ones(n)
        t, negative = 1, False
        while not negative and t < n:
            rho[t] = 1 - variogram(t) / (2 * post_var)
            if t % 2 == 0:
                negative = rho[t - 1] + rho[t] < 0
            t += 1
        return int(m * n / (1 + 2 * rho[1:t].sum()))

    return np.array([_ess_single(chains[:, :, p]) for p in range(num_params)])


# ==================================== General Minimal Plotting Functions ======================================================

def _plot_data(ax: Axes, Xtrain: jax.Array, ytrain: jax.Array) -> None:
    ax.plot(Xtrain, ytrain, 'k.', markersize=12, label='Data')
    ax.grid(True)
    ax.set_xlabel('Input $x$')
    ax.set_ylabel('Response $y$')
    ax.legend()

def plot_data(Xtrain: jax.Array, ytrain: jax.Array) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    _plot_data(ax, Xtrain, ytrain)

# Plot histogram

def plot_contour(ax: Axes,
                 f: Callable[[jax.Array, jax.Array], jax.Array],
                 x1s: jax.Array,
                 x2s: jax.Array,
                 num_contours: int = 10,
                 transform: Callable[[jax.Array], jax.Array] = lambda x: x,
                 color: str = 'b',
                 alpha: float = 1.0,
                 xlabel: str = '$x_1$',
                 ylabel: str = '$x_2$',
                 title: str = '') -> None:
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

def plot_heatmap(fig: Figure,
                 ax: Axes,
                 f: Callable[[jax.Array, jax.Array], jax.Array],
                 x1s: jax.Array,
                 x2s: jax.Array,
                 transform: Callable[[jax.Array], jax.Array] = lambda x: x,
                 xlabel: str = '$x_1$',
                 ylabel: str = '$x_2$',
                 title: str = '',
                 cmap: str = 'viridis') -> None:
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

# ==================================== MCMC Plotting Functions =========================================

def plot_trace(axes: Union[Axes, Sequence[Axes]],
               samples: jax.Array,
               param_names: Optional[list[str]] = None) -> None:
    """Trace plots for each parameter.

    arguments:
        axes:         single axes (1 parameter) or array/list of axes (one per parameter)
        samples:      array of shape (num_samples, num_params) or (num_samples,) for 1D
        param_names:  list of parameter name strings; defaults to θ_0, θ_1, …
    """
    samples = jnp.atleast_2d(samples) if jnp.ndim(samples) == 1 else samples
    num_params = samples.shape[1] if samples.ndim == 2 else 1
    axes_list: list[Axes] = [axes] if isinstance(axes, Axes) else list(axes)  # type: ignore[arg-type]
    names = param_names or [f'$\\theta_{i}$' for i in range(num_params)]
    for i, ax in enumerate(axes_list):
        ax.plot(samples[:, i] if samples.ndim == 2 else samples, lw=0.7)
        ax.set(xlabel='Iteration', ylabel=names[i], title=f'Trace of {names[i]}')


def plot_mcmc_diagnostics(chains: jax.Array | np.ndarray,
                          warm_up: int = 0,
                          param_names: Optional[list[str]] = None,
                          figsize: Optional[tuple[float, float]] = None,
                          ) -> tuple[Figure, np.ndarray]:
    """Full diagnostics panel: full trace | post-warmup trace | histogram, one row per parameter.
    Titles include R-hat and ESS where applicable (requires ≥2 chains).

    arguments:
        chains:       array of shape (num_chains, num_samples, num_params)
                      or (num_samples, num_params) for a single chain
        warm_up:      number of warm-up samples to discard for diagnostics and histograms
        param_names:  list of parameter name strings
        figsize:      figure size tuple; auto-scaled by default

    returns:
        fig, axes
    """
    chains = np.array(chains)
    if chains.ndim == 2:
        chains = chains[None, :, :]              # add chain dimension
    num_chains, num_samples, num_params = chains.shape
    names = param_names or [f'$\\theta_{i}$' for i in range(num_params)]

    post_chains = chains[:, warm_up:, :]
    multi = num_chains >= 2
    Rhat = compute_Rhat(post_chains) if multi else np.full(num_params, np.nan)
    ESS  = compute_effective_sample_size(post_chains) if multi else np.full(num_params, np.nan)

    fig, axes = plt.subplots(num_params, 3, figsize=figsize or (18, 3 * num_params), squeeze=False)

    for p in range(num_params):
        n = names[p]

        # full trace
        axes[p, 0].plot(chains[:, :, p].T, lw=0.6, alpha=0.8)
        if warm_up > 0:
            axes[p, 0].axvline(warm_up, color='r', linestyle='--', lw=1, label='warm-up end')
        axes[p, 0].set(xlabel='Iteration', ylabel=n, title=f'Full trace  {n}')

        # post-warmup trace
        rhat_str = f'  $\\hat{{R}}$={Rhat[p]:.3f}' if multi else ''
        axes[p, 1].plot(post_chains[:, :, p].T, lw=0.6, alpha=0.8)
        axes[p, 1].set(xlabel='Iteration', ylabel=n, title=f'Post-warmup trace  {n}{rhat_str}')

        # histogram
        ess_str = f'  ESS={ESS[p]:.0f}' if multi else ''
        flat = post_chains[:, :, p].ravel()
        axes[p, 2].hist(flat, bins=40, density=True, color=colors[0], alpha=0.7)
        lo, hi = np.percentile(flat, [2.5, 97.5])
        axes[p, 2].axvline(lo, color='r', linestyle='--', lw=1, label='95% interval')
        axes[p, 2].axvline(hi, color='r', linestyle='--', lw=1)
        axes[p, 2].set(xlabel=n, title=f'Histogram  {n}{ess_str}')
        axes[p, 2].legend(fontsize=9)

    fig.tight_layout()
    return fig, axes


def plot_posterior_1d(ax: Axes,
                      samples: jax.Array,
                      log_target: Optional[Callable[[jax.Array], jax.Array]] = None,
                      x_range: Optional[tuple[float, float]] = None,
                      num_bins: int = 40,
                      color: str = 'b',
                      label: str = 'Posterior') -> None:
    """Histogram of 1D samples with optional rescaled log_target overlay.

    arguments:
        ax:          matplotlib axes
        samples:     1D array of posterior samples
        log_target:  optional callable for the log unnormalized density; plotted as a rescaled curve
        x_range:     (lo, hi) tuple; defaults to sample range with 10% padding
        num_bins:    number of histogram bins
        color:       histogram color
        label:       legend label for the histogram
    """
    samples = np.array(samples).ravel()
    if x_range is None:
        margin = 0.1 * (samples.max() - samples.min())
        x_range = (float(samples.min() - margin), float(samples.max() + margin))
    lo, hi = x_range
    ax.hist(samples, bins=num_bins, density=True, color=color, alpha=0.5, label=label)
    if log_target is not None:
        xs = jnp.linspace(lo, hi, 500)
        log_p = np.array([log_target(jnp.array([x])) for x in xs])
        p = np.exp(log_p - log_p.max())
        ax.plot(xs, p / np.trapezoid(p, xs), color=color, lw=2, label='Target (rescaled)')
    ax.set(xlabel='$\\theta$', ylabel='Density')
    ax.legend()


def _plot_interval(ax: Axes, x: jax.Array, samples: Union[jax.Array, np.ndarray],
                   interval: float, color: str, alpha: float, **kwargs) -> None:
    lo = np.percentile(samples, 0.5 * (100 - interval), axis=0)
    hi = np.percentile(samples, 100 - 0.5 * (100 - interval), axis=0)
    ax.fill_between(np.array(x).ravel(), lo, hi, color=color, alpha=alpha, **kwargs)


def plot_predictions(ax: Axes,
                     x: jax.Array,
                     samples: jax.Array,
                     num_samples: int = 100,
                     sample_color: str = 'k',
                     sample_alpha: float = 0.3,
                     color: str = 'r',
                     legend: bool = False,
                     plot_mean: bool = True,
                     seed: int = 123,
                     title: str = '') -> None:
    """Layered credibility-interval plot for posterior predictive samples.
    Plots 99%, 95%, and 75% intervals as shaded bands plus individual sample curves.

    arguments:
        ax:           matplotlib axes
        x:            1D array of input values for the x-axis
        samples:      2D array of shape (num_samples, num_x_points)
        num_samples:  number of individual sample curves to overlay
        sample_color: color for individual sample curves
        sample_alpha: opacity for individual sample curves
        color:        color for interval bands and mean
        legend:       whether to show legend
        plot_mean:    whether to overlay the posterior mean
        seed:         random seed for selecting sample curves
        title:        axes title
    """
    samples = np.array(samples)
    x = np.array(x).ravel()

    for interval, alpha in [(99, 0.20), (95, 0.30), (75, 0.50)]:
        _plot_interval(ax, x, samples, interval, color, alpha,
                       label=f'{interval}% interval' if interval == 95 else None)

    if num_samples > 0:
        np.random.seed(seed)
        idx = np.random.choice(len(samples), size=min(num_samples, len(samples)), replace=False)
        ax.plot(x, samples[idx].T, color=sample_color, alpha=sample_alpha, lw=0.5)

    if plot_mean:
        ax.plot(x, samples.mean(axis=0), '-', color='k', lw=2.5, label='Mean')

    if title:
        ax.set_title(title, fontweight='bold')
    if legend:
        ax.legend(loc='best')


# ====================================== Gaussian Processes ============================================

def generate_samples(key: jax.Array, m: jax.Array, K: jax.Array,
                     num_samples: int, jitter: float = 0) -> jax.Array:
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

    def to_array(self) -> jax.Array:
        """ return hyperparameters as flat JaX-array (to be used later) """
        return jnp.array([self.kappa, self.lengthscale, self.sigma])

    @staticmethod
    def from_array(hyper_array: jax.Array) -> Hyperparameters:
        """ instantiates Hyperparameter object from flat JaX-array (or list) of hyperparameters (to be used later) """
        kappa, lengthscale, sigma = hyper_array
        return Hyperparameters(kappa, lengthscale, sigma)

    def __repr__(self) -> str:
        """ for reporting hyperparameter values """
        return f'Hyperparameters(kappa={self.kappa:3.2f}, lengthscale={self.lengthscale:3.2f}, sigma={self.sigma:3.2f})'

# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(tau: jax.Array, hyperparameters: Hyperparameters) -> jax.Array:
    return hyperparameters.kappa**2*jnp.exp(-0.5*tau**2/hyperparameters.lengthscale**2)

def matern12(tau: jax.Array, hyperparameters: Hyperparameters) -> jax.Array:
    return hyperparameters.kappa**2*jnp.exp(-tau/hyperparameters.lengthscale)

def matern32(tau: jax.Array, hyperparameters: Hyperparameters) -> jax.Array:
    return hyperparameters.kappa**2*(1 + jnp.sqrt(3)*tau/hyperparameters.lengthscale)*jnp.exp(-jnp.sqrt(3)*tau/hyperparameters.lengthscale)

class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun: Callable[[jax.Array, Hyperparameters], jax.Array]) -> None:
        """
            the argument kernel_fun must be a function of two arguments kernel_fun(||tau||, hyperparameters), e.g.
            squared_exponential = lambda tau, hyper: hyper.kappa**2*np.exp(-0.5*tau**2/hyper.lengthscale**2).
        """
        self.kernel_fun = kernel_fun

    def construct_kernel(self, X1: jax.Array, X2: jax.Array,
                         hyperparameters: Hyperparameters, jitter: float = 1e-8) -> jax.Array:
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

    def __init__(self, kernel_fun: Callable[[jax.Array, jax.Array, Hyperparameters], jax.Array]) -> None:
        """
        arguments:
            kernel_fun  -- function kernel_fun(x1, x2, hyperparameters) where
                           x1 (shape (D,)) and x2 (shape (D,)) are input vectors.
                           Must return a scalar kernel value k(x1, x2).
        """
        self.kernel_fun = kernel_fun

    def construct_kernel(self, X1: jax.Array, X2: jax.Array,
                         hyperparameters: Hyperparameters, jitter: float = 1e-8) -> jax.Array:
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

    def __init__(self, X: jax.Array, y: jax.Array,
                 kernel: StationaryIsotropicKernel | Kernel,
                 hyperparameters: Hyperparameters,
                 jitter: float = 1e-8) -> None:
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

    def check_dimensions(self) -> None:
        N, D = self.X.shape
        assert self.X.ndim == 2, f"The variable X must be of shape (N, D), however, the current shape is: {self.X.shape}"
        assert self.y.ndim == 2, f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        assert self.y.shape == (N, 1), f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"

    def set_hyperparameters(self, hyper: Hyperparameters) -> None:
        self.hyperparameters = hyper

    def posterior_samples(self, key: jax.Array, Xstar: jax.Array, num_samples: int) -> jax.Array:
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

    def predict_y(self, Xstar: jax.Array) -> tuple[jax.Array, jax.Array]:
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

    def predict_f(self, Xstar: jax.Array) -> tuple[jax.Array, jax.Array]:
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

    def log_marginal_likelihood(self, hyperparameters: Hyperparameters) -> jax.Array:
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

def optimize_marginal_likelihood(gp: GaussianProcessRegression,
                                 hyperparameters_init: Hyperparameters,
                                 verbose: bool = True) -> Hyperparameters:
    """ Optimize log marginal likelihood with gradient-based methods """

    def objective(log_hyperparam_array: jax.Array) -> jax.Array:
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

def add_colorbar(im, fig: Figure, ax: Axes) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

def plot_kernel(X: jax.Array, K: jax.Array, hyper: Hyperparameters,
                key: jax.Array, num_samples: int) -> None:
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


def _plot_with_uncertainty(ax: Axes, Xp: jax.Array, gp: GaussianProcessRegression,
                           color: str = 'r', color_samples: str = 'b',
                           title: str = "", num_samples: int = 0, seed: int = 0) -> None:

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

def plot_with_uncertainty(kernel: StationaryIsotropicKernel | Kernel,
                          hyper: Hyperparameters,
                          Xtrain: jax.Array,
                          ytrain: jax.Array,
                          Xstar: jax.Array) -> None:
    gp_prior = GaussianProcessRegression(jnp.zeros((0, 1)), jnp.zeros((0, 1)), kernel, hyper)
    gp_post = GaussianProcessRegression(Xtrain, ytrain, kernel, hyper)
    fig, ax = plt.subplots(1, 2, figsize=(25, 6))
    _plot_with_uncertainty(ax[0], Xstar, gp_prior, title='Prior predictive distribution', num_samples=30)
    _plot_with_uncertainty(ax[1], Xstar, gp_post, title='Posterior predictive distribution', num_samples=30)
    for i in range(2):
        _plot_data(ax[i], Xtrain, ytrain)
        ax[i].legend(loc='lower center', ncol=4)
