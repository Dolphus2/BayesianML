"""
exam_utils_example.py — Worked examples for every section of exam_utils.py.

Run the whole file or copy individual sections into a notebook cell.
Each section is self-contained: it generates its own data and shows a complete
workflow from data → model → inference → plot.

Sections
--------
1.  Activation functions
2.  Probability distributions (univariate & multivariate)
3.  Bayesian linear regression
4.  Laplace approximation
5.  MCMC — Metropolis-Hastings
6.  MCMC — Hamiltonian Monte Carlo + convergence diagnostics
7.  Gaussian Processes
8.  Variational inference — Black-Box VI
9.  Variational GMM (from exercise10)
"""

import matplotlib
matplotlib.use("Agg")          # headless — swap to "TkAgg" / remove for notebooks
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax import random, value_and_grad

import bayesian
from bayesian import (
    # activations
    sigmoid, softplus, relu, softmax,
    # distributions
    probit,
    gaussian_logpdf, gaussian_pdf,
    bernoulli_logpmf,
    mvn_logpdf, mvn_pdf, mvn_sample,
    gamma_logpdf, gamma_pdf, gamma_sample,
    beta_logpdf, beta_pdf, beta_sample,
    dirichlet_logpdf, dirichlet_pdf, dirichlet_sample,
    poisson_logpmf, poisson_pmf, poisson_sample,
    binomial_logpmf, binomial_pmf, binomial_sample,
    # regression
    compute_posterior_w, marginal_likelihood, laplace_approximation,
    # MCMC
    metropolis, HMC, leapfrog,
    compute_Rhat, compute_effective_sample_size,
    # GP
    Hyperparameters, StationaryIsotropicKernel, Kernel,
    GaussianProcessRegression, optimize_marginal_likelihood,
    squared_exponential, matern12, matern32, generate_samples,
    # VI
    kl_gaussian, BlackBoxVI, VariationalGMM,
    # plotting
    plot_data, plot_contour, plot_heatmap,
    plot_trace, plot_mcmc_diagnostics, plot_posterior_1d, plot_predictions,
    plot_elbo, plot_vi_diagnostics, plot_with_uncertainty,
)

from jax import config
config.update("jax_enable_x64", True)

key = random.PRNGKey(0)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Activation functions
# ──────────────────────────────────────────────────────────────────────────────

x = jnp.linspace(-4, 4, 200)

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
axes[0].plot(x, sigmoid(x));    axes[0].set_title("sigmoid")
axes[1].plot(x, softplus(x));   axes[1].set_title("softplus")
axes[2].plot(x, relu(x));       axes[2].set_title("relu")
logits = jnp.array([1.0, 2.0, 0.5, -1.0])
axes[3].bar(range(4), softmax(logits)); axes[3].set_title("softmax")
fig.suptitle("Activation functions", fontweight="bold")
plt.tight_layout(); plt.savefig("out_activations.png", dpi=80); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Probability distributions
# ──────────────────────────────────────────────────────────────────────────────

# --- Univariate distributions ------------------------------------------------

t = jnp.linspace(-4, 4, 300)
fig, ax = plt.subplots(figsize=(8, 3))
for mu, sigma in [(0.0, 1.0), (1.0, 0.5), (-1.0, 2.0)]:
    ax.plot(t, gaussian_pdf(t, mu, sigma), label=f"N({mu}, {sigma}²)")
ax.set(xlabel="x", ylabel="p(x)", title="Gaussian PDF"); ax.legend()
plt.tight_layout(); plt.savefig("out_gaussian.png", dpi=80); plt.close()

# Beta
p = jnp.linspace(0.01, 0.99, 300)
fig, ax = plt.subplots(figsize=(8, 3))
for a, b in [(0.5, 0.5), (1, 1), (2, 5), (5, 2)]:
    ax.plot(p, beta_pdf(p, a, b), label=f"Beta({a},{b})")
ax.set(xlabel="θ", ylabel="p(θ)", title="Beta PDF"); ax.legend()
plt.tight_layout(); plt.savefig("out_beta.png", dpi=80); plt.close()

# Gamma
x_g = jnp.linspace(0.01, 10, 300)
fig, ax = plt.subplots(figsize=(8, 3))
for a, scale in [(1, 1), (2, 1), (3, 2)]:
    ax.plot(x_g, gamma_pdf(x_g, a, scale), label=f"Gamma(a={a}, scale={scale})")
ax.set(xlabel="x", ylabel="p(x)", title="Gamma PDF"); ax.legend()
plt.tight_layout(); plt.savefig("out_gamma.png", dpi=80); plt.close()

# Poisson PMF
k_vals = jnp.arange(0, 20)
fig, ax = plt.subplots(figsize=(8, 3))
for lam in [1.0, 4.0, 8.0]:
    ax.plot(k_vals, poisson_pmf(k_vals, lam), "o-", label=f"Poisson(λ={lam})")
ax.set(xlabel="k", ylabel="P(K=k)", title="Poisson PMF"); ax.legend()
plt.tight_layout(); plt.savefig("out_poisson.png", dpi=80); plt.close()

# --- Multivariate normal -----------------------------------------------------

key, subkey = random.split(key)
mu_2d  = jnp.array([1.0, -0.5])
Sig_2d = jnp.array([[1.0, 0.7], [0.7, 0.5]])
samples_mvn = mvn_sample(subkey, mu_2d, Sig_2d, shape=(500,))

x1s = jnp.linspace(-3, 5, 100)
x2s = jnp.linspace(-3, 2, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(*np.array(samples_mvn).T, s=5, alpha=0.5)
axes[0].set(title="MVN samples", xlabel="x₁", ylabel="x₂")

plot_contour(axes[1],
             f=lambda X1, X2: mvn_logpdf(
                 jnp.stack([X1, X2], axis=-1), mu_2d, Sig_2d),
             x1s=x1s, x2s=x2s,
             transform=jnp.exp, num_contours=8, title="MVN density")
plt.tight_layout(); plt.savefig("out_mvn.png", dpi=80); plt.close()

# --- KL divergence between two diagonal Gaussians ----------------------------

m_q, v_q = jnp.array([1.0, 0.0]), jnp.array([0.5, 1.0])
m_p, v_p = jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])
kl = kl_gaussian(m_q, v_q, m_p, v_p)
print(f"KL(q‖p) = {float(kl):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Bayesian linear regression
# ──────────────────────────────────────────────────────────────────────────────

# Generate synthetic data:  y = w0 + w1*x + noise
np.random.seed(0)
N = 40
X_raw = np.sort(np.random.uniform(-3, 3, N))
w_true = np.array([0.5, 1.5])
y_raw  = w_true[0] + w_true[1] * X_raw + np.random.normal(0, 0.5, N)

# Design matrix (affine): Phi = [1, x]
Phi = jnp.column_stack([jnp.ones(N), jnp.array(X_raw)])
y   = jnp.array(y_raw)

# --- Posterior ---------------------------------------------------------------
alpha, beta_prec = 1.0, 4.0          # prior precision, noise precision

m, S = compute_posterior_w(Phi, y, alpha, beta_prec)
print(f"BLR posterior mean: {np.array(m)}")
print(f"BLR posterior std:  {np.sqrt(np.diag(np.array(S)))}")

# --- Predictions -------------------------------------------------------------
x_star = jnp.linspace(-4, 4, 300)
Phi_star = jnp.column_stack([jnp.ones(300), x_star])
mu_f  = Phi_star @ m
var_f = jnp.diag(Phi_star @ S @ Phi_star.T)
var_y = var_f + 1 / beta_prec

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(X_raw, y_raw, c="k", s=20, zorder=5, label="Data")
ax.plot(x_star, mu_f, "b-", label="Posterior mean")
ax.fill_between(x_star, mu_f - 2*jnp.sqrt(var_y), mu_f + 2*jnp.sqrt(var_y),
                alpha=0.2, color="b", label="95% predictive")
ax.set(xlabel="x", ylabel="y", title="Bayesian linear regression")
ax.legend(); plt.tight_layout(); plt.savefig("out_blr.png", dpi=80); plt.close()

# --- Marginal likelihood & hyperparameter optimisation -----------------------
log_ml = marginal_likelihood(Phi, y, alpha, beta_prec)
print(f"Log marginal likelihood: {float(log_ml):.4f}")

def neg_log_ml(log_theta):
    a, b = jnp.exp(log_theta[0]), jnp.exp(log_theta[1])
    return -marginal_likelihood(Phi, y, a, b)

from scipy.optimize import minimize
res = minimize(value_and_grad(neg_log_ml), jnp.zeros(2), jac=True)
alpha_opt, beta_opt = float(jnp.exp(res.x[0])), float(jnp.exp(res.x[1]))
print(f"Optimal  alpha={alpha_opt:.4f},  beta={beta_opt:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Laplace approximation
# ──────────────────────────────────────────────────────────────────────────────

# Target: skewed 1D distribution  p(w) ∝ exp(-w²/2) * sigmoid(3w)
def log_target_1d(w):
    return jnp.sum(gaussian_logpdf(w, 0.0, 1.0)) + jnp.sum(jnp.log(sigmoid(3.0 * w)))

w0 = jnp.zeros(1)
m_lap, S_lap = laplace_approximation(log_target_1d, w0)
print(f"Laplace MAP: {float(m_lap[0]):.4f},  std: {float(jnp.sqrt(S_lap[0,0])):.4f}")

# Compare to MCMC ground truth
samples_lap = metropolis(log_target_1d, num_params=1, tau=0.5,
                         num_iter=20_000, seed=42, verbose=False)
xs = jnp.linspace(-3, 3, 300)
log_p = np.array([float(log_target_1d(jnp.array([x]))) for x in xs])
p = np.exp(log_p - log_p.max())

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(np.array(samples_lap[1000:, 0]), bins=60, density=True,
        alpha=0.5, color="b", label="MCMC")
ax.plot(xs, p / np.trapezoid(p, xs), "b-", lw=1.5, label="True (rescaled)")
ax.plot(xs, gaussian_pdf(xs, float(m_lap[0]), float(jnp.sqrt(S_lap[0,0]))),
        "r--", lw=2, label="Laplace")
ax.set(xlabel="w", ylabel="density", title="Laplace approximation")
ax.legend(); plt.tight_layout(); plt.savefig("out_laplace.png", dpi=80); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 5. MCMC — Metropolis-Hastings
# ──────────────────────────────────────────────────────────────────────────────

# Target: correlated 2D Gaussian
mu_tgt  = jnp.array([1.0, -0.5])
Sig_tgt = jnp.array([[1.0, 0.8], [0.8, 1.0]])

def log_target_2d(theta):
    return mvn_logpdf(theta, mu_tgt, Sig_tgt)

# Run sampler
samples_mh = metropolis(
    log_target_2d,
    num_params=2,
    tau=1.0,           # proposal std. dev. — tune for ~20-40% acceptance
    num_iter=10_000,
    theta_init=jnp.zeros(2),
    seed=0,
)
# samples_mh shape: (10_001, 2)

warm_up = 1000
post_mh = samples_mh[warm_up:]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(*np.array(post_mh).T, s=2, alpha=0.3)
axes[0].set(xlabel="θ₀", ylabel="θ₁", title="MH posterior samples")
axes[1].hist(np.array(post_mh[:, 0]), bins=40, density=True, alpha=0.6)
axes[1].set(xlabel="θ₀", ylabel="density", title="MH marginal θ₀")
plt.tight_layout(); plt.savefig("out_mh.png", dpi=80); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 6. MCMC — HMC + convergence diagnostics
# ──────────────────────────────────────────────────────────────────────────────

# Same 2D Gaussian target as above
samples_hmc1 = HMC(log_target_2d, num_iterations=3000, theta0=jnp.zeros(2),
                   num_leapfrog_steps=10, step_size=0.3, seed=0)
samples_hmc2 = HMC(log_target_2d, num_iterations=3000, theta0=jnp.ones(2),
                   num_leapfrog_steps=10, step_size=0.3, seed=1, verbose=False)

# Stack chains: shape (num_chains, num_samples, num_params)
chains = jnp.stack([samples_hmc1[None], samples_hmc2[None]], axis=0)
# chains.shape == (2, 3001, 2)  →  squeeze the extra dim first
chains = jnp.concatenate([samples_hmc1[None], samples_hmc2[None]], axis=0)
chains = chains[:, :, :]   # (2, 3001, 2)

fig, axes = plot_mcmc_diagnostics(chains, warm_up=500,
                                  param_names=["θ₀", "θ₁"])
plt.tight_layout(); plt.savefig("out_hmc_diag.png", dpi=80); plt.close()

Rhat = compute_Rhat(chains[:, 500:, :])
ESS  = compute_effective_sample_size(chains[:, 500:, :])
print(f"R-hat:  {np.array(Rhat)}  (< 1.01 → converged)")
print(f"ESS:    {np.array(ESS)}")

# Leapfrog trajectory visualisation
theta0 = jnp.zeros(2)
nu0    = random.normal(key, shape=(2,))
from jax import grad, jit
potential    = jit(lambda t: -log_target_2d(t))
grad_pot     = jit(grad(potential))

thetas, nus = [theta0], [nu0]
theta, nu = theta0, nu0
for _ in range(20):
    nu    = nu    - 0.5 * 0.3 * grad_pot(theta)
    theta = theta + 0.3 * nu
    nu    = nu    - 0.5 * 0.3 * grad_pot(theta)
    thetas.append(theta); nus.append(nu)

traj = np.array(thetas)
fig, ax = plt.subplots(figsize=(6, 5))
x1s = jnp.linspace(-2, 4, 80)
x2s = jnp.linspace(-3, 2, 80)
plot_contour(ax, lambda X1, X2: mvn_logpdf(
    jnp.stack([X1, X2], axis=-1), mu_tgt, Sig_tgt),
    x1s, x2s, transform=jnp.exp, num_contours=6, title="Leapfrog trajectory")
ax.plot(traj[:, 0], traj[:, 1], "r.-", lw=1.5, ms=6)
ax.plot(*traj[0], "go", ms=10, label="start")
ax.legend(); plt.tight_layout(); plt.savefig("out_leapfrog.png", dpi=80); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Gaussian Processes
# ──────────────────────────────────────────────────────────────────────────────

# Synthetic 1D regression data
key, subkey = random.split(key)
f_true  = lambda x: jnp.sin(2 * x) * jnp.exp(-0.3 * x**2)
X_train = jnp.linspace(-3, 3, 12)[:, None]
y_train = f_true(X_train) + 0.15 * random.normal(subkey, X_train.shape)
X_star  = jnp.linspace(-4, 4, 300)[:, None]

# --- Squared-exponential kernel ----------------------------------------------
kernel = StationaryIsotropicKernel(squared_exponential)
hyper  = Hyperparameters(kappa=1.0, lengthscale=1.0, sigma=0.2)

gp = GaussianProcessRegression(X_train, y_train, kernel, hyper)
mu, Sigma = gp.predict_y(X_star)
std = jnp.sqrt(jnp.diag(Sigma))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(X_star, f_true(X_star), "k--", lw=1.5, label="True f")
ax.plot(X_star, mu, "b-", label="Posterior mean")
ax.fill_between(X_star.ravel(), mu.ravel()-2*std, mu.ravel()+2*std,
                alpha=0.2, color="b", label="95% predictive")
ax.scatter(X_train.ravel(), y_train.ravel(), c="k", s=40, zorder=5, label="Data")
ax.set(xlabel="x", ylabel="y", title=f"GP regression  ({hyper})")
ax.legend(); plt.tight_layout(); plt.savefig("out_gp.png", dpi=80); plt.close()

# --- Hyperparameter optimisation via marginal likelihood ---------------------
hyper_opt = optimize_marginal_likelihood(gp, hyper, verbose=True)
gp_opt    = GaussianProcessRegression(X_train, y_train, kernel, hyper_opt)
print(f"Log ML (init): {float(gp.log_marginal_likelihood(hyper)):.4f}")
print(f"Log ML (opt):  {float(gp_opt.log_marginal_likelihood(hyper_opt)):.4f}")

# --- Custom (non-stationary) kernel via Kernel class -------------------------
linear_kernel = Kernel(lambda x1, x2, h: h.kappa**2 * jnp.dot(x1, x2))
hyper_lin     = Hyperparameters(kappa=1.0)
gp_lin        = GaussianProcessRegression(X_train, y_train, linear_kernel, hyper_lin)
mu_lin, Sigma_lin = gp_lin.predict_y(X_star)
std_lin = jnp.sqrt(jnp.diag(Sigma_lin))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(X_star, mu_lin, "r-", label="Linear kernel mean")
ax.fill_between(X_star.ravel(), mu_lin.ravel()-2*std_lin, mu_lin.ravel()+2*std_lin,
                alpha=0.2, color="r")
ax.scatter(X_train.ravel(), y_train.ravel(), c="k", s=40, zorder=5)
ax.set(xlabel="x", ylabel="y", title="GP with linear kernel")
ax.legend(); plt.tight_layout(); plt.savefig("out_gp_linear.png", dpi=80); plt.close()

# --- Compare three kernel families -------------------------------------------
kernels = {
    "SE":       StationaryIsotropicKernel(squared_exponential),
    "Matérn½":  StationaryIsotropicKernel(matern12),
    "Matérn³⁄₂": StationaryIsotropicKernel(matern32),
}
hyper_init = Hyperparameters(kappa=1.0, lengthscale=1.0, sigma=0.2)

fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
for ax, (name, k) in zip(axes, kernels.items()):
    h_opt = optimize_marginal_likelihood(
        GaussianProcessRegression(X_train, y_train, k, hyper_init),
        hyper_init, verbose=False)
    gp_k = GaussianProcessRegression(X_train, y_train, k, h_opt)
    mu_k, Sig_k = gp_k.predict_y(X_star)
    std_k = jnp.sqrt(jnp.diag(Sig_k))
    ax.plot(X_star, mu_k, "b-"); ax.fill_between(
        X_star.ravel(), mu_k.ravel()-2*std_k, mu_k.ravel()+2*std_k,
        alpha=0.2, color="b")
    ax.scatter(X_train.ravel(), y_train.ravel(), c="k", s=20, zorder=5)
    ax.set(title=f"{name}\n{h_opt}", xlabel="x")
plt.suptitle("Kernel comparison", fontweight="bold")
plt.tight_layout(); plt.savefig("out_gp_kernels.png", dpi=80); plt.close()

# --- GP prior samples --------------------------------------------------------
K_prior = gp.kernel.construct_kernel(X_star, X_star, hyper_opt)
key, subkey = random.split(key)
f_prior = generate_samples(subkey, jnp.zeros(len(X_star)), K_prior,
                            num_samples=5, jitter=1e-6)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(X_star, f_prior, alpha=0.7, lw=1.5)
ax.set(xlabel="x", ylabel="f(x)", title="GP prior samples")
plt.tight_layout(); plt.savefig("out_gp_prior.png", dpi=80); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 8. Variational Inference — Black-Box VI
# ──────────────────────────────────────────────────────────────────────────────

# Task: Bayesian logistic regression on 2-class synthetic data

np.random.seed(1)
N_vi = 80
X_vi = jnp.array(np.random.randn(N_vi, 2))
w_vi_true = jnp.array([1.5, -1.0])
y_vi = (random.bernoulli(
    random.PRNGKey(9),
    sigmoid(X_vi @ w_vi_true),
    shape=(N_vi,)
)).astype(float)

# log-prior:  p(w) = N(w | 0, I),  w shape (S, D) → (S,)
def log_prior_vi(w):
    return jnp.sum(gaussian_logpdf(w, 0.0, 1.0), axis=-1)

# log-likelihood: Bernoulli,  w shape (S, D) → (S,)
def log_lik_vi(X, y, w):
    logits = w @ X.T                            # (S, N)
    return jnp.sum(bernoulli_logpmf(y, sigmoid(logits)), axis=-1)

D_vi = 2
vi = BlackBoxVI(
    log_prior_vi, log_lik_vi,
    num_params=D_vi,
    step_size=0.02,
    max_itt=1500,
    num_samples=20,
    seed=0,
    verbose=300,
)
vi.fit(X_vi, y_vi)

m_opt, v_opt = vi.unpack(vi.lam)
print(f"VI mean:  {np.array(m_opt)}")
print(f"VI std:   {np.sqrt(np.array(v_opt))}")
print(f"True w:   {np.array(w_vi_true)}")

# ELBO convergence
fig, ax = plt.subplots(figsize=(8, 3))
plot_elbo(ax, vi)
plt.tight_layout(); plt.savefig("out_vi_elbo.png", dpi=80); plt.close()

# Full diagnostics panel (ELBO + mean/std per parameter)
fig, axes = plot_vi_diagnostics(vi, param_names=["w₀", "w₁"])
plt.tight_layout(); plt.savefig("out_vi_diag.png", dpi=80); plt.close()

# Posterior samples from q(w)
key, subkey = random.split(key)
vi_samples = vi.generate_posterior_samples(subkey, num_samples=2000)
# vi_samples shape: (2000, 2)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(*np.array(vi_samples).T, s=3, alpha=0.3, label="VI samples")
ax.scatter(*np.array(w_vi_true), c="r", s=80, zorder=5, label="True w")
ax.set(xlabel="w₀", ylabel="w₁", title="VI posterior q(w)")
ax.legend(); plt.tight_layout(); plt.savefig("out_vi_samples.png", dpi=80); plt.close()

# Compare VI posterior to HMC ground truth
def log_target_logreg(theta):
    return log_prior_vi(theta[None])[0] + log_lik_vi(X_vi, y_vi, theta[None])[0]

hmc_lr = HMC(log_target_logreg, num_iterations=2000, theta0=m_opt,
             num_leapfrog_steps=10, step_size=0.1, seed=2, verbose=False)
hmc_post_lr = hmc_lr[500:]

fig, axes = plt.subplots(1, D_vi, figsize=(5*D_vi, 4))
param_names = ["w₀", "w₁"]
for i, ax in enumerate(axes):
    xs = jnp.linspace(float(hmc_post_lr[:, i].min()) - 0.3,
                      float(hmc_post_lr[:, i].max()) + 0.3, 300)
    ax.hist(np.array(hmc_post_lr[:, i]), bins=40, density=True,
            alpha=0.5, color="b", label="HMC")
    ax.plot(xs, gaussian_pdf(xs, m_opt[i], jnp.sqrt(v_opt[i])),
            "r-", lw=2, label="VI")
    ax.set(xlabel=param_names[i], ylabel="Density"); ax.legend()
plt.suptitle("VI vs. HMC posterior comparison", fontweight="bold")
plt.tight_layout(); plt.savefig("out_vi_vs_hmc.png", dpi=80); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 9. Variational GMM (VariationalGMM from exercise10)
# ──────────────────────────────────────────────────────────────────────────────

# Generate 3-component mixture data
key, k1, k2, k3 = random.split(key, 4)
X_gmm = jnp.concatenate([
    random.multivariate_normal(k1, jnp.array([-3.0, 0.0]),
                               jnp.eye(2), shape=(60,)),
    random.multivariate_normal(k2, jnp.array([3.0, 0.0]),
                               jnp.eye(2), shape=(60,)),
    random.multivariate_normal(k3, jnp.array([0.0, 3.0]),
                               0.5 * jnp.eye(2), shape=(60,)),
])

# Fit Variational GMM (CAVI)
vgmm = VariationalGMM(D=2, K=3, alpha0=0.1, beta0=0.01)
vgmm.fit(X_gmm, max_itt=200)

# Component probabilities on a grid
x1s = jnp.linspace(-6, 6, 80)
x2s = jnp.linspace(-3, 6, 80)
X1, X2 = jnp.meshgrid(x1s, x2s)
X_grid  = jnp.column_stack([X1.ravel(), X2.ravel()])
log_pred = vgmm.evaulate_log_predictive(X_grid, pointwise=True)

fig, ax = plt.subplots(figsize=(7, 6))
ax.contourf(x1s, x2s, np.array(jnp.exp(log_pred).reshape(80, 80)),
            levels=12, cmap="Blues")
ax.scatter(*np.array(X_gmm).T, s=8, c="k", alpha=0.4)
ax.set(xlabel="x₁", ylabel="x₂", title="Variational GMM predictive density")
plt.tight_layout(); plt.savefig("out_vgmm.png", dpi=80); plt.close()

print("All examples complete — figures saved to out_*.png")
