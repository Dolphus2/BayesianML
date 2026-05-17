from .exam_utils import (
    # utility
    Grid2D,
    GridApproximation2D,
    plot_grid_marginals,

    # activation functions
    sigmoid,
    softplus,
    relu,
    softmax,

    # probability distributions
    probit,
    gaussian_logpdf,
    gaussian_pdf,
    bernoulli_logpmf,
    mvn_logpdf,
    mvn_pdf,
    mvn_sample,
    gamma_logpdf,
    gamma_pdf,
    gamma_sample,
    beta_logpdf,
    beta_pdf,
    beta_sample,
    poisson_logpmf,
    poisson_pmf,
    poisson_sample,
    binomial_logpmf,
    binomial_pmf,
    binomial_sample,
    dirichlet_logpdf,
    dirichlet_pdf,
    dirichlet_sample,

    # Bayesian linear regression
    compute_posterior_w,
    marginal_likelihood,

    # Laplace approximation
    laplace_approximation,

    # MCMC / HMC
    metropolis,
    leapfrog,
    HMC,
    compute_Rhat,
    compute_effective_sample_size,

    # general plotting
    plot_data,
    plot_contour,
    plot_heatmap,
    add_colorbar,

    # MCMC plotting
    plot_trace,
    plot_mcmc_diagnostics,
    plot_posterior_1d,
    plot_predictions,

    # Gaussian processes
    generate_samples,
    Hyperparameters,
    squared_exponential,
    matern12,
    matern32,
    StationaryIsotropicKernel,
    Kernel,
    GaussianProcessRegression,
    optimize_marginal_likelihood,

    # GP plotting
    plot_kernel,
    plot_with_uncertainty,
    plot_with_uncertainty_laplace,

    # variational inference
    kl_gaussian,
    BlackBoxVI,
    plot_elbo,
    plot_vi_diagnostics,

    # re-exported from exercise10 / exercise11
    VariationalGMM,
    plot_std_dev_contour,
    PCA_dim_reduction,
    AdamOptimizer,
    create_linear_regression_data,
)
