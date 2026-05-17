import jax.numpy as jnp
import jax
from jax import jit
from jax import random
from jax import grad
import optax

from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree


@jit
def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*jnp.log(2*jnp.pi*v)

def plot_predictions(ax, Xp, mean, variances, color='b', seed=0):
    """ Plot the predictive distribution (mean and 95% credibility interval) obtained.
        Below, S denotes the number of samples and P denoted the number of points. For each of the S samples and
        for each of the P points, the function generates a sample from corresponding Normal distribution and computes the mean and 95% interval and plots it on ax.
        
    inputs:
    ax          -- axis object for plotting
    mean        -- array of mean values (np.array: S x P)
    variances   -- array of variance values (np.array: S x P)

    """
    key = random.PRNGKey(seed)
    f = mean + jnp.sqrt(variances)*random.normal(key, shape=mean.shape)
    f_mean  = jnp.mean(f, axis=0)
    f_lower = jnp.percentile(f, 2.5, axis=0)
    f_upper = jnp.percentile(f, 97.5, axis=0)

    ax.plot(Xp, f_mean, color=color)
    ax.fill_between(Xp.ravel(), f_lower, f_upper, alpha=0.2, color=color)


#############################################################################
# Adapted from
# https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
#############################################################################

@jit
def nn_predict(params, inputs):
    """ Implements a deep neural network.
        params is a list of (weights, bias) tuples.
        inputs is an (N x D) matrix.
        returns logits. """
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
    return outputs


class NeuralNetwork(object):

    def __init__(self, X, y, layer_sizes, log_lik_fun, alpha=1., step_size=0.01, num_iters=1000, batch_size=None, seed=0, verbose=200):

        # data
        self.X = X
        self.y = y
        self.N = len(self.X)

        # model and optimization parameters
        self.log_lik_fun = log_lik_fun
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.num_iters = num_iters
        self.alpha = alpha
        self.batch_size = batch_size
        self.verbose = verbose

        # random number genration
        self.seed=seed
        self.key = random.PRNGKey(seed)
        
        # initialize parameters and optimize
        self.params = self.init_random_params()
        self.optimize()


    def init_random_params(self):
        """Build a list of (weights, biases) tuples, one for each layer in the net."""
        params = []
        for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.key, key_weight, key_bias = random.split(self.key, num=3)
            w = jnp.sqrt(2/n) * random.normal(key_weight, shape=(m, n)) # weight matrix
            b = jnp.sqrt(2/n) * random.normal(key_bias, shape=(n,))     # bias vector
            params.append((w,b))
        return params


    def first_layers(self, params, inputs):
        """ implements the map from the input features to the activation of layer L-1 """
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = jnp.tanh(outputs)
        return inputs

    def last_layer(self, last_params, first_layers):
        """ implements the map from the activation of layer L-1 to the output of the network"""
    
        W, b = last_params
        return jnp.dot(first_layers, W) +b 


    def predict(self, inputs):
        return nn_predict(self.params, inputs)

    def log_prior(self, params):
        # implement a Gaussian prior on the weights
        flattened_params, _ = ravel_pytree(params)
        return  jnp.sum(log_npdf(flattened_params, 0., 1/self.alpha))
    
    def log_likelihood(self, params, key=None, compute_full=False):  
        if self.batch_size is None or compute_full:
            f = nn_predict(params, self.X)
            log_lik = jnp.sum(self.log_lik_fun(f.reshape((self.N, -1)), self.y))
        else:
            batch_idx = random.choice(key, jnp.arange(self.N), shape=(self.batch_size,), replace=False)
            X_batch, y_batch = self.X[batch_idx, :], self.y[batch_idx]
            f = nn_predict(params, X_batch)
            log_lik = self.N/self.batch_size*jnp.sum(self.log_lik_fun(f.reshape((self.batch_size, -1)), y_batch))
        
        return log_lik

    def log_joint(self, params, key=None, compute_full=False):
        return  self.log_prior(params) + self.log_likelihood(params, key, compute_full)

    def optimize(self, batch_size=None):

        # Define training objective and gradient of objective using JaX.
        @jit
        def objective(params, key):
            return -self.log_joint(params, key=key)
            
        # prepare gradient and optimizer
        objective_grad = grad(objective)
        optimizer = optax.adam(self.step_size)
        key = random.PRNGKey(self.seed)

        # Initialize parameters of the model + optimizer.
        params = self.params
        opt_state = optimizer.init(params)

        # Optimization loop
        for i in range(self.num_iters):
            key, key_opt = random.split(key)
            grads = objective_grad(params, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if self.verbose and (i+1) % int(self.num_iters/20) == 0:
                loss_ = self.log_joint(params, compute_full=True)
                print(f'Itt = {i+1}/{self.num_iters} ({100*(i+1)/self.num_iters}%), log joint = {loss_:3.2f}')



        self.params = params

        return self

    

