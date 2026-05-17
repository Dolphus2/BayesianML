import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten
from autograd.scipy.special import logsumexp
from pytest import param

def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*np.log(2*np.pi*v)



#############################################################################
# Adapted from
# https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
#############################################################################

class NeuralNetwork(object):

    def __init__(self, X, t, layer_sizes, log_lik_fun, alpha=1., step_size=0.01, max_itt=1000, seed=0):

        # data
        self.X = X
        self.t = t

        # model and optimization parameters
        self.log_lik_fun = log_lik_fun
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.max_itt = max_itt
        self.alpha = alpha

        # random number genration
        self.seed=seed
        self.rng = np.random.default_rng(seed)
        
        # initialize parameters and optimize
        self.params = self.init_random_params()
        self.optimize()


    def init_random_params(self):
        """Build a list of (weights, biases) tuples,
        one for each layer in the net."""
        return [(np.sqrt(2/n) * self.rng.standard_normal((m, n)),   # weight matrix
                np.sqrt(2/n) * self.rng.standard_normal(n))      # bias vector
                for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def neural_net_predict(self, params, inputs):
        """Implements a deep neural network for classification.
        params is a list of (weights, bias) tuples.
        inputs is an (N x D) matrix.
        returns logits."""
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs# - logsumexp(outputs, axis=1, keepdims=True)

    def predict(self, inputs):
        return self.neural_net_predict(self.params, inputs)

    def log_prior(self, params):
        # implement a Gaussian prior on the weights
        flattened_params, _ = flatten(params)
        return  np.sum(log_npdf(flattened_params, 0., 1/self.alpha))
    
    def log_likelihood(self, params):     
        y = self.neural_net_predict(params, self.X)
        return np.sum(self.log_lik_fun(y.ravel(), self.t))

    def log_posterior(self, params):
        return self.log_prior(params) + self.log_likelihood(params)

    def optimize(self):
    
        # Define training objective and gradient of objective using autograd.
        def objective(params, iter):
            return -self.log_posterior(params)
            
        objective_grad = grad(objective)

        # optimize
        self.params = adam(objective_grad, self.params, step_size=self.step_size, num_iters=self.max_itt)

        return self

    
