import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.numpy as tfp
from jax import vmap
from ssm_jax.hmm.models.gmm_hmm import GaussianMixtureHMM
from ssm_jax.hmm.models.tests.test_utils import normalized



def sample_from_parallelepiped(key, low, high, n_samples):
    X = vmap(lambda l, h: jr.uniform(key, shape=(n_samples,), minval=l, maxval=h))(low, high)
    return X.T


def prep_params(key, n_comps, n_mix, n_features, low, high):
    # the idea is to generate ``n_comps`` bounding boxes and then
    # generate ``n_mix`` mixture means in each of them
    # this generates a sequence of coordinates, which are then used as
    # vertices of bounding boxes for mixtures
    dim_key, mean_key, trans_key, cov_key, weight_key = jr.split(key, 5)
    dim_lims = jnp.cumsum(jr.uniform(dim_key, shape=(n_comps, n_features), minval=low, maxval=high), axis=0)
    dim_lims = jnp.vstack([jnp.zeros((1, n_features)), dim_lims])

    keys = jr.split(mean_key, n_comps)
    emission_means = vmap(lambda key, left, right: sample_from_parallelepiped(key, left, right, n_mix))(keys,
                                                                                                        dim_lims[:-1],
                                                                                                        dim_lims[1:])
    initial_probabilities = jnp.append(jnp.ones((1,)), jnp.zeros((n_comps - 1,)))

    transition_matrix = normalized(jr.uniform(trans_key, shape=(n_comps, n_comps)), axis=1)

    keys = jr.split(cov_key, n_comps * n_mix)

    def make_covariance_matrix(key):
        low = jr.uniform(key, shape=(n_features, n_features), minval=-2, maxval=2)
        return jnp.dot(low.T, low)

    emission_covars = vmap(make_covariance_matrix)(keys).reshape((n_comps, n_mix, n_features, n_features))
    emission_weights = normalized(jr.uniform(weight_key, (n_comps, n_mix)), axis=1)

    return emission_covars, emission_means, initial_probabilities, transition_matrix, emission_weights


def new_hmm(key, num_states, num_mix, emission_dim, low, high):
    emission_covars, emission_means, initial_probabilities, transition_matrix, emission_weights = prep_params(
        key, num_states, num_mix, emission_dim, low, high)

    return GaussianMixtureHMM(initial_probabilities, transition_matrix, emission_weights, emission_means,
                              emission_covars)


def test_sample(key=jr.PRNGKey(0), num_states=3, num_mix=2, emission_dim=2, low=10., high=15., num_timesteps=100):
    init_key, sample_key = jr.split(key)
    h = new_hmm(init_key, num_states, num_mix, emission_dim, low, high)

    states, emissions = h.sample(sample_key, num_timesteps)
    assert emissions.shape == (num_timesteps, emission_dim)
    assert len(states) == num_timesteps


def test_fit(key=jr.PRNGKey(0), num_states=3, num_mix=2, emission_dim=2, low=10., high=15., n_samples=1000):
    key0, key1, key2 = jr.split(key, 3)
    true_hmm = new_hmm(key0, num_states, num_mix, emission_dim, low, high)
    _, emissions = true_hmm.sample(key2, n_samples)

    # Mess up the parameters and see if we can re-learn them.
    covs0, means0, priors0, trans0, weights0 = prep_params(key1, num_states, num_mix, emission_dim, low, high)
    hmm = GaussianMixtureHMM(priors0, trans0, weights0, means0, covs0 * 100)
    lps = hmm.fit_em(emissions[None, ...])
    print(lps)
    raise "err"


test_fit()
