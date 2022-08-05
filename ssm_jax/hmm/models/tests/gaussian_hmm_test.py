import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import vmap
from ssm_jax.hmm.models.gaussian_hmm import DiagonalGaussianHMM
from ssm_jax.hmm.models.gaussian_hmm import GaussianHMM
from ssm_jax.hmm.models.gaussian_hmm import SphericalGaussianHMM


def get_random_gaussian_hmm_params(key, num_states, num_emissions):
    initial_key, transition_key, means_key, covs_key = jr.split(key, 4)
    initial_probabilities = jr.uniform(initial_key, shape=(num_states,))
    initial_probabilities = initial_probabilities / initial_probabilities.sum()
    transition_matrix = jr.uniform(transition_key, shape=(num_states, num_states))
    transition_matrix = transition_matrix / jnp.sum(initial_probabilities, axis=-1, keepdims=True)
    emission_means = jr.randint(means_key, shape=(num_states, num_emissions), minval=-20.,
                                maxval=20).astype(jnp.float32)
    emission_covar_sqrts = jr.normal(covs_key, (num_states, num_emissions))
    emission_covars = jnp.einsum('ki, kj->kij', emission_covar_sqrts, emission_covar_sqrts)
    emission_covars += 1e-1 * jnp.eye(num_emissions)
    return initial_probabilities, transition_matrix, emission_means, emission_covars


def test_fit_means(key=jr.PRNGKey(0), num_states=3, num_emissions=3, num_samples=10):

    init_key, sample_key = jr.split(key, 2)
    initial_probabilities, transition_matrix, emission_means, emission_covars = get_random_gaussian_hmm_params(
        init_key, num_states, num_emissions)
    hmm = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covars)

    keys = jr.split(sample_key, num_samples)
    batch_states, batch_emissions = vmap(lambda rng: hmm.sample(rng, 10))(keys)

    # Mess up the parameters and see if we can re-learn them.
    hmm.transition_matrix.freeze()
    hmm.initial_probs.freeze()
    hmm.emission_covariance_matrices.freeze()

    losses = hmm.fit_sgd(batch_emissions)

    assert jnp.allclose(hmm.initial_probs.value, initial_probabilities)
    assert jnp.allclose(hmm.transition_matrix.value, transition_matrix)
    assert jnp.allclose(hmm.emission_covariance_matrices.value, emission_covars)


def test_fit_covars(key=jr.PRNGKey(0), num_states=3, num_emissions=3, num_samples=10):

    init_key, sample_key = jr.split(key, 2)
    initial_probabilities, transition_matrix, emission_means, emission_covars = get_random_gaussian_hmm_params(
        init_key, num_states, num_emissions)
    hmm = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covars)

    keys = jr.split(sample_key, num_samples)
    batch_states, batch_emissions = vmap(lambda rng: hmm.sample(rng, 10))(keys)

    # Mess up the parameters and see if we can re-learn them.
    # TODO: change the params and uncomment the check
    hmm.transition_matrix.freeze()
    hmm.initial_probs.freeze()
    hmm.emission_means.freeze()

    losses = hmm.fit_sgd(batch_emissions)
    assert jnp.allclose(hmm.initial_probs.value, initial_probabilities)
    assert jnp.allclose(hmm.transition_matrix.value, transition_matrix)
    assert jnp.allclose(hmm.emission_means.value, emission_means)


def test_fit_transition_matrix(key=jr.PRNGKey(0), num_states=3, num_emissions=3, num_samples=10):

    init_key, sample_key = jr.split(key, 2)
    initial_probabilities, transition_matrix, emission_means, emission_covars = get_random_gaussian_hmm_params(
        init_key, num_states, num_emissions)
    hmm = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covars)

    keys = jr.split(sample_key, num_samples)
    batch_states, batch_emissions = vmap(lambda rng: hmm.sample(rng, 10))(keys)

    # Mess up the parameters and see if we can re-learn them.
    # TODO: change the params and uncomment the check
    hmm.initial_probs.freeze()
    hmm.emission_covariance_matrices.freeze()
    hmm.emission_means.freeze()

    losses = hmm.fit_sgd(batch_emissions)
    assert jnp.allclose(hmm.initial_probs.value, initial_probabilities)
    assert jnp.allclose(hmm.emission_means.value, emission_means)
    assert jnp.allclose(hmm.emission_covariance_matrices.value, emission_covars)


class TestGaussianHMMWithDiagonalCovars:

    def setup(self):
        self.num_states = 3
        self.emission_dim = 3

    def test_fit_left_right(self, key=jr.PRNGKey(0), num_timesteps=100):
        transition_matrix = np.zeros((self.num_states, self.num_states))

        # Left-to-right: each state is connected to itself and its
        # direct successor.
        for i in range(self.num_states):
            if i == self.num_states - 1:
                transition_matrix[i, i] = 1.0
            else:
                transition_matrix[i, i] = transition_matrix[i, i + 1] = 0.5

        transition_matrix = jnp.array(transition_matrix)
        print(transition_matrix)
        # Always start in first state
        startprob = np.zeros(self.num_states)
        startprob[0] = 1.0
        startprob = jnp.array(startprob)

        true_hmm = DiagonalGaussianHMM.random_initialization(key, self.num_states, self.emission_dim)

        state_sequence, emissions = true_hmm.sample(key, num_timesteps)
        hmm = DiagonalGaussianHMM.random_initialization(key, self.num_states, self.emission_dim)
        hmm.initial_probs.value = startprob
        hmm.transition_matrix.value = transition_matrix

        lps = hmm.fit_em(emissions[None, ...])
        hmm.initial_probs.freeze()

        assert (hmm.initial_probs.value[startprob == 0.0] == 0.0).all()
        assert (hmm.transition_matrix.value[transition_matrix == 0.0] == 0.0).all()

        posteriors = hmm.filter(emissions[None, ...])
        assert not jnp.isnan(posteriors.filtered_probs).any()
        assert jnp.allclose(posteriors.smoothed_probs.sum(axis=1), 1.)
