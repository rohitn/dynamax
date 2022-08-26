from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.scipy.special import logsumexp
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM
from ssm_jax.utils import PSDToRealBijector


@chex.dataclass
class TiedGMMHMMSuffStats:
    # Wrapper for sufficient statistics of a TiedGaussianMixtureHMM
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    m: chex.Array
    c: chex.Array
    post_mix_sum: chex.Array
    post_sum: chex.Array


@register_pytree_node_class
class TiedGaussianMixtureHMM(StandardHMM):
    """
    Hidden Markov Model with Gaussian mixture emissions with tied covariance matrices.
    Attributes
    ----------
    weights : array, shape (num_states, num_emission_components)
        Mixture weights for each state.
    emission_means : array, shape (num_states, num_emission_components, emission_dim)
        Mean parameters for each mixture component in each state.
    emission_covariance_matrices : array
        Covariance parameters for each mixture components in each state.
    """

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 weights,
                 emission_means,
                 emission_covariance_matrix,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_mixture_weights_concentration=1.,
                 emission_prior_mean=0.,
                 emission_prior_mean_scale=0.,
                 emission_prior_covariance_matrices_concentration=-1.5,
                 emission_prior_covariance_matrices_scale=0.):

        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self._emission_mixture_weights = Parameter(weights, bijector=tfb.Invert(tfb.SoftmaxCentered()))
        self._emission_means = Parameter(emission_means)
        self._emission_cov = Parameter(emission_covariance_matrix, bijector=PSDToRealBijector)

        num_states, num_components, emission_dim = emission_means.shape

        _emission_mixture_weights_concentration = emission_mixture_weights_concentration * jnp.ones(
            (num_states, num_components)) if isinstance(emission_mixture_weights_concentration,
                                                        float) else emission_mixture_weights_concentration
        assert _emission_mixture_weights_concentration.shape == (num_states, num_components)
        self._emission_mixture_weights_concentration = Parameter(_emission_mixture_weights_concentration,
                                                                 is_frozen=True,
                                                                 bijector=tfb.Invert(tfb.Softplus()))

        _emission_prior_mean = emission_prior_mean * jnp.ones((num_states, num_components, emission_dim)) if isinstance(
            emission_prior_mean, float) else emission_prior_mean
        assert _emission_prior_mean.shape == (num_states, num_components, emission_dim)
        self._emission_prior_mean = Parameter(_emission_prior_mean, is_frozen=True)

        _emission_prior_mean_scale = emission_prior_mean_scale * jnp.ones(
            (num_states, num_components, emission_dim)) if isinstance(emission_prior_mean_scale,
                                                                      float) else emission_prior_mean_scale
        assert _emission_prior_mean_scale.shape == (num_states, num_components, emission_dim)
        self._emission_prior_mean_scale = Parameter(_emission_prior_mean_scale, is_frozen=True)

        _emission_prior_covariance_matrices_concentration = emission_prior_covariance_matrices_concentration * jnp.ones(
            (num_states, num_components, emission_dim)) if isinstance(
                emission_prior_covariance_matrices_concentration,
                float) else emission_prior_covariance_matrices_concentration
        assert _emission_prior_covariance_matrices_concentration.shape == (num_states, num_components, emission_dim)
        self._emission_prior_covariance_matrices_concentration = Parameter(
            _emission_prior_covariance_matrices_concentration, is_frozen=True)

        _emission_prior_covariance_matrices_scale = emission_prior_covariance_matrices_scale * jnp.ones(
            (num_states, num_components, emission_dim)) if isinstance(
                emission_prior_covariance_matrices_scale, float) else emission_prior_covariance_matrices_scale
        assert _emission_prior_covariance_matrices_scale.shape == (num_states, num_components, emission_dim)
        self._emission_prior_covariance_matrices_scale = Parameter(_emission_prior_covariance_matrices_scale,
                                                                   is_frozen=True)

    @classmethod
    def random_initialization(cls, key, num_states, num_components, emission_dim):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_mixture_weights = jr.dirichlet(key3, jnp.ones(num_components), shape=(num_states,))
        emission_means = jr.normal(key4, (num_states, num_components, emission_dim))
        emission_covs = jnp.eye(emission_dim) * jnp.ones((num_states, emission_dim, emission_dim))
        return cls(initial_probs, transition_matrix, emission_mixture_weights, emission_means, emission_covs)

    # Properties to get various parameters of the model

    @property
    def emission_mixture_weights(self):
        return self._emission_mixture_weights

    @property
    def emission_means(self):
        return self._emission_means

    @property
    def emission_covariance_matrix(self):
        return self._emission_cov

    def emission_distribution(self, state):
        _, num_components, emission_dim = self._emission_means.value.shape
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self._emission_mixture_weights.value[state]),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=self._emission_means.value[state],
                covariance_matrix=self._emission_cov.value[state] * jnp.ones(
                    (num_components, emission_dim, emission_dim))))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        _, num_components, emission_dim = self._emission_means.value.shape

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(), self._compute_transition_matrices(),
                                     self._compute_conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            def prob_fn(x):
                logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalFullCovariance(
                    loc=mus, covariance_matrix=sigmas * jnp.ones(
                        (num_components, emission_dim, emission_dim))).log_prob(x) + jnp.log(weights))(
                            self._emission_means.value, self._emission_cov.value, self._emission_mixture_weights.value)
                logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)

                return jnp.exp(logprobs)

            prob_denses = vmap(prob_fn)(emissions)

            post_comp_mix = posterior.smoothed_probs[:, :, None] * prob_denses
            post_mix_sum = post_comp_mix.sum(axis=0)
            post_sum = posterior.smoothed_probs.sum(axis=0)
            m_n = jnp.einsum('ijk,il->jkl', post_comp_mix, emissions)

            centered = emissions[:, None, None, :] - self._emission_means.value
            centered_dots = centered[..., :, None] * centered[..., None, :]

            c_n = jnp.einsum('ijk,ijklm->jklm', post_comp_mix, centered_dots)

            stats = TiedGMMHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                        initial_probs=initial_probs,
                                        trans_probs=trans_probs,
                                        m=m_n,
                                        c=c_n,
                                        post_mix_sum=post_mix_sum,
                                        post_sum=post_sum)

            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches

        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)
        self._emission_mixture_weights.value = stats.post_mix_sum / stats.post_sum[..., None]
        m_d = stats.post_mix_sum
        m_d = jnp.where((self._emission_mixture_weights.value == 0) & (stats.m == 0).all(axis=-1), jnp.ones_like(m_d),
                        m_d)[..., None]
        self._emission_means.value = stats.m / m_d

        c_d = m_d[..., None]
        emission_covs = stats.c / c_d
        emission_dim = batch_emissions.shape[-1]
        self._emission_cov.value = emission_covs + jnp.eye(emission_dim) * 1e-6
