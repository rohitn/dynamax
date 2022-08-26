import jax.numpy as jnp
import jax.random as jr
from ssm_jax.hmm.models import DiagonalGaussianMixtureHMM
from ssm_jax.utils import monotonically_increasing


def test_random_initialization(key=jr.PRNGKey(0), num_states=3, num_mix=4, emission_dim=2):
    hmm = DiagonalGaussianMixtureHMM.random_initialization(key, num_states, num_mix, emission_dim)

    assert hmm.initial_probs.value.shape == (num_states,)
    assert hmm.initial_probs.value.sum() == 1.

    assert hmm.transition_matrix.value.shape == (num_states, num_states)
    assert jnp.allclose(hmm.transition_matrix.value.sum(axis=-1), 1.)

    assert hmm.emission_mixture_weights.value.shape == (num_states, num_mix)
    assert jnp.allclose(hmm.emission_mixture_weights.value.sum(axis=-1), 1.)

    assert hmm.emission_means.value.shape == (num_states, num_mix, emission_dim)

    assert hmm.emission_cov_diag_factors.value.shape == (num_states, num_mix, emission_dim)


def test_sample(key=jr.PRNGKey(0), num_states=3, num_mix=4, emission_dim=2, num_timesteps=100):
    init_key, sample_key = jr.split(key)
    h = DiagonalGaussianMixtureHMM.random_initialization(init_key, num_states, num_mix, emission_dim)

    states, emissions = h.sample(sample_key, num_timesteps)
    assert emissions.shape == (num_timesteps, emission_dim)
    assert len(states) == num_timesteps


def test_fit(key=jr.PRNGKey(0), num_states=3, num_mix=4, emission_dim=2, n_samples=1000):
    key0, key1, key2 = jr.split(key, 3)
    true_hmm = DiagonalGaussianMixtureHMM.random_initialization(key0, num_states, num_mix, emission_dim)
    _, emissions = true_hmm.sample(key2, n_samples)

    # Mess up the parameters and see if we can re-learn them.
    hmm = DiagonalGaussianMixtureHMM.random_initialization(key1, num_states, num_mix, emission_dim)
    lps = hmm.fit_em(emissions[None, ...], 3)
    assert monotonically_increasing(lps)
