"""
Sampling from and decoding an HMM
---------------------------------
This script shows how to sample points from a Hidden Markov Model (HMM):
we use a 4-state model with specified mean and covariance.
The plot shows the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.
Then, we decode our model to recover the ijnput parameters.

Based on
https://github.com/hmmlearn/hmmlearn/blob/main/examples/plot_hmm_sampling_and_decoding.py
"""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from ssm_jax.hmm.models.gaussian_hmm import GaussianHMM


def main(test_mode=False):
    # Prepare parameters for a 4-components HMM
    # Initial population probability
    initial_probabilities = jnp.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transition_matrix = jnp.array([[0.7, 0.2, 0.0, 0.1], [0.3, 0.5, 0.2, 0.0], [0.0, 0.3, 0.5, 0.2],
                                   [0.2, 0.0, 0.2, 0.6]])
    # The means of each component
    emission_means = jnp.array([[0.0, 0.0], [0.0, 11.0], [9.0, 10.0], [11.0, -1.0]])
    emission_dim = emission_means.shape[-1]

    # The covariance of each component
    emission_covariance_matrices = .5 * jnp.tile(jnp.identity(2), (4, 1, 1))

    # Build an HMM instance and set parameters
    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    gen_model = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covariance_matrices)

    # Generate samples
    key = jr.PRNGKey(0)
    states, emissions = gen_model.sample(key, 500)

    if not test_mode:
        # Plot the sampled data
        fig, ax = plt.subplots()
        ax.plot(emissions[:, 0], emissions[:, 1], ".-", label="observations", ms=6, mfc="orange", alpha=0.7)

        # Indicate the component numbers
        for i, m in enumerate(emission_means):
            ax.text(m[0],
                    m[1],
                    'Component %i' % (i + 1),
                    size=17,
                    horizontalalignment='center',
                    bbox=dict(alpha=.7, facecolor='w'))
        ax.legend(loc='best')
        fig.show()

    # %%
    # Now, let's ensure we can recover our parameters.

    scores = list()
    models = list()
    for num_states in (3, 4, 5):
        for idx in range(10):
            # define our hidden Markov model
            model = GaussianHMM.random_initialization(jr.PRNGKey(idx), num_states, emission_dim)
            lps = model.fit_em(emissions[None, :emissions.shape[0] // 2])  # 50/50 train/validate
            models.append(model)
            scores.append(model.log_prob(states[states.shape[0] // 2:], emissions[emissions.shape[0] // 2:]))

            print(f'\tScore: {scores[-1]}')

    # get the best model
    model = models[jnp.argmax(jnp.array(scores))]
    print(f'The best model had a score of {max(scores)} and {model.num_states} '
          'states')

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    states = model.most_likely_states(emissions)

    if not test_mode:
        # %%
        # Let's plot our states compared to those generated and our transition matrix
        # to get a sense of our model. We can see that the recovered states follow
        # the same path as the generated states, just with the identities of the
        # states transposed (i.e. instead of following a square as in the first
        # figure, the nodes are switch around but this does not change the basic
        # pattern). The same is true for the transition matrix.

        # plot model states over time
        fig, ax = plt.subplots()
        ax.plot(states, states)
        ax.set_title('States compared to generated')
        ax.set_xlabel('Generated State')
        ax.set_ylabel('Recovered State')
        plt.savefig("gaussian_hmm_states.png", dpi=300)
        fig.show()

        # plot the transition matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.imshow(gen_model.transition_matrix.value, aspect='auto', cmap='spring')
        ax1.set_title('Generated Transition Matrix')
        ax2.imshow(model.transition_matrix.value, aspect='auto', cmap='spring')
        ax2.set_title('Recovered Transition Matrix')
        for ax in (ax1, ax2):
            ax.set_xlabel('State To')
            ax.set_ylabel('State From')

        fig.tight_layout()
        plt.savefig("gaussian_hmm_transition_matrix.png", dpi=300)
        fig.show()


if __name__ == "__main__":
    main()
