from functools import partial
from jax import numpy as jnp
from jax import random as jr
from jax import vmap

from ssm_jax.bp.gauss_bp_utils import potential_from_conditional_linear_gaussian, pair_cpot_condition
from ssm_jax.bp.gauss_factor_graph import (GaussianVariableNode,
                                           CanonicalFactor,
                                           CanonicalPotential,
                                           GaussianFactorGraph)
from ssm_jax.linear_gaussian_ssm.info_inference import lgssm_info_smoother
from ssm_jax.linear_gaussian_ssm.info_inference_test import build_lgssm_moment_and_info_form

_all_close = lambda x,y: jnp.allclose(x,y,rtol=1e-3, atol=1e-3)

def canonical_factor_from_clg(A,u,Lambda,x,y,factorID):
    (Kxx, Kxy, Kyy), (hx,hy) = potential_from_conditional_linear_gaussian(A,u,Lambda)
    K = jnp.block([[Kxx, Kxy],
                    [Kxy.T, Kyy]])
    h = jnp.concatenate((hx,hy))
    cpot = CanonicalPotential(eta=h, Lambda=K)
    return CanonicalFactor(factorID, [x,y], cpot)

def factor_graph_from_lgssm(lgssm_params,inputs, obs, T=None):
    
    if inputs is None:
        if T is not None:
            D_in = lgssm_params.dynamics_input_weights.shape[1]
            inputs = jnp.zeros((T, D_in))
        else:
            raise ValueError("One of `inputs` or `T` must not be None.")

    num_timesteps = len(inputs)
    Lambda0, mu0 = lgssm_params.initial_precision, lgssm_params.initial_mean
    prior_pot = (Lambda0, Lambda0 @ mu0)
    latent_dim = len(mu0)

    latent_vars = [GaussianVariableNode(f"x{i}", latent_dim) for i in range(num_timesteps)]
    x0 = latent_vars[0]
    x0.prior = CanonicalPotential(eta = Lambda0 @ mu0, Lambda = Lambda0)
    
    B, b = lgssm_params.dynamics_input_weights, lgssm_params.dynamics_bias
    F, Q_prec = lgssm_params.dynamics_matrix, lgssm_params.dynamics_precision
    latent_net_inputs = vmap(jnp.dot, (None, 0))(B, inputs) + b
    latent_factors = [canonical_factor_from_clg(F,latent_net_inputs[i], Q_prec,
                                                latent_vars[i], latent_vars[i+1],
                                                f"latent_{i},{i+1}")
                      for i in range(num_timesteps-1)]
    
    D, d = lgssm_params.emission_input_weights, lgssm_params.emission_bias
    H, R_prec = lgssm_params.emission_matrix, lgssm_params.emission_precision
    
    emission_net_inputs = vmap(jnp.dot, (None, 0))(D, inputs) + d
    emission_pots = vmap(potential_from_conditional_linear_gaussian, (None, 0, None))(H, emission_net_inputs, R_prec)
    local_evidence_pots = vmap(partial(pair_cpot_condition, obs_var=2))(emission_pots, obs)
    
    emission_factors = [CanonicalFactor(f"emission_{i}",
                                        [latent_var],
                                        CanonicalPotential(eta,Lambda))
                        for i, latent_var, Lambda, eta in zip(range(num_timesteps), latent_vars, *local_evidence_pots)]
    
    fg = GaussianFactorGraph()
    for x in latent_vars:
        fg.add_var_node(x)

    for latent_fact in latent_factors:
        fg.add_factor(latent_fact)

    for em_fact in emission_factors:
        fg.add_factor(em_fact)

    fg.set_messages_to_zero()
    
    return fg




def test_gauss_factor_graph_lgssm():
    """Test that Gaussian chain belief propagation gets the same results as 
     information form RTS smoother."""

    lgssm, lgssm_info = build_lgssm_moment_and_info_form()

    key = jr.PRNGKey(111)
    num_timesteps = 5 # Fewer timesteps so that we can run fewer iterations.
    input_size = lgssm.dynamics_input_weights.shape[1]
    inputs = jnp.zeros((num_timesteps, input_size))
    _, y = lgssm.sample(key, num_timesteps, inputs)

    lgssm_info_posterior = lgssm_info_smoother(lgssm_info, y, inputs)

    fg = factor_graph_from_lgssm(lgssm_info,inputs, y)

    for _ in range(num_timesteps):
        fg.synchronous_iteration()

    fg_etas = jnp.vstack([var.belief.eta for var in fg.var_nodes])
    fg_Lambdas = jnp.vstack([var.belief.Lambda[None,...] for var in fg.var_nodes])
    assert _all_close(fg_etas, lgssm_info_posterior.smoothed_etas)
    assert _all_close(fg_Lambdas, lgssm_info_posterior.smoothed_precisions)