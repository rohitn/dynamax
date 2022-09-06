from jax import numpy as jnp
from jax import random as jr
from jax import vmap, jit

from ssm_jax.distributions import InverseWishart
from ssm_jax.bp.gauss_bp_utils import potential_from_conditional_linear_gaussian
from ssm_jax.bp.gauss_factor_graph import (GaussianVariableNode,
                                           CanonicalFactor,
                                           CanonicalPotential,
                                           GaussianFactorGraph,
                                           canonical_factor_from_clg,
                                           factor_graph_from_lgssm)
from ssm_jax.linear_gaussian_ssm.info_inference import lgssm_info_smoother
from ssm_jax.linear_gaussian_ssm.info_inference_test import build_lgssm_moment_and_info_form

_all_close = lambda x,y: jnp.allclose(x,y,rtol=1e-3, atol=1e-3)


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


def test_tree_factor_graph():
    key = jr.PRNGKey(0)
    dim = 2

    ### Construct variables in moment form ###
    IW = InverseWishart(dim, jnp.eye(dim)*0.1)
    key, subkey = jr.split(key)
    covs = jit(IW.sample,static_argnums=0)(5,subkey)

    key, subkey1  = jr.split(key)
    mu1 = jr.normal(subkey1,(dim,))
    Sigma1 = covs[0]

    key, subkey = jr.split(key)
    mu2 = jr.normal(subkey,(dim,))
    Sigma2 = covs[1]

    # x_3 | x_1, x_2 ~ N(x_3| A_31 x_1 + A_32 x_2, Sigma_{3|1,2})
    key, *subkeys = jr.split(key,3)
    A31 = jr.normal(subkeys[0],(dim,dim))
    A32 = jr.normal(subkeys[1],(dim,dim))
    Sigma3_cond = covs[2]
    mu3 = A31 @ mu1 + A32 @ mu2
    Sigma3 = Sigma3_cond + A31 @ Sigma1 @ A31.T + A32 @ Sigma2 @ A32.T

    # x_4 | x_3 ~ N(x_4 | A_4 x_3, Sigma_{3|4})
    key, subkey = jr.split(key)
    A4 = jr.normal(subkey,(dim,dim))
    Sigma4_cond = covs[3]
    mu4 = A4 @ mu3 
    Sigma4 = Sigma4_cond + A4 @ Sigma3 @ A4.T

    # x_5 | x_3 ~ N(x_5 | A_5 x_3, Sigma_{5|4})
    key, subkey = jr.split(key)
    A5 = jr.normal(subkey,(dim,dim))
    Sigma5_cond = covs[4]
    mu5 = A5 @ mu3 
    Sigma5 = Sigma5_cond + A5 @ Sigma3 @ A5.T

    ### Construct variables and factors in Canonical Form ###
    Lambda1 = jnp.linalg.inv(Sigma1)
    eta1 = Lambda1 @ mu1
    prior_x1 = CanonicalPotential(eta1, Lambda1)

    Lambda2 = jnp.linalg.inv(Sigma2)
    eta2 = Lambda2 @ mu2
    prior_x2 = CanonicalPotential(eta2, Lambda2)

    x1_var = GaussianVariableNode(1, dim, prior_x1)
    x2_var = GaussianVariableNode(2, dim, prior_x2)
    x3_var = GaussianVariableNode(3, dim)
    x4_var = GaussianVariableNode(4, dim)
    x5_var = GaussianVariableNode(5, dim)

    offset = jnp.zeros(dim)
    Lambda3_cond = jnp.linalg.inv(Sigma3_cond)
    A3_joint = jnp.hstack((A31,A32))
    (Kxx, Kxy, Kyy), (hx,hy) = potential_from_conditional_linear_gaussian(A3_joint,
                                                                          offset,
                                                                          Lambda3_cond)
    K = jnp.block([[Kxx, Kxy],
                    [Kxy.T, Kyy]])
    h = jnp.concatenate((hx,hy))
    cpot_123 = CanonicalPotential(eta=h, Lambda=K)
    factor_123 = CanonicalFactor("factor_123", [x1_var, x2_var, x3_var], cpot_123)

    Lambda4_cond = jnp.linalg.inv(Sigma4_cond)
    factor_34 = canonical_factor_from_clg(A4, offset, Lambda4_cond, x3_var, x4_var, "factor_34")

    Lambda5_cond = jnp.linalg.inv(Sigma5_cond)
    factor_35 = canonical_factor_from_clg(A5, offset, Lambda5_cond, x3_var, x5_var, "factor_35")

    # Build factor graph.
    fg = GaussianFactorGraph({"damping":0.})

    for var in [x1_var, x2_var, x3_var, x4_var, x5_var]:
        fg.add_var_node(var)
        
    for factor in [factor_123, factor_34, factor_35]:
        fg.add_factor(factor)
        
    fg.set_messages_to_zero()

    # Loopy BP
    for _ in range(10):
        fg.synchronous_iteration()
    
    # Extract marginal etas and Lambas from factor graph.
    fg_etas = jnp.vstack([var.belief.eta for var in fg.var_nodes])
    fg_Lambdas = jnp.stack([var.belief.Lambda for var in fg.var_nodes])

    # Convert to moment form
    fg_means = vmap(jnp.linalg.solve)(fg_Lambdas, fg_etas)
    fg_covs = jnp.linalg.inv(fg_Lambdas)

    means = jnp.vstack([mu1,mu2,mu3,mu4,mu5])
    covs = jnp.stack([Sigma1,Sigma2,Sigma3,Sigma4,Sigma5])

    # Compare to moment form marginals.
    assert jnp.allclose(fg_means,means,rtol=1e-2,atol=1e-2)
    assert jnp.allclose(fg_covs,covs,rtol=1e-2,atol=1e-2) 