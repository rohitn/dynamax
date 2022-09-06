import functools
from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.tree_util import tree_leaves
from typing import List, NamedTuple, Optional, Tuple

from ssm_jax.bp.gauss_bp_utils import (info_multiply,
                                       info_divide,
                                       info_marginalize, 
                                       potential_from_conditional_linear_gaussian,
                                       pair_cpot_condition)

class CanonicalPotential(NamedTuple):
    eta: jnp.ndarray
    Lambda: jnp.ndarray

def extract_canonical_potential_blocks(can_pot, idxs):
    """
    E.g. K = [[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24]]
        h = [0, 1, 2, 3, 4, 5]
        idxs = [1,2]
        gets split into:
            K11 - [[ 6,  7],
                   [11, 12]]

            K12 - [[ 5,  8,  9],
                   [10, 13, 14]]

            K22 - [[ 0,  3,  4],
                   [15, 18, 19],
                   [20, 23, 24]]
        and
            h1 - [1, 2]
            h2 - [0, 3, 4]

    Args:
        can_pot - a CanonicalPotential object (or similar tuple) with elements (h, K) where,
                    K - (D x D) precision matrix.
                    h - (D,) potential vector.
        idxs (N,) array of indices in 1,...,D.
    Returns:
        (K11, K12, K22), (h1, h2) - blocks of the potential parameters where:
            K11 (N x N) block of precision elements with row and column in `indxs`
            K12 (N x D-N) block of precision elements with row in `indxs` but column not in `indxs`.
            K22 (D-N x D-N) block of precision elements with neither row nor column in `indxs`.
            h1 (N,) elements of potential vector in `indxs`.
            h2 (D-N,) elements of potential vector not in `indxs`.
    """
    h, K = can_pot
    idx_range = jnp.arange(len(h))
    b = jnp.isin(idx_range,idxs)
    K11 = K[b,:][:,b]
    K12 = K[b,:][:,~b]
    K22 = K[~b,:][:,~b]
    h1 = h[b]
    h2 = h[~b]
    return (K11, K12, K22), (h1, h2)

def zeros_canonical_pot(dim):
    eta = jnp.zeros(dim)
    Lambda = jnp.zeros((dim,dim))
    return CanonicalPotential(eta=eta, Lambda=Lambda)

def clone_canonical_pot(cpot):
    eta = cpot.eta.clone()
    Lambda = cpot.Lambda.clone()
    return CanonicalPotential(eta, Lambda)

@partial(jax.jit,static_argnums=2)
def absorb_canonical_message(cpot,message,message_scope):
    var_start, var_stop = message_scope
    eta = cpot.eta.at[var_start : var_stop].add(message.eta)
    Lambda = cpot.Lambda.at[var_start:var_stop, var_start:var_stop].add(
        message.Lambda
    )
    return CanonicalPotential(eta, Lambda)


def _tree_reduce(function, tree, initializer=None, is_leaf=None):
    if initializer is None:
        return functools.reduce(function, tree_leaves(tree,is_leaf))
    else:
        return functools.reduce(function, tree_leaves(tree,is_leaf), initializer)

@jax.jit
def sum_reduce_cpots(cpots, initializer=None):
    return _tree_reduce(info_multiply, cpots, initializer, 
                        is_leaf=lambda l: isinstance(l,CanonicalPotential))

class GaussianVariableNode:
    def __init__(self, id: int, dim: int, prior: Optional[CanonicalPotential] = None) -> None:
        self.variableID = id
        self.dim = dim
        self.adj_factors = []
        # TODO: Put prior in a CanonicalFactor instead? 
        self.prior = prior if prior is not None else zeros_canonical_pot(self.dim)
        self.belief = jax.tree_map(lambda l: l.clone(), self.prior)

    def update_belief(self) -> None:
        """Update local belief estimate by taking product of all incoming messages along all edges."""
        prior = clone_canonical_pot(self.prior)
        cpots = [f.messages_to_vars[self.variableID] for f in self.adj_factors]
        belief = sum_reduce_cpots(cpots, prior)
        # belief = clone_canonical_pot(self.prior)
        # for factor in self.adj_factors:
        #     message = factor.messages_to_vars[self.variableID]
        #     belief = info_multiply(belief, message)
        self.belief = belief


class CanonicalFactor:
    def __init__(self, id: int, adj_var_nodes: List[GaussianVariableNode], potential: CanonicalPotential):
        self.factorID = id
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = [var.variableID for var in adj_var_nodes]
        self.potential = potential
        self.dim = self.potential.Lambda.shape[0]
        self.var_scopes = self._calculate_var_scopes()
        self.messages_to_vars = dict()
        
        assert self.dim == sum([var.dim for var in adj_var_nodes])


    def compute_messages(self, damping: float = 0.0) -> None:
        """Compute all outgoing messages from the factor."""
        # TODO: The current handling of single variable factors feels a bit ugly.
        # Could change zero_messages --> init_messages and set message to potential
        #  if len(adj_vars) == 1, then just pass here.
        if len(self.adj_var_nodes) == 1:
            vID = self.adj_var_nodes[0].variableID
            self.messages_to_vars[vID] = clone_canonical_pot(self.potential)
        else:
            pot_plus_all_messages = self._absorb_var_messages()
            for var in self.adj_var_nodes:
                vID = var.variableID
                marginal_pot = self._marginalise_onto_var(pot_plus_all_messages, vID)
                # Subtract off the message sent from the variable to this factor
                var_to_factor_message = info_divide(var.belief, self.messages_to_vars[vID])
                new_factor_to_var_message = info_divide(marginal_pot, var_to_factor_message)
                # Apply damping
                damped_message = jax.tree_map(lambda x,y: damping * x + (1-damping) * y,
                                            self.messages_to_vars[vID], new_factor_to_var_message)
                # TODO: Might need to replace nans with zeros somewhere here.
                # self.messages_to_vars[vID] = jax.tree_map(jnp.nan_to_num, damped_message)
                self.messages_to_vars[vID] = damped_message

    def _set_messages_to_zero(self):
        for var in self.adj_var_nodes:
            self.messages_to_vars[var.variableID] = zeros_canonical_pot(var.dim)
    
    def _calculate_var_scopes(self):
        var_dims = jnp.array([var.dim for var in self.adj_var_nodes])
        var_starts = jnp.concatenate((jnp.zeros(1),jnp.cumsum(var_dims)[:-1]))
        var_stops = var_starts + var_dims
        var_scopes = {var_id:(int(start), int(stop))
                      for var_id, start, stop in zip(self.adj_vIDs, var_starts, var_stops)}
        return var_scopes

    def _absorb_var_messages(self):
        pot = clone_canonical_pot(self.potential)
        for var in self.adj_var_nodes:
            vID = var.variableID
            var_message = info_divide(var.belief, self.messages_to_vars[vID])
            pot = absorb_canonical_message(pot,var_message, self.var_scopes[vID])
        return pot 

    def _marginalise_onto_var(self, potential, varID):
        var_idxs = jnp.arange(*self.var_scopes[varID])
        (K11, K12, K22), (h1, h2) = extract_canonical_potential_blocks(potential, var_idxs)
        K_marg, h_marg = info_marginalize(K22, K12.T, K11, h2, h1)
        return CanonicalPotential(eta=h_marg, Lambda=K_marg)



class GaussianFactorGraph:
    def __init__(self, settings = {"damping":0}) -> None:
        self.var_nodes = []
        self.factors = []
        self.settings = settings

    def add_var_node(self, var: GaussianVariableNode) -> None:
        self.var_nodes.append(var)

    def add_factor(self, factor: CanonicalFactor) -> None:
        self.factors.append(factor)
        for var in factor.adj_var_nodes:
            var.adj_factors.append(factor)

    def set_messages_to_zero(self) -> None:
        for factor in self.factors:
            factor._set_messages_to_zero()

    def update_all_beliefs(self) -> None:
        for var_node in self.var_nodes:
            var_node.update_belief()

    def compute_all_messages(self) -> None:
        damping = self.settings["damping"]
        for factor in self.factors:
            factor.compute_messages(damping)

    def synchronous_iteration(self) -> None:
        self.compute_all_messages()
        self.update_all_beliefs()

    def gbp_solve(
        self, n_iters: Optional[int] = 20, converged_threshold: Optional[float] = 1e-6, include_priors: bool = True
    ) -> None:
        energy_log = [self.energy()]
        print(f"\nInitial Energy {energy_log[0]:.5f}")
        i = 0
        count = 0
        not_converged = True
        while not_converged and i < n_iters:
            self.synchronous_iteration()

            energy_log.append(self.energy(include_priors=include_priors))
            print(
                f"Iter {i+1}  --- "
                f"Energy {energy_log[-1]:.5f} --- "
                # f"Belief means: {self.belief_means()} --- "
            )
            i += 1
            if abs(energy_log[-2] - energy_log[-1]) < converged_threshold:
                count += 1
                if count == 3:
                    not_converged = False
            else:
                count = 0

    def energy(self, eval_point: jnp.array = None, include_priors: bool = True) -> float:
        """Computes the sum of all of the squared errors in the graph using the appropriate local loss function."""
        var_dofs = jnp.array([v.dofs for v in self.var_nodes])
        var_ix = jnp.concatenate([0, jnp.cumsum(var_dofs, axis=0)[:-1]])
        energy = 0.0
        for f in self.factors:
            local_eval_point = jnp.concatenate(
                [eval_point[var_ix[v.variableID] : var_ix[v.variableID] + v.dofs] for v in f.adj_var_nodes]
            )
            energy += f.get_energy(local_eval_point)
        if include_priors:
            prior_energy = sum([var.get_prior_energy() for var in self.var_nodes])
            energy += prior_energy
        return energy

    def get_joint_dim(self) -> int:
        return sum([var.dim for var in self.var_nodes])

    def reset_beliefs(self):
        for var in self.var_nodes:
            var.belief = zeros_canonical_pot(var.dim)

    def print(self, brief=False, print_beliefs=False) -> None:
        print("\nFactor Graph:")
        print(f"# Variable nodes: {len(self.var_nodes)}")
        if not brief:
            for i, var in enumerate(self.var_nodes):
                print(f"Variable {i}: connects to factors {[f.factorID for f in var.adj_factors]}")
                print(f"    dim: {var.dim}")
        print(f"# Factors: {len(self.factors)}")
        if not brief:
            for i, factor in enumerate(self.factors):
                print(f"Factor {i}: connects to variables {factor.adj_vIDs}")
        if print_beliefs:
            for var in self.var_nodes:
                print(f"Variable {var.variableID} - beliefs:")
                print(f"eta: {var.belief.eta}")
                print(f"Lambda: {var.belief.Lambda}")
        print("\n")


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
