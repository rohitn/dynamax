import jax
import jax.numpy as jnp
from jax.scipy import linalg
from typing import List, NamedTuple, Optional, Tuple

from ssm_jax.bp.gauss_bp_utils import info_multiply, info_divide, info_marginalize

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


class GaussianVariableNode:
    def __init__(self, id: int, dim: int, prior: Optional[CanonicalPotential] = None) -> None:
        self.variableID = id
        self.dim = dim
        self.adj_factors = []
        self.prior = prior if prior is not None else zeros_canonical_pot(self.dim)
        self.belief = jax.tree_map(lambda l: l.clone(), self.prior)

    def update_belief(self) -> None:
        """Update local belief estimate by taking product of all incoming messages along all edges."""
        belief = jax.tree_map(lambda l: l.clone(), self.prior)
        for factor in self.adj_factors:
            message = factor.messages_to_vars[self.variableID]
            belief = info_multiply(belief, message)
        self.belief = belief


class CanonicalFactor:
    def __init__(self, id: int, adj_var_nodes: List[GaussianVariableNode], factor: CanonicalPotential):
        self.factorID = id
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = [var.variableID for var in adj_var_nodes]
        self.factor = factor
        self.dim = self.factor.Lambda.shape[0]
        self.var_scopes = self._calculate_var_scopes()
        self.messages_to_vars = dict()

        assert self.dim == sum([var.dim for var in adj_var_nodes])


    def compute_messages(self, damping: float = 0.0) -> None:
        """Compute all outgoing messages from the factor."""
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
        eta, Lambda = self.factor.eta.clone(), self.factor.Lambda.clone()
        for var in self.adj_var_nodes:
            vID = var.variableID
            var_start, var_stop = self.var_scopes[vID]
            # TODO: make a function to wrap these two operations
            #  f(pot, message, scope) -> pot_plus_message
            eta = eta.at[var_start : var_stop].add(
                var.belief.eta - self.messages_to_vars[vID].eta
            )
            Lambda = Lambda.at[var_start:var_stop, var_start:var_stop].add(
                var.belief.Lambda - self.messages_to_vars[vID].Lambda
            )
        return CanonicalPotential(eta, Lambda)

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
        if eval_point is None:
            energy = sum([factor.get_energy() for factor in self.factors])
        else:
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
        return sum([var.dofs for var in self.var_nodes])

    def print(self, brief=False) -> None:
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
        print("\n")
