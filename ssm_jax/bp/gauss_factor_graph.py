import jax
import jax.numpy as jnp
from jax.scipy import linalg
from typing import List, NamedTuple, Optional, Tuple

from ssm_jax.bp.gauss_bp_utils import info_multiply, info_divide

class CanonicalPotential(NamedTuple):
    eta: jnp.ndarray
    Lambda: jnp.ndarray

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
        self.belief = None
        self.messages_to_factors = dict()

    def update_belief(self) -> None:
        """Update local belief estimate by taking product of all incoming messages along all edges."""
        belief = jax.tree_map(lambda l: l.copy(), self.prior)
        for factor in self.adj_factors:
            message = factor.messages_to_vars[self.variableID]
            belief = info_multiply(belief, message)
        self.belief = belief

    def set_messages_to_zero(self) -> None:
        for factor in self.adj_factors:
            self.messages_to_factors[factor.factorID] = zeros_canonical_pot(self.dim)


class CanonicalFactor:
    def __init__(self, id: int, adj_var_nodes: List[GaussianVariableNode], factor: CanonicalPotential):
        self.factorID = id
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = [var.variableID for var in adj_var_nodes]
        self.factor = factor
        self.dim = self.factor.Lambda.shape[0]
        self.messages_to_vars = dict()

        assert self.dim == sum([var.dim for var in adj_var_nodes])

    def set_messages_to_zero(self):
        for var in self.adj_var_nodes:
            self.messages_to_vars[var.variableID] = zeros_canonical_pot(var.dim)

    def compute_messages(self, damping: float = 0.0) -> None:
        # TODO: Make this code vaguely comprehensible.
        """Compute all outgoing messages from the factor."""
        messages_eta, messages_lam = [], []

        start_dim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = self.factor.eta.clone(), self.factor.lam.clone()
            # Take product of factor with incoming messages other than message from
            #  the current variable.
            start = 0
            for var in self.adj_var_nodes:
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_factor = eta_factor.at[start : start + var_dofs].add(
                        self.adj_var_nodes[var].belief.eta - self.messages[var].eta
                    )
                    lam_factor = lam_factor.at[start : start + var_dofs, start : start + var_dofs].add(
                        self.adj_var_nodes[var].belief.lam - self.messages[var].lam
                    )

                start += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            mess_dofs = self.adj_var_nodes[v].dofs
            # Extract the relevant section of the joint potential vector.
            eo = eta_factor[start_dim : start_dim + mess_dofs]
            # Combine the rest of the joint eta vector.
            eno = jnp.concatenate((eta_factor[:start_dim], eta_factor[start_dim + mess_dofs :]))

            # Extract the relevant block of the joint precision matrix.
            loo = lam_factor[start_dim : start_dim + mess_dofs, start_dim : start_dim + mess_dofs]

            # This is the cross precision of current rest x var.
            lnoo = jnp.concatenate(
                (
                    lam_factor[:start_dim, start_dim : start_dim + mess_dofs],
                    lam_factor[start_dim + mess_dofs :, start_dim : start_dim + mess_dofs],
                ),
                axis=0,
            )
            # This is the remaining precision block rest x rest.
            lnono = jnp.concatenate(
                (
                    jnp.concatenate(
                        (lam_factor[:start_dim, :start_dim], lam_factor[:start_dim, start_dim + mess_dofs :]), axis=1
                    ),
                    jnp.concatenate(
                        (
                            lam_factor[start_dim + mess_dofs :, :start_dim],
                            lam_factor[start_dim + mess_dofs :, start_dim + mess_dofs :],
                        ),
                        axis=1,
                    ),
                ),
                axis=0,
            )

            G = linalg.solve(lnono, lnoo)
            new_message_lam = loo - G.T @ lnoo
            new_message_eta = eo - G.T @ eno
            messages_eta.append((1 - damping) * new_message_eta + damping * self.messages[v].eta)
            messages_lam.append((1 - damping) * new_message_lam + damping * self.messages[v].lam)
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            # Update saved messages.
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]



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

    def set_messages_to_zero(self):
        for factor in self.factors:
            factor.set_messages_to_zero()
        for var in self.var_nodes:
            var.set_messages_to_zero()

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
