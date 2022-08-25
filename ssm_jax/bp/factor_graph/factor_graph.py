import jax.numpy as jnp
from typing import List, Callable, Optional, Union
from .gaussian import Gaussian, MeasModel
from .factor import Factor
from .variable_node import VariableNode
import random

"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""

class GBPSettings:
    def __init__(self, damping: float = 0.0) -> None:
        # Parameters for damping the eta component of the message
        self.damping = damping


class FactorGraph:
    def __init__(self, gbp_settings: GBPSettings = GBPSettings()) -> None:
        self.var_nodes = []
        self.factors = []
        self.gbp_settings = gbp_settings

    def add_var_node(
        self,
        dofs: int,
        prior_mean: Optional[jnp.array] = None,
        prior_diag_cov: Optional[Union[float, jnp.array]] = None,
        properties: dict = {},
    ) -> None:
        variableID = len(self.var_nodes)
        self.var_nodes.append(VariableNode(variableID, dofs, properties=properties))
        if prior_mean is not None and prior_diag_cov is not None:
            prior_cov = jnp.zeros((dofs, dofs), dtype=prior_diag_cov.dtype)
            prior_cov = prior_cov.at[:, :].set(prior_diag_cov)
            self.var_nodes[-1].prior.set_with_cov_form(prior_mean, prior_cov)
            self.var_nodes[-1].update_belief()

    def add_factor(
        self, adj_var_ids: List[int], measurement: jnp.array, meas_model: MeasModel, properties: dict = {}
    ) -> None:
        factorID = len(self.factors)
        adj_var_nodes = [self.var_nodes[i] for i in adj_var_ids]
        new_factor = Factor(factorID, adj_var_nodes, measurement, meas_model, properties=properties)
        self.factors.append(new_factor)
        for var in adj_var_nodes:
            var.adj_factors.append(new_factor)

    def update_all_beliefs(self) -> None:
        for var_node in self.var_nodes:
            var_node.update_belief()

    def compute_all_messages(self, apply_dropout: bool = True) -> None:
        damping = self.gbp_settings.damping
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
                # f"Belief means: {self.belief_means().numpy()} --- "
                # f"Robust factors: {[factor.meas_model.loss.robust() for factor in self.factors]}"
                # f"Relins: {sum([(factor.iters_since_relin==0 and not factor.meas_model.linear) for factor in self.factors])}"
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

    def get_joint(self) -> Gaussian:
        """
        Get the joint distribution over all variables in the information form
        If nonlinear factors, it is taken at the current linearisation point.
        """
        dim = self.get_joint_dim()
        joint = Gaussian(dim)

        # Priors
        var_ix = [0] * len(self.var_nodes)
        counter = 0
        for var in self.var_nodes:
            var_ix[var.variableID] = int(counter)
            joint.eta[counter : counter + var.dofs] += var.prior.eta
            joint.lam[counter : counter + var.dofs, counter : counter + var.dofs] += var.prior.lam
            counter += var.dofs

        # Other factors
        for factor in self.factors:
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                vID = adj_var_node.variableID
                # Diagonal contribution of factor
                joint.eta[var_ix[vID] : var_ix[vID] + adj_var_node.dofs] += factor.factor.eta[
                    factor_ix : factor_ix + adj_var_node.dofs
                ]
                joint.lam[
                    var_ix[vID] : var_ix[vID] + adj_var_node.dofs, var_ix[vID] : var_ix[vID] + adj_var_node.dofs
                ] += factor.factor.lam[
                    factor_ix : factor_ix + adj_var_node.dofs, factor_ix : factor_ix + adj_var_node.dofs
                ]
                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if other_adj_var_node.variableID > adj_var_node.variableID:
                        other_vID = other_adj_var_node.variableID
                        # Off diagonal contributions of factor
                        joint.lam[
                            var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                            var_ix[other_vID] : var_ix[other_vID] + other_adj_var_node.dofs,
                        ] += factor.factor.lam[
                            factor_ix : factor_ix + adj_var_node.dofs,
                            other_factor_ix : other_factor_ix + other_adj_var_node.dofs,
                        ]
                        joint.lam[
                            var_ix[other_vID] : var_ix[other_vID] + other_adj_var_node.dofs,
                            var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                        ] += factor.factor.lam[
                            other_factor_ix : other_factor_ix + other_adj_var_node.dofs,
                            factor_ix : factor_ix + adj_var_node.dofs,
                        ]
                    other_factor_ix += other_adj_var_node.dofs
                factor_ix += adj_var_node.dofs

        return joint

    def MAP(self) -> jnp.array:
        return self.get_joint().mean()

    def dist_from_MAP(self) -> jnp.array:
        return jnp.linalg.norm(self.get_joint().mean() - self.belief_means())

    def belief_means(self) -> jnp.array:
        """Get an array containing all current estimates of belief means."""
        return jnp.concatenate([var.belief.mean() for var in self.var_nodes])

    def belief_covs(self) -> List[jnp.array]:
        """Get a list containing all current estimates of belief covariances."""
        covs = [var.belief.cov() for var in self.var_nodes]
        return covs

    def print(self, brief=False) -> None:
        print("\nFactor Graph:")
        print(f"# Variable nodes: {len(self.var_nodes)}")
        if not brief:
            for i, var in enumerate(self.var_nodes):
                print(f"Variable {i}: connects to factors {[f.factorID for f in var.adj_factors]}")
                print(f"    dofs: {var.dofs}")
                print(f"    prior mean: {var.prior.mean()}")
                print(f"    prior covariance: diagonal sigma {jnp.diag(var.prior.cov())}")
        print(f"# Factors: {len(self.factors)}")
        if not brief:
            for i, factor in enumerate(self.factors):
                if factor.meas_model.linear:
                    print("Linear", end=" ")
                else:
                    print("Nonlinear", end=" ")
                print(f"Factor {i}: connects to variables {factor.adj_vIDs}")
                print(
                    f"    measurement model: {type(factor.meas_model).__name__},"
                    f" {type(factor.meas_model.loss).__name__},"
                    f" diagonal sigma {jnp.diag(factor.meas_model.loss.effective_cov)}"
                )
                print(f"    measurement: {factor.measurement}")
        print("\n")
