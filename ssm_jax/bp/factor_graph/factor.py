import jax.numpy as jnp
from typing import List, Callable, Optional, Union
from .variable_node import VariableNode
from .gaussian import Gaussian, MeasModel


class Factor:
    def __init__(
        self,
        id: int,
        adj_var_nodes: List[VariableNode],
        measurement: jnp.array,
        meas_model: MeasModel,
        type: jnp.dtype = float,
        properties: dict = {},
    ) -> None:

        self.factorID = id
        self.properties = properties

        self.adj_var_nodes = adj_var_nodes
        self.dofs = sum([var.dofs for var in adj_var_nodes])
        self.adj_vIDs = [var.variableID for var in adj_var_nodes]
        self.messages = [Gaussian(var.dofs) for var in adj_var_nodes]

        self.factor = Gaussian(self.dofs)
        self.linpoint = jnp.zeros(self.dofs, dtype=type)

        self.measurement = measurement
        self.meas_model = meas_model

        self.compute_factor()

    def get_adj_means(self) -> jnp.array:
        adj_belief_means = [var.belief.mean() for var in self.adj_var_nodes]
        return jnp.concatenate(adj_belief_means)

    def get_residual(self, eval_point: jnp.array = None) -> jnp.array:
        """Compute the residual vector."""
        if eval_point is None:
            eval_point = self.get_adj_means()
        return self.meas_model.meas_fn(eval_point) - self.measurement

    def get_energy(self, eval_point: jnp.array = None) -> float:
        """Computes the squared error using the appropriate loss function."""
        residual = self.get_residual(eval_point)
        # print("adj_belifes", self.get_adj_means())
        # print("pred and meas", self.meas_model.meas_fn(self.get_adj_means()), self.measurement)
        # print("residual", self.get_residual(), self.meas_model.loss.effective_cov)
        # TODO: replace the `inv` with a solve.
        return 0.5 * residual @ jnp.linalg.inv(self.meas_model.loss.effective_cov) @ residual

    def compute_factor(self) -> None:
        """
        Compute the factor at current adjacente beliefs using robust.
        If measurement model is linear then factor will always be the same regardless of linearisation point.
        """
        # TODO: can probably get rid of lots of this linearisation stuff.
        self.linpoint = self.get_adj_means()
        J = self.meas_model.jac_fn(self.linpoint)
        pred_measurement = self.meas_model.meas_fn(self.linpoint)
        self.meas_model.loss.get_effective_cov(pred_measurement - self.measurement)
        effective_lam = jnp.linalg.inv(self.meas_model.loss.effective_cov)
        JT_eff_lam = J.T @ effective_lam
        self.factor.lam = JT_eff_lam @ J
        self.factor.eta = ((JT_eff_lam) @ (J @ self.linpoint + self.measurement - pred_measurement)).flatten()

    def compute_messages(self, damping: float = 0.0) -> None:
        # TODO: Make this code vaguely comprehensible.
        """Compute all outgoing messages from the factor."""
        messages_eta, messages_lam = [], []

        start_dim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = self.factor.eta.clone(), self.factor.lam.clone()

            # Take product of factor with incoming messages
            start = 0
            for var in range(len(self.adj_vIDs)):
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
            eo = eta_factor[start_dim : start_dim + mess_dofs]
            eno = jnp.concatenate((eta_factor[:start_dim], eta_factor[start_dim + mess_dofs :]))

            loo = lam_factor[start_dim : start_dim + mess_dofs, start_dim : start_dim + mess_dofs]
            lono = jnp.concatenate(
                (
                    lam_factor[start_dim : start_dim + mess_dofs, :start_dim],
                    lam_factor[start_dim : start_dim + mess_dofs, start_dim + mess_dofs :],
                ),
                axis=1,
            )
            lnoo = jnp.concatenate(
                (
                    lam_factor[:start_dim, start_dim : start_dim + mess_dofs],
                    lam_factor[start_dim + mess_dofs :, start_dim : start_dim + mess_dofs],
                ),
                axis=0,
            )
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

            # TODO: replace inv with solve.
            new_message_lam = loo - lono @ jnp.linalg.inv(lnono) @ lnoo
            new_message_eta = eo - lono @ jnp.linalg.inv(lnono) @ eno
            messages_eta.append((1 - damping) * new_message_eta + damping * self.messages[v].eta)
            messages_lam.append((1 - damping) * new_message_lam + damping * self.messages[v].lam)
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]
