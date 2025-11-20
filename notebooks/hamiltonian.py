from typing import Tuple

import torch
import torch.nn as nn


def differentiate(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiate H w.r.t. x."""
    return torch.autograd.grad(
        H,
        x,
        grad_outputs=torch.ones_like(H),
        create_graph=True,
    )[0]


class Hamiltonian(nn.Module):
    """
    Base class for a generic Hamiltonian.

    This implementation assumes that all quantities are expressed in a
    dimensionless form. The user is responsible for nondimensionalizing
    the Hamiltonian, coordinates, momenta, time step, and thermostat
    parameters such that the resulting equations of motion take the
    canonical form dq/dt = dH/dp and dp/dt = -dH/dq with unit mass and
    Boltzmann constant k_B = 1. No unit conversions or physical constants
    are applied internally.
    """

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Evaluate H(q, p).

        Either implements a Hamiltonian function analytically or runs a
        PyTorch model. If multiple realizations are passed, arange them along
        the first axis in `q` and `p`: `q.shape = (batch, particles, dimensions)`.

        Iplement in derived classes.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.

        Returns:
            torch.Tensor: Evaluated Hamiltonian, `H(q, p)`.
        """
        raise NotImplementedError()

    def dq_dt(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate dq/dt = dH/dp.

        Compute the time-derivative of the generalized coordinates via the derivative
        of the Hamiltonian w.r.t. the generalized momenta.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.

        Returns:
            torch.Tensor: Time-derivative of the generalized coordinates, dq/dt = dH/dp.
        """
        q = q.clone().detach()
        p = p.clone().detach()
        p.requires_grad_(True)
        H = self(q, p)
        dH_dp = differentiate(H, p)
        # dq_dt = dH_dp
        return dH_dp

    def dp_dt(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate dp/dt = -dH/dq.

        Compute the time-derivative of the generalized momenta via the derivative
        of the Hamiltonian w.r.t. the generalized coordinates.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.

        Returns:
            torch.Tensor: Time-derivative of the generalized momenta, dp/dt = -dH/dq.
        """
        q = q.clone().detach()
        p = p.clone().detach()
        q.requires_grad_(True)
        H = self(q, p)
        dH_dq = differentiate(H, q)
        # dp_dt = -dH_dq
        return -dH_dq

    def time_derivatives(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate dq/dt = dH/dp and dp/dt = -dH/dq.

        Compute the time-derivative of the generalized coordinates and momenta via
        the derivatives of the Hamiltonian w.r.t. the generalized momenta and coordinates.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.

        Returns:
            torch.Tensor: Time-derivative of the generalized coordinates, dq/dt = dH/dp.
            torch.Tensor: Time-derivative of the generalized momenta, dp/dt = -dH/dq.
        """
        q = q.clone().detach()
        p = p.clone().detach()
        q.requires_grad_(True)
        p.requires_grad_(True)
        H = self.forward(q, p)
        dH_dq = differentiate(H, q)
        dH_dp = differentiate(H, p)
        # dq_dt = dH_dp, dp_dt = -dH_dq
        return dH_dp, -dH_dq

    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        delta_t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a time step for generalized coordinates and momenta.

        Propagate the generalied coordinates and momenta according to the Hamiltonian
        via a kick-drift symplectic Euler.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.
            delta_t (float): Length of the time step.

        Returns:
            torch.Tensor: Generalized coordinates after the time step.
            torch.Tensor: Generalized momenta after the time step.
        """
        p = p + delta_t * self.dp_dt(q, p)
        q = q + delta_t * self.dq_dt(q, p)
        return q, p
