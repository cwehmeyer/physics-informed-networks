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


class HarmonicOscillator(Hamiltonian):
    """
    Hamiltonian for a 1D harmonic oscillator.

    This implements a 1D harmonic oscillator H = p^2 / (2m) + k/2 * (q-q0)^2
    with mass `m`, spring constant `k`, and equilibrium position `q0`.

    Args:
        mass (float): Mass `m`, default is 1.
        spring_constant (float): Spring constant `k`, default is 1.
        equilibrium_position (float): Equilibrium position `q0`, default is 0.

    Raises:
        ValueError: If any parameter is not float or mass or spring constant are
            not positive.
    """

    def __init__(
        self,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        equilibrium_position: float = 0.0,
    ):
        super().__init__()
        if not isinstance(mass, (float, int)) or mass <= 0:
            raise ValueError(f"Invalid parameter {mass=:}: must be a positive float.")
        if not isinstance(spring_constant, (float, int)) or spring_constant <= 0:
            raise ValueError(
                f"Invalid parameter {spring_constant=:}: must be a positive float."
            )
        if not isinstance(equilibrium_position, (float, int)):
            raise ValueError(
                f"Invalid parameter {equilibrium_position=:}: must be a float."
            )
        self.mass = torch.as_tensor(mass)
        self.spring_constant = torch.as_tensor(spring_constant)
        self.equilibrium_position = torch.as_tensor(equilibrium_position)

    def epot(self, q: torch.Tensor) -> torch.Tensor:
        """Potential energy V(q) = k/2 * (q-q0)^2."""
        return 0.5 * self.spring_constant * (q - self.equilibrium_position) ** 2

    def ekin(self, p: torch.Tensor) -> torch.Tensor:
        """Kinetic energy T(p) = p^2 / (2m)."""
        return p**2 / (2 * self.mass)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Total energy H(q, p) = T(p) + V(q)."""
        if q.shape != q.shape:
            raise ValueError(
                f"Incompatible tensors {q.shape=:} ≠ {p.shape=:}: shapes must match."
            )
        return self.ekin(p) + self.epot(q)
