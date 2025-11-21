from typing import Tuple

import torch

from hamiltonian import Hamiltonian


class Integrator:
    """
    Base class for a generic Integrator.

    Integrators act on a Hamiltonian's time derivatives to propagate the
    equations of motion over time, with an optional langevin thermostat.

    This implementation assumes that all quantities are expressed in a
    dimensionless form. The user is responsible for nondimensionalizing
    the Hamiltonian, coordinates, momenta, time step, and thermostat
    parameters such that the resulting equations of motion take the
    canonical form dq/dt = dH/dp and dp/dt = -dH/dq with unit mass and
    Boltzmann constant k_B = 1. No unit conversions or physical constants
    are applied internally.

    args:
        hamiltonian (Hamiltonian): The Hamiltonian object to integrate.
        delta_t (float): Time step size for a single propagation event.
        damping (float): Damping constant for the decay factor, default
            is 0 (no damping or perturbations).
        thermal_energy (float): Thermal energy scale `T * k_B` for the
            random perturbations, default is 0 (no perturbations).
        mass (float): Mass for the random perturbations, default is 1.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        delta_t: float,
        *,
        damping: float = 0.0,
        thermal_energy: float = 0.0,
        mass: float = 1.0,
    ):
        self.hamiltonian = hamiltonian
        self.delta_t = torch.as_tensor(delta_t)
        self.damping = torch.as_tensor(damping)
        self.thermal_energy = torch.as_tensor(thermal_energy)
        self.mass = torch.as_tensor(mass)
        if self.damping > 0:
            self.decay = torch.exp(-self.damping * self.delta_t)
            self.sigma = torch.sqrt(
                self.thermal_energy * self.mass * (1.0 - self.decay**2)
            )
            self.use_thermostat = True
        else:
            self.decay = torch.tensor(1.0)
            self.sigma = torch.tensor(0.0)
            self.use_thermostat = False

    def thermostat(self, p: torch.Tensor) -> torch.Tensor:
        """Apply thermostatting to the generalized momenta."""
        if self.sigma > 0:
            noise = torch.randn_like(p) * self.sigma
            return p * self.decay + noise
        return p * self.decay

    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single time step.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.

        Returns:
            torch.Tensor: Updated generalized coordinates.
            torch.Tensor: Updated generalized momenta.
        """
        raise NotImplementedError()

    def __call__(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        steps: int,
        *,
        keep: int = 1,
        time_offset: float = 0.0,
        keep_initial: bool = True,
        squeeze: bool = False,
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate the equations of motion.

        This performs multiple steps to sample the dynamic evolution
        of the generalized coordinates and momenta.

        Args:
            q (torch.Tensor): Generalized coordinates for one or more realizations,
                must match `p` in shape.
            p (torch.Tensor): Generalized momenta for one or more realizations,
                must match `q` in shape.
            steps (int): Number of steps to simulate.
            keep (int): Return only every keep-th step, the default `keep=1`
                returns all steps. If `keep=2` only every second step is returned.
                The choice `keep=0` returns only the final coordinates and momenta.
            time_offset (float): Shifts the returned time array, default is 0.
            keep_initial (bool): Whether to keep the initial state (time offset,
                coordinates, momenta) at the beginning of the returned tensors,
                default is `True`.

        Returns:
            torch.Tensor: Times of the sampled trajectories.
            torch.Tensor: Generalized coordinates' trajectory.
            torch.Tensor: Generalized momenta' trajectory.
        """
        time, qt, pt = [], [], []
        if keep_initial:
            time.append(0.0)
            qt.append(q.unsqueeze(0))
            pt.append(p.unsqueeze(0))
        q = q.clone().detach()
        p = p.clone().detach()
        for step in range(1, steps + 1):
            q, p = self.step(q, p)
            if keep > 0 and step % keep == 0:
                time.append(step * self.delta_t)
                qt.append(q.detach().unsqueeze(0))
                pt.append(p.detach().unsqueeze(0))
        if keep == 0:
            return steps * self.delta_t + time_offset, q, p
        return (
            torch.tensor(time) + time_offset,
            torch.cat(qt, dim=0),
            torch.cat(pt, dim=0),
        )
