#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  solver_steady.py
#
#  Copyright 2025 Martinez Tossas
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

"""
solver_steady.py

This module implements the spatial steady-state solver for the curled wake model.
It solves the governing equations in x using a Runge-Kutta method and incorporates
turbulence, wake effects, and turbine-induced momentum deficits.

Main Function:
- solve(self, f=4, ...): Integrates the wind field downstream assuming steady inflow.

Expected `self`:
- Should be an instance of `wind_farm_class` with attributes like U, V, W, dx, nu_T, etc.
"""

from ..wind_farm_utils import *
from .turbulence_model import *

import numpy as np
import time


def solve(self, f=4, cf=2, rk_order=4, check_stability=False, nut_x=False, nut_model='standard', TI=0.16):
    """
    Solve the steady-state wind field using a general Runge-Kutta method.

    This method integrates the governing equations in the streamwise (x) direction
    assuming steady-state conditions. It supports several turbulence models and 
    optional stability checking. The default integration scheme is RK4.

    Parameters
    ----------
    f : float, optional
        Scaling factor for the viscous (diffusion) term. Default is 4.
        This is the scaling used as C in the 2021 CWM to scale the turbulent visocisty
    cf : float, optional
        Condition factor controlling the spatial influence of turbine-induced vortices. Default is 2.
    rk_order : int, optional
        Order of the Runge-Kutta scheme (2 or 4 supported). Default is 4.
    check_stability : bool, optional
        If True, enables stability diagnostics during integration. Default is False.
    nut_x : bool, optional
        If True, enables streamwise-dependent eddy viscosity behind turbines (per Scott et al., 2023). Default is False.
    nut_model : str, optional
        Choice of turbulence model. Options are:
        - 'standard': constant mixing length model
        - 'Scott': streamwise decay model (Scott et al., 2023)
        - 'kl': turbulent kinetic energy model (Howland Lab)
        Default is 'standard'.
    TI : float, optional
        Turbulence intensity used to initialize turbulence properties. Default is 0.16 (dimensionless).

    Returns
    -------
    None
        The solution is stored in-place on the instance attributes (e.g., self.U, self.k).
    """


    # Store these so that the turbulence class can access them
    self.TI = TI
    self.cf = cf

    # The turbulence model classes
    model_classes = {
    'standard': StandardTurbulenceModel,
    'Scott': ScottTurbulenceModel,
    'kl': KlTurbulenceModel,
    }

    # Optional inputs for the turbulence models
    model_kwargs = {
        'standard': {'C': f},
        'Scott': {},
        'kl': {}
    }

    # Load the appropriate turbulence model
    turbulence_model = model_classes[nut_model](self, **model_kwargs.get(nut_model, {}), debug=False)

    # Find the plane for each turbine
    for turbine in self.turbines: turbine.n = np.abs(self.x - turbine.location[0]).argmin()        

    # Compute time
    start = time.time()
    print("Starting solver")

    # Initialize the tubulence model after creating the variables above (needs turbine.n)
    turbulence_model.initialize()

    # Initialize variables
    U_all, V_all, W_all = self.U, self.V, self.W
    uw, vw, ww          = self.uw, self.vw, self.ww
    nu                   = self.nu
    dx, dy, dz           = self.dx, self.dy, self.dz
    Y_all, Z_all         = self.Y, self.Z
    turbines             = self.turbines
    cf_local             = self.cf
        
    # Zero wake fields
    uw.fill(0); vw.fill(0); ww.fill(0)

    # Scratch buffer to avoid allocating "fact" every step
    fact = np.empty_like(uw[0], dtype=uw.dtype)

    # Loop over downstream distances
    for i in range(1, self.Nx):
        U = U_all[i]
        V = V_all[i]
        W = W_all[i]
        U_prev = U_all[i-1]

        # Perform RK step
        uw[i] = runge_kutta_step(uw[i-1], dx, U, V + vw[i-1], W + ww[i-1], dy, dz, nu[i-1])

        # Simple evolution model for v,w magnitudes
        # fact = (U + uw[i-1]) / (U + uw[i]) with clipping in-place
        np.add(U, uw[i-1], out=fact)
        np.divide(fact, U + uw[i], out=fact)
        np.clip(fact, 0.1, 1.0, out=fact)
        vw[i] = vw[i-1] * fact
        ww[i] = ww[i-1] * fact

        # This is used to check the stability of the numerical algorithm. 
        # It is recommended to leave this off (default) unless debugging
        if check_stability:

            U_tot = U + uw[i]
            nu_min_v = np.abs(dx * (V + vw[i-1])**2 / (2 * U_tot))
            nu_min_w = np.abs(dx * (W + ww[i-1])**2 / (2 * U_tot))
            nu_min = np.maximum(nu_min_v, nu_min_w)
            nu_max = np.abs(U_tot * (dy**2) / (2 * dx))
            nu[i] = np.clip(nu[i], nu_min, np.maximum(nu_min, nu_max))
            num_unstable = np.count_nonzero(nu_min > nu_max)
            if num_unstable > 0:
                print(f'Instability detected at x={self.x[i]} in {num_unstable} points')

        # Add turbine wake effects on this plane
        Yi = Y_all[i]
        Zi = Z_all[i]
        for turb in turbines:
            if turb.n == i and (turb.state is True):
                uw[i] += turb.initial_condition(
                    Yi - turb.location[1],
                    Zi - turb.location[2],
                    U_prev + uw[i-1], V=V,
                    sigma=max(2, 15 / dy),
                )
                # Curl (only near the turbine)
                if (turb.alpha != 0) or (turb.tilt != 0):
                    cond = (np.abs(Yi - turb.location[1]) < (turb.D * cf_local)).nonzero()
                    vw[i, cond[0], cond[1]], ww[i, cond[0], cond[1]] = turb.add_curl(
                        Yi[cond] - turb.location[1],
                        Zi[cond] - turb.location[2],
                        vw[i, cond[0], cond[1]],
                        ww[i, cond[0], cond[1]],
                    )
                # Rotation (entire plane)
                turb.add_rotation(
                    Yi - turb.location[1],
                    Zi - turb.location[2],
                    vw[i], ww[i],
                )
                
        # This will update the variable nu
        # It should be done after the rest of the updates to ensure proper access to all updated quantities in turbines
        turbulence_model.update(i)
        
        # Apply boundary conditions
        uw[i, :, [0, -1]] = 0
        uw[i, [0, -1], :] = 0

    turbulence_model.postprocess()

    print("Solver finished")

    # Compute runtime
    end = time.time()
    print("Solver running time=", end - start, 's')
