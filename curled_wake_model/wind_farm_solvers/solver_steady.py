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
#    turbulence_model = model_classes[nut_model](self, debug=False)

    # Find the plane for each turbine
    for turbine in self.turbines: turbine.n = np.abs(self.x - turbine.location[0]).argmin()        

    # Compute time
    start = time.time()
    print("Starting solver")

    # Initialize the tubulence model after creating the variables above (needs turbine.n)
    turbulence_model.initialize()

    # Initialize variables
    uw = self.uw
    vw = self.vw
    ww = self.ww
    uw.fill(0)
    vw.fill(0)
    ww.fill(0)

    dx = self.dx
    dy = self.dy
    dz = self.dz

    # Loop over downstream distances
    for i in range(1, self.Nx):
        U = self.U[i, :, :]
        V = self.V[i, :, :]
        W = self.W[i, :, :]

        # Perform RK step
        uw[i,:,:] = runge_kutta_step(uw[i-1,:,:], dx, U, V + vw[i-1,:,:], W+ ww[i-1,:,:], self.dy, self.dz, self.nu[i-1,:,:])
                
        # A simple evolution model to scale v the same way that U has scaled
        # This saves all the work of resolving the transport equation (du/dx~dv/dx)
        fact = (U + uw[i-1,:,:]) / (U + self.uw[i,:,:])
        # This ensures that V and W do not become larger (they should always decay)
        fact = np.clip(fact, .1, 1.)
        vw[i,:,:] = vw[i-1,:,:] * fact
        ww[i,:,:] = ww[i-1,:,:] * fact

        # This is used to check the stability of the numerical algorithm. 
        # It is recommended to leave this off (default) unless debugging
        if check_stability:
            U_tot = U+uw[i,:,:]
            # Let's make the nu stability check here
            nu_min_v = np.abs(self.dx * (V + vw[i-1,:,:])**2 / (2 * U_tot))
            nu_min_w = np.abs(self.dx * (W + ww[i-1,:,:])**2 / (2 * U_tot))
            nu_min = np.maximum(nu_min_v, nu_min_w)

            # Upper bound from Eq. B
            nu_max = np.abs(U_tot * self.dy**2 / (2 * self.dx))
            self.nu[i,:,:] = np.clip(self.nu[i,:,:], nu_min, np.maximum(nu_min, nu_max))

            # The total number of points where we expect an instability
            num_unstable = np.count_nonzero(nu_min > nu_max)
            if num_unstable > 0: 
                print(f'Instability detected at x={self.x[i]} in {num_unstable} points')

        # Add wake deficit from turbines
        for turbine in self.turbines:
            
            # Plane of the turbine
            n = turbine.n

            # Add the wake here
            if (n == i) and (turbine.state is True):
                #print('Activating turbine', str(j))
                uw[i,:,:] += turbine.initial_condition(
                    self.Y[i,:,:] - turbine.location[1], 
                    self.Z[i,:,:] - turbine.location[2], 
                    self.U[i-1, :, :] + uw[i-1,:,:], V=V,  # this is the same as in the dynamic model (plane before)
                    #sigma=max(2, 7/self.dy),
                    sigma=max(2, 15/self.dy),
                )

                # Add vortex effects
                if (turbine.alpha != 0) or (turbine.tilt !=0):
                    #print(f'Adding curl for {turbine.name}')
                    # Only account for curl in the vicinity of the turbine
                    cond = (np.abs(self.Y[i,:,:] - turbine.location[1]) < (turbine.D * cf)).nonzero()
                    # This function takes in a numpy indexed array and returns the modified version of the array with curl
                    vw[i, cond[0], cond[1]], ww[i,  cond[0], cond[1]] =  turbine.add_curl(
                                self.Y[i, cond[0], cond[1]] - turbine.location[1], 
                                self.Z[i, cond[0], cond[1]] - turbine.location[2], 
                                vw[i, cond[0], cond[1]], ww[i, cond[0], cond[1]]
                            )

                turbine.add_rotation(
                    self.Y[i,:,:] - turbine.location[1], 
                    self.Z[i,:,:] - turbine.location[2], 
                    vw[i,:,:], ww[i,:,:]
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
