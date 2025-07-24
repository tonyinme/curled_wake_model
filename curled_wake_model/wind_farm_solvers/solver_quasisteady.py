#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  solver_quasisteady.py
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

"""
solver_quasisteady.py

This module implements the quasi-steady time integration solver for the
Computational Wake Model (CWM). It evolves the wind field in time by repeatedly
solving the steady-state problem at each time step with time-varying inflow,
turbine behavior, and layout configuration.

The main method provided is:
- solve_steady_state_in_time(self, ...): performs a pseudo-transient simulation by
  stepping forward in time and solving the steady flow field at each step.

Requirements:
- This solver is intended to be used as a method of the wind_farm_class.
- The class must define attributes such as U, V, W, turbines, Re, h, etc.
"""

from ..wind_farm_utils import *

import numpy as np
import time


def solve_steady_state_in_time(self, tn=100, dt=1, f=4,
    # Functions of the U and V large scale advection changes 
    # (as function of time)
    dVt=lambda t: 0,
    dUt=lambda t: 0,
    N=1,
    nut_model='standard',
    layout_time_fn = None,
    ):
    """
    Solve the Computational Wake Model (CWM) using a quasi-steady time-marching approach.

    At each time step, the method assumes the flow field reaches steady state and
    updates turbine states and inflow conditions accordingly. It is suitable for slowly
    evolving inflow conditions or layouts (e.g., floating turbines, ramped wind).

    Parameters
    ----------
    tn : float
        Total simulation time in seconds. The solver marches from t=0 to t=tn.
    dt : float
        Time step size in seconds.
    f : float, optional
        Scaling factor for the viscous term in the spatial solver (passed to `self.solve`). Default is 4.
    dVt : callable, optional
        Function dVt(t) that returns a time-dependent lateral velocity offset to be added to the background flow. Default is zero.
    dUt : callable, optional
        Function dUt(t) that returns a time-dependent streamwise velocity offset. Default is zero.
    N : int, optional
        Downsampling factor for saving snapshots to the time history arrays. Default is 1 (save every step).
    nut_model : str, optional
        Turbulence model to use in the steady-state solve. Options: 'standard', 'Scott', 'kl'. Default is 'standard'.
    layout_time_fn : callable, optional
        Function layout_time_fn(t) that returns updated turbine (x, y) positions at time `t`. Used for moving turbines.

    Returns
    -------
    None
        Updates self.u_time and self.time_video with velocity snapshots.
        Modifies self.turbines and internal flow fields in-place.
    """
    # Time-step of simulation
    self.dt = dt
    
    # Compute time
    start = time.time()
    print("Starting steady state time solver")
        
    # Initialize time
    t = 0
    # Time list of all the velocity components
    self.u_time = []
    self.time_video = []
    
    # The initial conditions
    U0 = self.U.copy()
    V0 = self.V.copy()
    W0 = self.W.copy()

    # Time loop        
    while t<=tn:

        # Adjust the background flow
        self.U = U0 + dUt(t)
        self.V = V0 + dVt(t)

        # Get the new turbine location
        if layout_time_fn:
            layout_x, layout_y = layout_time_fn(t)

        # If the turbine is off, don't activate it and set coefficient to zero
        for it, turbine in enumerate(self.turbines):

            # Update the turbine locations
            if layout_time_fn: turbine.location = (layout_x[it], layout_y[it], turbine.th)

            turbine.update_alpha(t)

            if not turbine.on_off(t):
                turbine.Ct=0
                turbine.Cp=0
                turbine.pwr=0
                turbine.state=False

            else:
                turbine.state=True

        # Adjust the viscosity here to ensure stability
        self.U = np.maximum(0.2 * U0, self.U)  # Jul 3, 2025 - ensure numerical stability
        self.nu = self.U * self.h / self.Re
        self.add_turbulence_model()

        # Run the steady state solver
        self.solve(f=f, nut_model=nut_model)

        # Let's update the turbine time history here
        # and initialize Cp and Ct to force calculation again
        for turbine in self.turbines:
            
            turbine.update_time_vars(t=t)
            turbine.Ct=None
            turbine.Cp=None

        # Populate the time history list
        if t % (N * dt) < dt:  # update only every N time-steps
            self.u_time.append(self.uw.copy() + self.U)
            self.time_video.append(t)

        # Update the time
        t+=dt
        # Save time in class
        self.t = t  
        
        print('Finished time step, t=', t, '[s]')
        # Compute runtime
        end_ts = time.time()
        print("Quasi steady solver running time so far=", "{:.2f}".format(end_ts - start), 's')
        print("Simulated time so far=", t, 's')
        print("The solver is performing ", "{:.2f}".format(t/(end_ts-start)), 'time faster than realtime')

    
    print("Solver finished")    
    # Compute runtime
    end = time.time()
    print("Time solver running time=", "{:.2f}".format(end - start), 's')
    print("Simulated time=", tn, 's')
    print("The solver was ", "{:.2f}".format(tn/(end-start)), 'time faster than realtime')
