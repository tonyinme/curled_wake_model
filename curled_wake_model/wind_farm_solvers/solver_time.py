#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  solver_time.py
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
solver_time.py

This module implements the time-dependent dynamic solver for the
Computational Wake Model (CWM). It advances the wake field in time using
a fourth-order Runge-Kutta (RK4) method and accounts for transient changes
in background flow and turbine operation.

The solver handles wake advection, time-varying inflow conditions, and
optional turbine initial condition smoothing.

Main Method:
- solve_in_time(self, ...): Performs full dynamic wake evolution using RK4 in time.

Requirements:
- Must be attached to a `wind_farm_class` instance with attributes like U, V, W, turbines, Re, h, etc.
- Requires `compute_rhs_dynamic()` and `add_turbulence_model()` functions.
"""

from ..wind_farm_utils import *

import numpy as np
import time


def solve_in_time(self, tn=100, dt=1, f=4,
    # Functions of the U and V large scale advection changes 
    # (as function of time)
    dVt=lambda t: 0,
    dUt=lambda t: 0,
    include_ic=True, # include the initial condition or not
    ic_time=0, # the initial condition time length
    N=1, # save data every N time-steps
    ):
    """
    Solve the Computational Wake Model (CWM) dynamically from t = 0 to t = tn using RK4 in time.

    This solver performs a full time-dependent integration of the wake field, using a Runge-Kutta
    scheme to advance the solution with time-varying inflow, wake interactions, and turbine control.
    It allows for optional smoothing of the initial condition using a quasi-steady phase.

    Parameters
    ----------
    tn : float
        Total simulation time in seconds. The solver marches from t=0 to t=tn.
    dt : float
        Time step size in seconds.
    f : float, optional
        Scaling factor for the viscous (diffusion) term in the wake advection equation.
    dVt : callable, optional
        Function dVt(t) that returns the lateral inflow adjustment at time t. Default is constant (0).
    dUt : callable, optional
        Function dUt(t) that returns the streamwise inflow adjustment at time t. Default is constant (0).
    include_ic : bool, optional
        If True, applies a smooth initial condition over a short ramp-in period (`ic_time`). Default is True.
    ic_time : float, optional
        Time duration for the initial condition ramp-in phase. Default is 0 (disabled).
    N : int, optional
        Downsampling rate for storing wake snapshots in time history. Saves data every N time steps. Default is 1.

    Returns
    -------
    None
        The final wake field is stored in `self.uw`.
        Time history snapshots are stored in `self.u_time` and `self.time_video`.
        Turbine power and control states are updated throughout the simulation.
    """
    
    # Time-step of simulation
    self.dt = dt

    # Compute time
    start = time.time()
    print("Starting time solver")

    # Initial conditions from the first time step
    #ic_ls = [np.zeros_like(self.V[0,:,:]) for t in self.turbines]
    #ic_ls = [t.ic for t in self.turbines]
    #ic_ls = [0] * len(self.turbines)

    # First, generate the initial condition
    '''
    if include_ic: 

        # The initial conditions
        U0 = self.U.copy()
        V0 = self.V.copy()

        # Adjust the background flow
        self.U += dUt(0.)
        self.V += dVt(0.)

        self.solve(f=f)
        for turbine in self.turbines: turbine.update_time_vars(t=0)

        self.U, self.V = U0, V0
    '''

    u_current = self.uw.copy()
                
    # Initialize time
    t = 0
    # Time list of all the velocity components
    self.u_time = [u_current + self.U + dUt(t)]
    self.time_video = [0]

    # Allocate memory once
    k1 = np.empty_like(u_current, dtype=float)
    k2 = np.empty_like(u_current, dtype=float)
    k3 = np.empty_like(u_current, dtype=float)
    k4 = np.empty_like(u_current, dtype=float)
    u_temp = np.empty_like(u_current, dtype=float)

    # Counter for the number of iterations
    iteration = 1
    # The iterations needed for the boundary condition
    preloop_iterations = int(ic_time/dt)

    # Time loop        
    while t<tn:

        # Only adjust for the initial condition
        if iteration < preloop_iterations:

            assert np.isclose(t, 0), f"Expected t == 0, but got t = {t}"

            U_now  = self.U + dUt(t)
            V_now  = self.V + dVt(t)
            U_half = U_now
            V_half = V_now
            U_full = U_now
            V_full = V_now
        
        else:
            U_now  = self.U + dUt(t)
            V_now  = self.V + dVt(t)
            U_half = self.U + dUt(t + dt / 2)
            V_half = self.V + dVt(t + dt / 2)
            U_full = self.U + dUt(t + dt)
            V_full = self.V + dVt(t + dt)

        # Adjust the viscosity here to ensure stability
        self.nu = U_now * self.h / self.Re
        self.add_turbulence_model()

        # --- RK4 steps using the Numba-compiled compute_rhs ---
        k1[:] = compute_rhs_dynamic(u_current, U_now, V_now, self.W, self.dx, self.dy, self.dz, f, self.nu)
        u_temp[:] = u_current + 0.5 * dt * k1
        k2[:] = compute_rhs_dynamic(u_temp, U_half, V_half, self.W, self.dx, self.dy, self.dz, f, self.nu)
        u_temp[:] = u_current + 0.5 * dt * k2
        k3[:] = compute_rhs_dynamic(u_temp, U_half, V_half, self.W, self.dx, self.dy, self.dz, f, self.nu)
        u_temp[:] = u_current + dt * k3
        k4[:] = compute_rhs_dynamic(u_temp, U_full, V_full, self.W, self.dx, self.dy, self.dz, f, self.nu)
        
        # Next time-step
        un = u_current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Add wake deficit from turbines
        for turbine in self.turbines:
            # Initialize to make sure that the code recalculates it
            turbine.Ct=None
            turbine.Cp=None

            turbine.update_alpha(t)

            # If the turbine is off, don't activate it and set coefficient to zero
            if not turbine.on_off(t):
                turbine.Ct=0
                turbine.Cp=0
                turbine.pwr=0
                if (iteration > preloop_iterations):
                    turbine.update_time_vars(t=t)
                continue

            # Plane of the turbine
            i = np.abs(self.x - turbine.location[0]).argmin()
            
            # The initial condition of this turbine
            Uic = U_now[i-1,:,:] + un[i-1,:,:] 
            Vic = V_now[i-1,:,:] 
            ic = turbine.initial_condition(
                self.Y[i,:,:] - turbine.location[1], 
                self.Z[i,:,:] - turbine.location[2], 
                # ~ self.U[i,:,:] + un[i,:,:], V=self.V[i,:,:]
                Uic,#-ic_ls[j], # subtract the initial condition from the last time to make sure to sample right inflow wout wake
                V=Vic,
                sigma=max(2, 15/self.dy),
            )

            # Only the wake area
            idx = np.nonzero(ic)                

            # Need to add 2 planes to better match steady state solution (old)
            # This probably has something to do with the numerics of the transient case (old)
            # This works well with the latest version (only one plane)
            un[i, idx[0], idx[1]] = ic[idx[0], idx[1]] + un[i-1, idx[0], idx[1]]

            # Let's update the turbine time history here
            #turbine.update_time_vars(t=t+dt)
            if (iteration > preloop_iterations):
                turbine.update_time_vars(t=t)


        # Set all boundary conditions in one go
        un[:, [0, -1], :] = 0
        un[:, :, [0, -1]] = 0

        # Sace the last time-step as the current to be used in next iteration
        u_current = un
        
        if (iteration > preloop_iterations):
            # Update the time
            t+=dt
            
            # Populate the time history list
            if (t % (N * dt) < dt):  # update only every N time-steps
                #self.u_time.append(self.uw.copy())            
#                self.u_time.append(u_current)
                self.u_time.append(u_current + U_full)

                self.time_video.append(t)

        iteration += 1

        print('Finished time step, t=', t, '[s]')
        # Compute runtime
        end_ts = time.time()
        print("Dynamic solver running time so far=", "{:.2f}".format(end_ts - start), 's')
        print("The dynamic solver is performing ", "{:.2f}".format(iteration*dt/(end_ts-start)), 'time faster than realtime')

    # Save the last time            
    self.uw = un
    
    print("Solver finished")    

    # Compute runtime
    end = time.time()
    print("Dynamic solver running time=", "{:.2f}".format(end - start), 's')
    print("Simulated time=", tn, 's')
    print("The dynamic solver was ", "{:.2f}".format(iteration*dt/(end-start)), 'time faster than realtime')
