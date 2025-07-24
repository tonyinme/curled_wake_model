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

    # Find the plane for each turbine
    for turbine in self.turbines: turbine.n = np.abs(self.x - turbine.location[0]).argmin()        

    C_nu = 0.04
    if nut_model=='Scott': nut_x=True

    if nut_model=='kl': 

        f=1
        self.kw = np.zeros_like(self.uw)
        self.x_kl = np.zeros_like(self.uw)  # the distance from the wake plane
        self.k_tot = np.zeros_like(self.uw)  # the total tke
        self.lmix = np.full_like(self.uw, self.turbines[0].D)  # the mixing length array

        # Minimum viscosity for numerical stability
        #self.nu_min = self.U * self.h / self.Re

        for turbine in self.turbines:
            i = turbine.n

            # Create boolean mask over Y at slice i
            yz_mask = np.abs(self.Y[i, :, :] - turbine.location[1]) < (turbine.D * cf)
            # Offset along X axis starting from i
            offset = self.X[i:, :, :] - turbine.location[0]
            # Create full 3D mask for the i: range where offset > 5D and within yz region
            full_mask = np.zeros_like(offset, dtype=bool)
            full_mask[:, :, :] = yz_mask  # broadcast along x
            x0 = 2.5 * turbine.D
            valid = (offset > x0) & full_mask
            # Assign only where valid
            self.x_kl[i:, :, :][valid] = offset[valid] - x0

        # The minimum mixing length (rotor diameter)
        l_min = self.turbines[0].D

        # Field of the base k
        self.kb = (TI * self.U) ** 2 * 3 / 2
        self.nu = C_nu * np.sqrt(self.kb) * l_min

        def compute_lmix(du, z, l_min=l_min, l_max=None, du_min=0.5):
            """
            Compute mixing length from vertical extent of |du| > du_min, for each y.

            Parameters:
            - du: 2D array [ny, nz]
            - z: 1D array [nz]
            - l_min: minimum allowed mixing length
            - l_max: maximum allowed mixing length
            - du_min: threshold for |du|

            Returns:
            - lmix: 2D array [ny, nz] with constant lmix in z at each y
            """
            abs_du = np.abs(du)
            mask = abs_du > du_min

            ny, nz = du.shape
            lmix = np.full_like(du, l_min)

            # Find first and last non-zero index in each row (y-line)
            first = np.argmax(mask, axis=1)  # first True along z
            last = nz - 1 - np.argmax(mask[:, ::-1], axis=1)  # last True along z

            has_wake = mask.any(axis=1)
            z_low = np.where(has_wake, z[first], 0)
            z_high = np.where(has_wake, z[last], 0)
            extent = np.abs(z_high - z_low)

            if l_max is not None:
                extent = np.clip(extent, l_min, l_max)
            else:
                extent = np.maximum(extent, l_min)

            # Broadcast to all z
            return np.repeat(extent[:, np.newaxis], nz, axis=1)



        def compute_dkdx(dk, du, U, V, W, nu_T, lmix, dy, dz, C_k1=1, C_k2=1):
            """
            Computes the dk/dx term for the turbulence model.
            """                

            if np.any(lmix <= 0):
                raise IntegrationException("lmix is non-positive")

            # This is needed for a numerical instability near the wall
            term1 = np.gradient(du, dz, axis=1, edge_order=1) * np.gradient(U, dz, axis=1, edge_order=1)
            term1[:,[0,1,2]] = 0  # should try other fixes as well?

            term2 = np.gradient(nu_T * np.gradient(dk, dz, axis=1, edge_order=1), dz, axis=1, edge_order=1)
            term2[:,[0,1,2]] = 0 

            # transport equation for k_wake, written in parabolic form:
            dkdx = (
                -V * np.gradient(dk, dy, axis=0, edge_order=2)
                -W * np.gradient(dk, dz, axis=1, edge_order=2)
                + nu_T
                * (
                    np.gradient(du, dy, axis=0, edge_order=2) * np.gradient(U, dy, axis=0, edge_order=2)
                    + term1
                )
                + C_k1  # pull out of gradient as C_k1 is constant
                * (
                    np.gradient(nu_T * np.gradient(dk, dy, axis=0), dy, axis=0)
                    + term2
                )
                # need np.clip for the sqrt here
                - C_k2 * (np.clip(dk, 0, None) ** (3 / 2) / lmix)
            ) / U

            return dkdx

        def rk4_step_kwake(
            dk, du, U, V, W, nu_T, lmix, dy, dz, dx, compute_dkdx_func,
        ):
            """
            One RK4 step for integrating k_wake in x-direction.
            """
            k1 = compute_dkdx_func(dk, du, U, V, W, nu_T, lmix, dy, dz, )
            k2 = compute_dkdx_func(dk + 0.5 * dx * k1, du, U, V, W, nu_T, lmix, dy, dz, )
            k3 = compute_dkdx_func(dk + 0.5 * dx * k2, du, U, V, W, nu_T, lmix, dy, dz, )
            k4 = compute_dkdx_func(dk + dx * k3, du, U, V, W, nu_T, lmix, dy, dz, )

            dk_next = dk + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return dk_next

    

    # Compute time
    start = time.time()
    print("Starting solver")

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
        uw[i,:,:] = runge_kutta_step(uw[i-1,:,:], dx, U, V + vw[i-1,:,:], W+ ww[i-1,:,:], self.dy, self.dz, f, self.nu[i-1,:,:])
        
        
        # A simple evolution model to scale v the same way that U has scaled
        # This saves all the work of resolving the transport equation (du/dx~dv/dx)
        fact = (U + uw[i-1,:,:]) / (U + self.uw[i,:,:])
        # This ensures that V and W do not become larger
        fact = np.clip(fact, .1, 1.)
        vw[i,:,:] = vw[i-1,:,:] * fact
        ww[i,:,:] = ww[i-1,:,:] * fact

        if nut_model=='kl':

            # Calculate the mixing length
            self.lmix[i-1,:,:] = compute_lmix(uw[i-1,:,:], self.z)

            # This is the original implementation
            self.kw[i,:,:] = rk4_step_kwake(self.kw[i-1,:,:], uw[i-1,:,:], U + uw[i-1,:,:], 
                V + vw[i-1,:,:], W+ ww[i-1,:,:], self.nu[i-1,:,:], self.lmix[i-1,:,:], 
                dy, dz, dx, compute_dkdx)

            cond = self.x_kl[i, :, :] > 0
            yi, zi = np.where(cond)

            self.k_tot[i, :, :] = np.clip(self.kb[i, :, :] + self.kw[i, :, :], 0, None)
            self.nu[i, yi, zi] = C_nu * np.sqrt(self.k_tot[i, yi, zi]) * self.lmix[i, yi, zi]

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
#           for j, n in enumerate(self.activate):
        for turbine in self.turbines:
            
            # Plane of the turbine
            n = turbine.n #np.abs(self.x - turbine.location[0]).argmin()

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

                if nut_model=='Scott':
                    '''
                    Implement viscosity from R Scott et al, WES, 2023
                    '''
                    idx = i+1
                    x = (self.X[idx:, cond[0], cond[1]]-turbine.location[0]) #/ turbine.D  # dimensionless
                    sigma = 5.5  # dimensionless
                    A = turbine.D / 2 * turbine.Uh * np.sqrt(1-turbine.Ct) / 2
                    nu = A * (0.01 + x/sigma**2 * np.exp(-x**2/(2*sigma**2)))
                    print(f'plane {i}')
                    print(f'Turbine x {turbine.location[0]}')
                    print(f'A={A}')
                    print(f'D={turbine.D}')
                    print(f'Uh={turbine.Uh}')
                    print(nu.shape)
                    print(x.shape)
                    print(nu[:,-100])
                    self.nu[idx:, cond[0], cond[1]] = nu #np.maximum(nu, self.nu[idx:, cond[0], cond[1]])
                
        # Apply boundary conditions
        uw[i, :, [0, -1]] = 0
        uw[i, [0, -1], :] = 0

    if nut_model=='kl':
        field_names = ["kw", "nu", "kb", "k_tot", "x_kl", "lmix"]
        z_index = int(self.h / self.dz)
        z_index = 1
        n = len(field_names)

        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        # Don't use constrained_layout — we'll manage spacing manually
        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axs = np.ravel(axs)

        for i, field_name in enumerate(field_names):
            field = getattr(self, field_name)
            ax = axs[i]

            # Plot field
            pcm = ax.pcolormesh(self.x, self.y, field[:, :, z_index].T, shading='gouraud')
            ax.set_title(field_name)
            ax.set_aspect('equal')

            # Add matched-height colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)

        # Turn off unused axes
        for j in range(n, len(axs)):
            axs[j].axis('off')

        # Adjust layout to prevent overlap
        fig.tight_layout()

        # Save
        time_label = self.t if hasattr(self, "t") else 'unkown'
        #time_label = #self.time_video[-1] if self.time_video else "unknown"
        plt.savefig(f'{self.saveDir}/test_nut_{time_label}.png', dpi=150)
        plt.close(fig)



    print("Solver finished")

    # Compute runtime
    end = time.time()
    print("Solver running time=", end - start, 's')
