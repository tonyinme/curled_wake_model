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
turbulence_model.py

This module defines the turbulence models to be used in the curled wake model. 
New turbulence models can be added in here.

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class TurbulenceModel:
    def __init__(self, wf, debug=False, **kwargs):
        self.wf = wf
        self.debug=debug

        self.params = kwargs  # optional: store for debugging/logging


    def initialize(self):
        pass

    def update(self, i):
        pass

    def postprocess(self):
        pass


class StandardTurbulenceModel(TurbulenceModel):
    def __init__(self, wf, **kwargs):
        super().__init__(wf, **kwargs)
        # Constant to scale mixing turbulent viscosity (default is 4)
        self.C = kwargs.get('C', 4.0)

    def initialize(self):
        self.wf.add_turbulence_model(C=self.C)


class ScottTurbulenceModel(TurbulenceModel):
    def update(self, i):
        '''
        Implement viscosity from R Scott et al, WES, 2023
        '''
        wf = self.wf
        idx = i+1

        for turbine in wf.turbines:

            # Plane of the turbine
            n = turbine.n #np.abs(self.x - turbine.location[0]).argmin()

            # Add the wake here
            if (n == i) and (turbine.state is True):

                cond = (np.abs(wf.Y[i,:,:] - turbine.location[1]) < (turbine.D * wf.cf)).nonzero()
                x = (wf.X[idx:, cond[0], cond[1]]-turbine.location[0]) #/ turbine.D  # dimensionless
                sigma = 5.5  # dimensionless
                A = turbine.D / 2 * turbine.Uh * np.sqrt(1-turbine.Ct) / 2
                nu = A * (0.01 + x/sigma**2 * np.exp(-x**2/(2*sigma**2)))
                if self.debug:
                    print(f'plane {i}')
                    print(f'Turbine x {turbine.location[0]}')
                    print(f'A={A}')
                    print(f'D={turbine.D}')
                    print(f'Uh={turbine.Uh}')
                    print(nu.shape)
                    print(x.shape)
                    print(nu[:,-100])
                wf.nu[idx:, cond[0], cond[1]] = nu #np.maximum(nu, self.nu[idx:, cond[0], cond[1]])


class KlTurbulenceModel(TurbulenceModel):
    
    def initialize(self, C_nu=0.04):
        wf = self.wf
        self.C_nu=C_nu
        
        wf.kw = np.zeros_like(wf.uw)
        wf.x_kl = np.zeros_like(wf.uw)  # the distance from the wake plane
        wf.k_tot = np.zeros_like(wf.uw)  # the total tke
        wf.lmix = np.full_like(wf.uw, wf.turbines[0].D)  # the mixing length array

        for turbine in wf.turbines:
            i = turbine.n

            # Create boolean mask over Y at slice i
            yz_mask = np.abs(wf.Y[i, :, :] - turbine.location[1]) < (turbine.D * wf.cf)
            # Offset along X axis starting from i
            offset = wf.X[i:, :, :] - turbine.location[0]
            # Create full 3D mask for the i: range where offset > 5D and within yz region
            full_mask = np.zeros_like(offset, dtype=bool)
            full_mask[:, :, :] = yz_mask  # broadcast along x
            x0 = 2.5 * turbine.D
            valid = (offset > x0) & full_mask
            # Assign only where valid
            wf.x_kl[i:, :, :][valid] = offset[valid] - x0

        # The minimum mixing length (rotor diameter)
        l_min = wf.turbines[0].D

        # Field of the base k (notice the use of TI)
        wf.kb = (wf.TI * wf.U) ** 2 * 3 / 2
        wf.nu = C_nu * np.sqrt(wf.kb) * l_min


    def update(self, i):
        '''
        Update the k-l model at plane i
        '''

        wf = self.wf

        # Calculate the mixing length
        wf.lmix[i-1,:,:] = self.compute_lmix(wf.uw[i-1,:,:], wf.z, l_min=wf.h)

        # This is the original implementation
        wf.kw[i,:,:] = self.rk4_step_kwake(wf.kw[i-1,:,:], wf.uw[i-1,:,:], wf.U[i,:,:] + wf.uw[i-1,:,:], 
            wf.V[i, :, :] + wf.vw[i-1,:,:], wf.W[i,:,:] + wf.ww[i-1,:,:], wf.nu[i-1,:,:], wf.lmix[i-1,:,:], 
            wf.dy, wf.dz, wf.dx, self.compute_dkdx)

        cond = wf.x_kl[i, :, :] > 0
        yi, zi = np.where(cond)

        wf.k_tot[i, :, :] = np.clip(wf.kb[i, :, :] + wf.kw[i, :, :], 0, None)
        wf.nu[i, yi, zi] = self.C_nu * np.sqrt(wf.k_tot[i, yi, zi]) * wf.lmix[i, yi, zi]



    def compute_lmix(self, du, z, l_min=0, l_max=None, du_min=0.5):
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

    def compute_dkdx(self, dk, du, U, V, W, nu_T, lmix, dy, dz, C_k1=1, C_k2=1):
        """
        Computes the dk/dx term for the turbulence model.
        """                

        if np.any(lmix <= 0):
            raise ValueError("lmix is non-positive")

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
        self, dk, du, U, V, W, nu_T, lmix, dy, dz, dx, compute_dkdx_func,
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

    def postprocess(self):

        wf = self.wf
        if self.debug:
            field_names = ["kw", "nu", "kb", "k_tot", "x_kl", "lmix"]
            z_index = int(wf.h / wf.dz)
            z_index = 1
            n = len(field_names)

            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))

            # Don't use constrained_layout — we'll manage spacing manually
            fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axs = np.ravel(axs)

            for i, field_name in enumerate(field_names):
                field = getattr(wf, field_name)
                ax = axs[i]

                # Plot field
                pcm = ax.pcolormesh(wf.x, wf.y, field[:, :, z_index].T, shading='gouraud')
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
            time_label = wf.t if hasattr(wf, "t") else 'unkown'
            #time_label = #self.time_video[-1] if self.time_video else "unknown"
            plt.savefig(f'{wf.saveDir}/test_nut_{time_label}.png', dpi=150)
            plt.close(fig)

