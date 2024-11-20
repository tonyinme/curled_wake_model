#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plotConvectingCurledWake.py
#
#  Copyright 2017 Martinez <lmartin1@LMARTIN1-31527S>
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
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RectBivariateSpline

# Set plottling parameters
# These parameters can be changed to make the plot look better
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['xtick.major.pad'] = 5
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.major.pad'] = 5
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 24 # 34 for paper
matplotlib.rcParams['axes.titlesize'] = 24 # 34 for paper
matplotlib.rcParams['legend.fontsize'] = 14

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'figure.autolayout': True})

# For font types (Journal does not accept type 3 fonts)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def main(args):
    '''
    The curled wake
    '''

    # Initialize the linear wake model
    wake = turbine_model_class(
                    # Turbine diameter [m]
                    D=77,
                    # Tower height [m]
                    th=80,
                    # Thrust coefficient at Uh (without yaw)
                    Ct=0.75,
                    # Yaw angle [deg]
                    alpha=-25,
                    # Hub height velocity [m/s]
                    Uh=10.,
                    # Tip speed ratio (specifies the direction with + or -)
                    tsr=-8,
                    # Flag to include the ground effects
                    ground=True,
                    # Flag to decide if active or not
                    # ~ active=False,
                    )



class turbine_model_class():
    '''
    This is the class for an individual wake.
    The wake is solved using a linearized version of the Euler
    equations with approximations.
    Different features can be added to the wake (curl, veer, etc).
    The linearization is done as:
        u = U + u'
        v = V + v'
        w = W + w',
    where the base solution (U, V, W) as a function of z and y
    is prescribed and the perturbation velocities (u', v', w') as a
    function of x, y and z are solved.

    The coordinate system is referenced by the origin (0, 0, 0) being at the
    center of the rotor
    '''

    def __init__(self,
                    D=1,             # Turbine diameter [m]
                    th=1,            # Tower height [D]
                    Ct=None,          # Thrust coefficient [non-dim]
                    Cp=None,          # Power coefficient [non-dim]
                    alpha=0,        # Yaw angle [degrees]
                    Uh=1.,           # Inflow velocity at hub height [m/s]
                    tsr=8.,          # The tip speed ratio (non-dimensional)
                    rho=1.,          # Air density [kg/m^3]
                    Lx=5.,          # Streamwise lenght [D]
                    Ly=2.,           # Spanwise length [D]
                    Lz=2.,           # Wall-normal length [D]
                    Nx=250,          # Number of grid points in x
                    Ny=150,          # Number of grid points in y
                    Nz=150,          # Number of grid points in z
                    ground=True,     # Include ground effect
                    n_cp=2,         # The power of cosine for the yaw Cp
                    saveDir='./',    # Directory to save data
                    location=(1000,1000,90),     # turbine rotor hub location
                    # The power coefficient [-]
                    cp_ls=  [       0.0,         0.0, 0.1780851,  0.28907459,  0.34902166,
                               0.3847278, 0.40605878, 0.4202279,  0.42882274, 0.43387274,
                              0.43622267, 0.43684468, 0.43657497, 0.43651053, 0.4365612,
                              0.43651728, 0.43590309, 0.43467276, 0.43322955, 0.43003137,
                              0.37655587, 0.33328466, 0.29700574, 0.26420779, 0.23839379,
                              0.21459275, 0.19382354, 0.1756635,  0.15970926, 0.14561785,
                              0.13287856, 0.12130194, 0.11219941, 0.10311631, 0.09545392,
                              0.08813781, 0.08186763, 0.07585005, 0.07071926, 0.06557558,
                              0.06148104, 0.05755207, 0.05413366, 0.05097969, 0.04806545,
                              0.04536883, 0.04287006, 0.04055141],
                    # The wind speed [m/s]
                    u_ls_cp= [   2.0,  2.5,   3.0,  3.5,  4.0,  4.5,  5.0,  5.5,
                              6.0,  6.5,   7.0,  7.5,  8.0,  8.5,  9.0,  9.5,
                              10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
                              14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5,
                              18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5,
                              22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5],
                    # The thrust coefficient [-]
                    ct_ls= [1.19187945, 1.17284634, 1.09860817, 1.02889592, 0.97373036,
                            0.92826162, 0.89210543, 0.86100905, 0.835423, 0.81237673,
                            0.79225789, 0.77584769, 0.7629228, 0.76156073, 0.76261984,
                            0.76169723, 0.75232027, 0.74026851, 0.72987175, 0.70701647,
                            0.54054532, 0.45509459, 0.39343381, 0.34250785, 0.30487242,
                            0.27164979, 0.24361964, 0.21973831, 0.19918151, 0.18131868,
                            0.16537679, 0.15103727, 0.13998636, 0.1289037, 0.11970413,
                            0.11087113, 0.10339901, 0.09617888, 0.09009926, 0.08395078,
                            0.0791188, 0.07448356, 0.07050731, 0.06684119, 0.06345518,
                            0.06032267,  0.05741999, 0.05472609],
                    # The wind speed [m/s]
                    u_ls_ct= [   2.0,  2.5,   3.0,  3.5,  4.0,  4.5,  5.0,  5.5,
                              6.0,  6.5,   7.0,  7.5,  8.0,  8.5,  9.0,  9.5,
                              10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
                              14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5,
                              18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5,
                              22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5],
                    ):
        '''
        Initialize the curled wake object variables
        '''
        # Turbine diameter
        self.D = D
        # Tower height
        self.th = th
        # Thrust coefficient
        self.Ct = Ct
        # Power coefficient
        self.Cp = Cp
        # Yaw angle in radians
        self.alpha = np.deg2rad(alpha)
        # Thrust coefficient power of the cosine for cp
        self.n_cp = n_cp
        # Hub height velocity
        self.Uh = Uh
        # Tip speed ratio
        self.tsr = tsr
        # The air density
        self.rho = rho
        # Effects of ground
        self.ground = ground
        # rotor hub location
        self.location = location

        # The list of velocity vs thrust coefficient
        self.u_ls_ct = u_ls_ct
        self.ct_ls = ct_ls
        # The list of velocity vs power coefficient
        self.u_ls_cp = u_ls_cp
        self.cp_ls = cp_ls


        '''
        Compute needed quantities from inputs
        '''
        # ~ self._compute_induction()

    def _compute_induction(self):
        '''
        Induction factor from actuator disk theory
        The solution to the Ct equation has 2 possible solutions
        Pick the lowest induction value (works well for a<0.5)
        '''

        self.a = (1 - np.sqrt(1 - self.Ct * np.cos(self.alpha)**2)) / 2
        # ~ self.a = (1 - np.sqrt(1 - self.Ct )) / 2

        # ~ print('Induction a is:', self.a)

        # Compute the circulation due to wake rotation
        self.GammaWakeRotation = (2 * np.pi * self.D * (self.a - self.a**2) *
                                    self.Uh / self.tsr)

        # ~ print('Circulation due to wake rotation is:',
                    # ~ self.GammaWakeRotation, '[m^2/s]')

    # A function to compute Cp
    def ct_function(self):
        '''
        Function used to interpolate the cp from velocity
        ct_ls list of thrust coefficients [-]
        u_ls list of ct values [m/s]
        '''

        # The thrust coefficient include no yaw.
        # When induction and curl are computed, the cosine^2 term for yaw is included
        # Linear interpolation of the cp
        if self.Ct==None: self.Ct = np.interp(self.Uh, self.u_ls_ct, self.ct_ls)  #* np.cos(self.alpha)**2

        return self.Ct


    # A function to compute Cp
    def cp_function(self):
        '''
        Function used to interpolate the cp from velocity
        ct_ls list of thrust coefficients [-]
        u_ls list of ct values [m/s]
        '''

        # Linear interpolation of the cp
        if self.Cp==None: self.Cp = np.interp(self.Uh, self.u_ls_cp, self.cp_ls) * np.abs(np.cos(self.alpha)**self.n_cp)
        # ~ if self.Cp==None: self.Cp = np.interp(self.Uh, self.u_ls_cp, self.cp_ls) * np.abs(np.cos(self.alpha)**2)

        return self.Cp

    def power(self):
        '''
        The turbine power
        '''

        # Compute the power from the power coefficient
        p = 1/2 * self.Cp * self.rho * np.pi * self.D**2 / 4 * self.Uh**3

        return p

    def initial_condition(self, Y, Z, U, sigma=2):
        '''
        Set the initial condition for the wake profile
        This is the first profile
        Y, Z - the mesh in the y-z plane
        U - the velocity at the plane
        '''

        # The size of the Gaussian function
        sig = self.D/2

        # Rotor Area
        A = np.pi * self.D**2 / 4

        # Factor to increase wake deficit (assuming the initial condition
        #   is after the rotor plane)
        # This is based on the induced velocity in the fully developed wake
        # to be U(1-2a) from actuator disk theory
        f = 2

        # Project the rotor onto the x plane
        yp = Y * np.cos(self.alpha)
        xp = Y * np.sin(self.alpha)

        # The z axis in reference rame where 0 is the center of the rotor
        Zt = Z #- self.th

        # The radial distance for these points
        r2 = np.sqrt(Y**2 + Zt**2 + xp**2)

        # The values inside the rotor
        condition = np.where(r2 <= self.D/2)
        
        # The average velocity at the rotor plane used to compute the induced
        #   velocity in the wake
        self.Uh = np.mean(U[condition])
        
        # Compute the mean velocity
        # Compute the ct and cp
        self.ct_function()
        self.cp_function()
        # ~ print('Velocity of turbine is ', self.Uh)
        # ~ print('Cp of turbine is ', self.cp_function())
        # ~ print('Power of turbine is ', self.power())
        # ~ print('Ct of turbine is ', self.ct_function())

        # The induction takes into account the yaw angle
        self._compute_induction()

        # Initial condition yawed
        uw_initial = - (U * f * self.a)
        ratio = np.sqrt((1-self.a) / (1-f * self.a))# The area ration of wind turbine wakes
        uw = gaussian_filter(uw_initial * (r2 <= ratio * self.D/2), sigma=sigma)

        return uw





    def add_boundary_layer_log_law(self, z0=.01):
        '''
        Add a boundary layer to the inflow
        This adds a boundary layer based on a shear exponent.

        alpha_shear - this is the shear exponent
        '''
        # von Karman constant
        kappa = 0.4

        # Friction velocity
        us = self.Uh * kappa / np.log(self.th / z0)

        print('Friction velocity is: ', us, 'm/s')

        # Log law profile
        Ubl = us / kappa * np.log((self.Z + 1e-4) / z0)

        # Only allow velocities higher than half the hub height
        self.U = np.maximum(0.2 * self.U, Ubl)

        print( 'Added Boudary Layer')

    def add_rotation(self, Y, Z, V, W):
        '''
        Add rotation to the wake
        '''

        # The circulation
        Gamma = self.GammaWakeRotation


        # Compute the rotational vortex only inside the rotor area
        v, w = (self.vortex(Y, Z, eps=0.2 * self.D, Gamma=Gamma)
                            * (np.sqrt(Y**2 + Z**2) <= self.D / 2)
                            )

        #~ # Add the groun effects
        #~ if self.ground:
            #~ # Add mirror
            #~ vm, wm = self.vortex(self.Y, self.Zt + 2 * self.th,
                            #~ eps=0.2 * self.D, Gamma=-Gamma)
            #~ v += vm
            #~ w += wm

        # Add rotational vortex to velocity
        #~ self.V += v
        #~ self.W += w
        V += v
        W += w

        print('Added Rotation')


    def add_curl(self, Y, Z, V, W):
        '''
        Add vortices due to the curled wake using an elliptical distribution

        Y - the spanwise location with 0 being the rotor center
        Z - the wall-normal location with 0 being the rotor center
        '''

        # The Ct changes according to sin * cos^2
        # The units for circulation are m^2/s
        # Circulation is per unit span (1/D)
        self.Gamma = -(np.pi * self.D / 4 * 1/2 * self.Ct * self.Uh *
                    np.sin(self.alpha) *  np.cos(self.alpha)**2 
                    )

        # ~ print('Circulation due to curl is:', self.Gamma, '[m^2/s]')

        # The widht of the vortex
        eps = 0.2 * self.D

        # The range of radii from the center of the rotor to the tip (0 to R)
        z_vector = np.linspace(0, self.D/2, 50)

        # The length of each section dz
        dz = z_vector[1] - z_vector[0]

        # Scale the circulation by the circulation at the center
        Gamma0 = 4 / np.pi * self.Gamma

        # Loop through all the vortices from an elliptic wing distribution
        # Skip the last point because it has zero circulation
        for z in z_vector[:-1]:

            # Compute the non-dimensional circulation
            Gamma = (- 4 * Gamma0 * z * dz /
                    (self.D**2 * np.sqrt(1 - (2 * z/self.D)**2)))

            # Locations of the tip vortices
            # Top
            yt1, zt1 = 0,  z
            # Bottom
            yt2, zt2 = 0, -z

            # Tip vortex velocities
            # Top
            vt1, wt1 = self.vortex(Y - yt1, Z - zt1,
                                    Gamma=Gamma, eps=eps,
                                  )

            # Bottom
            vt2, wt2 = self.vortex(Y - yt2, Z - zt2,
                                    Gamma=-Gamma, eps=eps
                                  )

            # Add the velocity components
            V += vt1 + vt2
            W += wt1 + wt2

            '''
            Add the ground effects my mirroring the vortices from the curl
            #~ '''
            if self.ground:
                # Tip vortex velocities
                # Top
                vt1, wt1 = self.vortex(Y - yt1, Z + zt1 + 2 * self.th,
                                        Gamma=-Gamma, eps=eps,
                                      )

                # Bottom
                vt2, wt2 = self.vortex(Y - yt2, Z + zt2 + 2 * self.th ,
                                        Gamma=Gamma, eps=eps
                                      )

                # Add the velocity components
                V += vt1 + vt2
                W += wt1 + wt2

            #~ # Loop through all the vortices from an elliptic wing distribution
            #~ # Skip the last point because it has zero circulation
            #~ for z in z_vector[:-1]:

                #~ # Compute the non-dimensional circulation
                #~ Gamma = - (- 4 * Gamma0 * z * dz /
                        #~ (self.D**2 * np.sqrt(1 - (2 * z/self.D)**2)))

                #~ # Locations of the tip vortices
                #~ # Top
                #~ yt1, zt1 = 0, - 2 * self.th - z
                #~ # Bottom
                #~ yt2, zt2 = 0, - 2 * self.th + z

                #~ # Tip vortex velocities
                #~ # Top
                #~ vt1, wt1 = self.vortex(Y - yt1, Z - zt1,
                                        #~ Gamma=Gamma, eps=eps)

                #~ # Bottom
                #~ vt2, wt2 = self.vortex(Y - yt2, Z - zt2,
                                        #~ Gamma=-Gamma, eps=eps)

                #~ # Add the velocity components
                #~ V += vt1 + vt2
                #~ W += wt1 + wt2

            #~ print('Added ground effect due to curl')

        # ~ if self.ground: print('Added ground effect due to curl')

        return V, W


    def stability(self, correct=True):
        '''
        Determine the numerical stability of the algorithm
        '''

        # The convective stability number
        c = (self.dx / self.dy)**2 * (np.amax(self.V / self.U))**2
        # Viscous stability number
        d = 2 * self.dx / self.dy**2 * np.amax(self.nu / self.U)

        #~ condition=(c <= d and d <= 1)
        dxlim = 2 * np.amin(self.U/self.W**2)
        dylim = np.sqrt(np.amax(2 * self.nu * self.dx / self.U))
        condition=(self.dx <= dxlim and self.dy >= dylim)

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # Establish numerical stability
        print('Based on numerical stablity analysis for the')
        print('forward time centered space method:')
        print('dx=' + str(self.dx) + ' <= ' + str(dxlim))
        print('dy=' + str(self.dy) + ' >= ' + str(dylim))
        #~ print(str(c) + ' <= ' + str(d) + ' <= 1')

        if condition==True:
            print('The criteria have been met.')
            print('You can expect a numerically stable solution')

        else:
            print('Stability ciretria NOT met,' +
                    ' the solver might become unstable')
            print('Current value of dx= ' + str(self.dx) + '[m]')

            # Print the new suggested value to the screen
            print('A value of dx<=' + str(dxlim) + '[m] is suggested')
            print('A value of dy>=' + str(dylim) + '[m] is suggested')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def solve(self):
        '''
        Numerical solution
        '''

        # Compute the stability requirements
        self.stability()

        # Compute time
        start = time.time()
        print("Starting solver")

        # Compute the wakes by looping through every x
        xi = self.x[0]

        # Point to uw - This is the field for the perturbation streamwise
        # velocity Uh
        uw = self.uw
        #~ vw = self.vw
        #~ ww = self.ww

        # Loop over all distances downstream
        for xi1 in self.x[1:]:

            # Compute the change in x
            dx = xi1 - xi

            # Write the advecting velocities
            U = self.U
            V = self.V
            W = self.W

            '''
            First compute the velocity gradients
            '''
            # Compute the derivatives in the plane (y and z)
            #~ duwdy = (np.gradient(self.uw[-1], self.dy, edge_order=2, axis=1))
            #~ duwdz = (np.gradient(self.uw[-1], self.dz, edge_order=2, axis=0))
            duwdy = (np.gradient(self.U + self.uw[-1], self.dy, edge_order=2, axis=1))
            duwdz = (np.gradient(self.U + self.uw[-1], self.dz, edge_order=2, axis=0))

            '''
            Now solve the marching problem for u'
            '''
            # Discrete equation
            # Notice laplacian term - this is a stabilizing term
            uwi = (
                    # Term from previous x location
                    self.uw[-1]
                    +
                    # Coefficient from discretization
                    #~ dx / (U) *
                    dx / (U + self.uw[-1]) *
                        # V and W advection
                        ( - V * duwdy - W * duwdz
                        +
                        # Viscous term
                        self.nu * self.laplacian(self.U + self.uw[-1], self.dy, self.dz)
                        #~ self.nu * self.laplacian(self.uw[-1], self.dy, self.dz)
                        )
                    )

            # Plot the turbulent viscosity
            #~ plt.clf()
            #~ img=plt.pcolormesh(self.Y/self.D, self.Z/self.D, self.laplacian(self.uw[-1], self.dy, self.dz),
                    #~ cmap='viridis', shading='gouraud')
            #~ # Set the colorbar
            #~ self.colorbar(img)
            #~ plt.show()

            # Adjust the boundary conditions to be zero at the edges
            uwi[ :,  0] *= 0
            uwi[ :, -1] *= 0
            uwi[ 0,  :] *= 0
            uwi[-1,  :] *= 0

            #~ print(np.shape(uwi))
            # Add the new time
            self.uw.append(uwi)

            # Store the previous xi
            xi = xi1

        print("Solver finished")

        # Compute the time of the solver by subtracting and and start time
        end = time.time()
        print("Solver running time=", end - start, 's')

        self.generate_velocity_field()

    def solve_u_v_w(self):
        '''
        Numerical solution
        '''

        # Compute time
        start = time.time()
        print("Starting solver")

        # Compute the wakes by looping through every x
        xi = self.x[0]

        # Point to uw - This is the field for the perturbation streamwise
        # velocity Uh
        uw = self.uw
        vw = self.vw
        ww = self.ww

        # Loop over all distances downstream
        for xi1 in self.x[1:]:

            # Compute the change in x
            dx = xi1 - xi

            # Write the advecting velocities
            U = self.U + self.uw[-1]
            V = self.vw[-1]
            W = self.ww[-1]

            '''
            First compute the velocity gradients
            '''
            # Compute the derivatives in the plane (y and z)
            duwdy = (np.gradient(U, axis=1)
                        / np.gradient(self.Y, axis=1))
            duwdz = (np.gradient(U, axis=0)
                        / np.gradient(self.Z, axis=0))

            # Compute the derivatives in the plane (y and z)
            dvwdy = (np.gradient(V, axis=1)
                        / np.gradient(self.Y, axis=1))
            dvwdz = (np.gradient(V, axis=0)
                        / np.gradient(self.Z, axis=0))

            # Compute the derivatives in the plane (y and z)
            dwwdy = (np.gradient(W, axis=1)
                        / np.gradient(self.Y, axis=1))
            dwwdz = (np.gradient(W, axis=0)
                        / np.gradient(self.Z, axis=0))

            '''
            Now solve the marching problem for u', v' and w'
            '''
            # Discrete equation
            # Notice laplacian term - this is a stabilizing term
            uwi = (
                    # Term from previous x location
                    self.uw[-1]
                    -
                    # Coefficient from discretization
                    dx / (U) *
                    #~ dx / (self.U) *
                    # V and W advection
                    (V * duwdy + W * duwdz)
                    +
                    # Viscous term
                    self.nu * self.laplacian(self.uw[-1], self.Y, self.Z)
                    )

            # Notice laplacian term - this is a stabilizing term
            vwi = (
                    # Term from previous x location
                    self.vw[-1]
                    -
                    # Coefficient from discretization
                    dx / (U) *
                    #~ dx / (self.U) *
                    # V and W advection
                    (V * dvwdy + W * dvwdz)
                    +
                    # Viscous term
                    self.nu * self.laplacian(self.vw[-1], self.Y, self.Z)
                    )

            # Notice laplacian term - this is a stabilizing term
            wwi = (
                    # Term from previous x location
                    self.ww[-1]
                    -
                    # Coefficient from discretization
                    dx / (U) *
                    #~ dx / (self.U) *
                    # V and W advection
                    (V * dwwdy + W * dwwdz)
                    +
                    # Viscous term
                    self.nu * self.laplacian(self.ww[-1], self.Y, self.Z)
                    )

            # Add the new time
            self.uw.append(uwi)
            self.vw.append(vwi)
            self.ww.append(wwi)

            # Store the previous xi
            xi = xi1

        print("Solver finished")

        # Compute the time of the solver by subtracting and and start time
        end = time.time()
        print("Solver running time=", end - start, 's')

    def generate_velocity_field(self):
        '''
        This function generates a 3D velocity field from the data
        In the linearization u = U + u'
        '''

        # Generate the axial velocity field based on the linearization
        self.u_x = [self.U + u for u in self.uw]

        return self.u_x

    def extract_data_y(self, x=1., z=None, name='line_horizontal'):
        '''
        Extract the data to plot over line
        '''
        # Assign the hub height as default z location
        if z is None:
            z = self.th

        # Find the x and z index
        idx = (np.abs(self.x - x)).argmin()
        idz = (np.abs(self.z - z)).argmin()

        # Line data
        y = self.y

        # Velocity
        #~ u = self.uw[idx][:, idz] + self.U[:, idz]
        u = self.U[idz, :] + self.uw[idx][idz, :]

        # Save the data as a numpy array
        #~ np.save(self.saveDir + '/line-x-' + str(x) + '-z-' + str(z), [y, u])
        np.save(self.saveDir + '/' + name, [y, u])


    def extract_data_z(self, x=1., y=None, name='line_vertical'):
        '''
        Extract the data to plot over line
        '''
        # Assign the hub height as default z location
        if y is None:
            y = 0

        # Find the x and z index
        idx = (np.abs(self.x - x)).argmin()
        idy = (np.abs(self.y - y)).argmin()

        # Line data
        z = self.z - self.th

        # Velocity
        u = self.U[:, idy] + self.uw[idx][:, idy]

        # Save the data as a numpy array
        #~ np.save(self.saveDir + '/line-x-' + str(x) + '-z-' + str(z), [y, u])
        np.save(self.saveDir + '/' + name, [z, u])

    def tracer(self, x0=(0, 0, 90), name='tracer'):
        '''
        Trace a particle in the flow

        x0 - tuple - the initial position of the particle to trace
        '''
        # List of all points
        x_vec = [x0]

        # Loop through all the x locations
        for nx in range(1, self.Nx):

            # Create interpolating functions
            u_interp = RectBivariateSpline(self.y, self.z, self.u_x[nx].T)
            v_interp = RectBivariateSpline(self.y, self.z, self.V.T)
            w_interp = RectBivariateSpline(self.y, self.z, self.W.T)

            # The x coordinate
            x = x_vec[-1][0]
            # The y coordinate
            y = x_vec[-1][1]
            # The z coordinate
            z = x_vec[-1][2]

            # Velocity components
            u = u_interp(y, z)
            v = v_interp(y, z)
            #~ w = 0
            w = w_interp(y, z)

            # The time step to jump to the next cell
            dt = self.dx / u

            # The coordinates
            dx = u * dt
            dy = v * dt
            dz = w * dt

            # The new point
            p1 = (x + dx, y + dy, z + dz)

            # Add the point to the vector
            x_vec.append(p1)

        # Save as numpy array
        np.save(self.saveDir + '/' + name, x_vec)



    @staticmethod
    def vortex(x, y, Gamma=1., eps=.2):
        '''
        Compute the vortex velocity
        x - x location
        y - y location
        eps -  the core size of the tip vortex
        '''
        u =  (Gamma / (2 * np.pi) * y / (x**2 + y**2) *
                (1 - np.exp(-(x**2 + y**2)/eps**2 )))
        v = -(Gamma / (2 * np.pi) * x / (x**2 + y**2) *
                (1 - np.exp(-(x**2 + y**2)/eps**2 )))

        return u, v


    @staticmethod
    def laplacian(u, dy, dz):
        '''
        Compute the laplacian in 2D
        '''
        d2udy2 = np.gradient(np.gradient(u, dy, axis=1), dy, axis=1)
        #~ dy = np.gradient(y,axis=1)**2
        d2udz2 = np.gradient(np.gradient(u, dz, axis=0), dz, axis=0)
        #~ dz = np.gradient(z,axis=0)**2

        return d2udy2 + d2udz2

#~ class PlotClass():
    #~ '''
    #~ A class used for plotting
    #~ '''
    def plot_streamwise(self):
        '''
        Plot horizontal profiles
        '''

        # The field perturbation
        uw = np.asarray(self.uw)

        plt.clf()

        # The 2-D coordinates in a plane
        X, Y = np.meshgrid(self.x / self.D, self.y / self.D)

        # Index that shows minimum distance to tower height position
        i = np.abs(self.z - self.th).argmin()

        # Slice the data at a plane at hub height
        data = self.Uh + np.asarray(uw[:, i, :]).T
        data /= self.Uh

        # The point at which to sample
        # Same size x and y
        plt.gca().set_aspect('equal', adjustable='box')
        img = plt.pcolormesh(X, Y, data,
                cmap='viridis', shading='gouraud', vmin=0.4, vmax=1.0)

        # Set the figure labels
        plt.xlabel(r'$x/D$')
        plt.ylabel(r'$y/D$')

        # Set the colormap the same size as the y axis
        x1, x2 = plt.xlim()
        y1, y2 = plt.ylim()

        # Set the colorbar
        ax = img.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img, cax=cax)
        #~ cax.yaxis.get_major_locator().set_params(nbins=3)

        # Update the number of ticks
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
        cbar.locator = tick_locator
        cbar.update_ticks()

        # ~ for t in self.turbines: 
            # ~ x = t.

        plt.savefig(self.saveDir + '/horizontalProfile.pdf',
                bbox_inches='tight')

    def plot_streamwise_contour(self):
        '''
        Plot horizontal profiles
        '''

        # The field perturbation
        uw = np.asarray(self.uw)

        plt.clf()

        # The 2-D coordinates in a plane
        X, Y = np.meshgrid(self.x / self.D, self.y / self.D)

        # Index that shows minimum distance to tower height position
        i = np.abs(self.z - self.th).argmin()

        # Slice the data at a plane at hub height
        data = self.Uh + np.asarray(uw[:, i, :]).T
        data /= (self.Uh)

        # The point at which to sample
        # Same size x and y
        plt.gca().set_aspect('equal', adjustable='box')
        img = plt.contourf(X, Y, data,
                50,
                cmap='jet',
                vmin=0.2, vmax=1.0)

        # Set the figure labels
        plt.xlabel(r'$x/D$')
        plt.ylabel(r'$y/D$')

        # Set the colorbar
        #~ ax = img.axes
        #~ fig = ax.figure
        #~ divider = make_axes_locatable(ax)
        #~ cax = divider.append_axes("right", size="5%", pad=0.1)
        #~ cbar = fig.colorbar(img, cax=cax)
        plt.colorbar()
        #~ # Update the number of ticks
        #~ tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
        #~ cbar.locator = tick_locator
        #~ cbar.update_ticks()

        plt.savefig(self.saveDir + '/horizontal_contour.pdf',
                bbox_inches='tight')


    def plot(self, xD=3, invert=False, vmin=0.6, vmax=1.0,
                xlim=None, ylim=None, title=True):
        '''
        Plot the wake profiles

        xD - the distance in terms of diameters downstream
        vmin - the minimum range for the colormap
        vmax - the maximum range for the colormap
        '''

        # Find the nearest grid point to the desired location
        #~ # The wake starts at D=1, so there is a +1 in the equation
        #~ n = (np.abs(self.x - (xD + 1) * self.D)).argmin()
        n = (np.abs(self.x - (xD) * self.D)).argmin()

        # The field perturbation
        uw = self.uw

        plt.clf()

        # The diameter as a string
        d = str("{0:.1f}".format(xD))

        # Plot the cirle
        xc = np.linspace(-1/2, 1/2, 200)
        plt.plot(xc, self.th/self.D + np.sqrt((1/2)**2 - xc**2),
                                        lw=4, c='k')
        plt.plot(xc, self.th/self.D - np.sqrt((1/2)**2 - xc**2),
                                        lw=4, c='k')

        # Same size x and y
        plt.gca().set_aspect('equal', adjustable='box')
        u = (self.U + uw[n]) / self.Uh
        #~ u = (uw[n]) / self.Uh

        # Number of sub-samples to take for plotting
        #~ N = int(self.Nz * self.D / (self.Lz * 3.5))
        #~ N=1
        #~ print(self.Lz, self.D, N)
        #~ img = plt.pcolormesh(
                #~ self.Y[N::N, N::N]/self.D,
                #~ self.Z[N::N, N::N]/self.D,
                #~ u[N::N, N::N],
                #~ cmap='viridis',
                #~ shading='gouraud',
                #~ vmin=0.6, vmax=1.0)
        img = plt.pcolormesh(self.Y/self.D, self.Z/self.D, u,
                cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
                #~ cmap='viridis',  vmin=0.6, vmax=1.0)

        plt.xlabel(r'$y/D$')
        plt.ylabel(r'$z/D$')

        if title:
            #~ plt.title(r'$\gamma=$'+str(int(round(abs(self.alpha) * 180./np.pi)))
                    #~ + '$^o$, $x/D=$'+str(int(round(xD))))
            plt.title('Model ' + str(int(round(xD))) + '-D')

        # Limits of the axes if they are not specified
        if not xlim: xlim = [-self.Ly / 2 / self.D, self.Ly / 2 / self.D]
        if not ylim: ylim = [0, self.Lz / self.D]

        plt.ylim(ylim)
        plt.xlim(xlim)


        # Invert the x axis if specified
        if invert:
            plt.gca().invert_xaxis()

        # Scale for the density based on axes
        scale = (self.Ly / self.D) / np.abs(xlim[1]-xlim[0])
        # Plot streamlines
        plt.streamplot(self.Y/self.D, self.Z/self.D,
                        self.V,
                        self.W,
                        color='k', density=1.5 * scale)
        #~ plt.contour(self.Y[N::N, N::N]/self.D,
                        #~ self.Z[N::N, N::N]/self.D,
                        #~ u[N::N, N::N],
                        #~ colors='w',
                        #~ )

        # Set the colorbar
        self.colorbar(img)

        # Save the figure
        plt.savefig(self.saveDir + '/wake'+ d + '-D.pdf',
                        bbox_inches='tight')


    def plot_base_flow(self, xlim=None, ylim=None):
        '''
        Plot the wake profiles

        N - the total number of profiles
        '''

        plt.clf()

        # Plot the cirle
        xc = np.linspace(-1/2, 1/2, 200)
        plt.plot(xc, self.th/self.D + np.sqrt((1/2)**2 - xc**2), lw=4, c='k')
        plt.plot(xc, self.th/self.D - np.sqrt((1/2)**2 - xc**2), lw=4, c='k')


        # Same size x and y
        plt.gca().set_aspect('equal', adjustable='box')
        #~ img = plt.pcolormesh(self.Y/self.D, self.Z/self.D,
                #~ np.sqrt(self.U**2 + self.V**2 + self.W**2),
                #~ cmap='viridis_r', )#shading='gouraud')#, vmin=0., vmax=0.35)

        plt.xlabel(r'$y$')
        plt.ylabel(r'$z$')
        #~ plt.xlim([-self.D, self.D])
        #~ plt.ylim([-self.th/self.D, self.D/self.D])

        # Limits of the axes if they are not specified
        if not xlim: xlim = [-self.Ly / 2 / self.D, self.Ly / 2 / self.D]
        if not ylim: ylim = [0, self.Lz / self.D]

        plt.ylim(ylim)
        plt.xlim(xlim)

        plt.streamplot(self.Y/self.D, self.Z/self.D, self.V, self.W, color='k',
                        density=1.5)
                       #~ minlength=self.minlength, density=3,
                       #~ arrowsize=0.001, color='k')
        #~ self.colorbar(img)

        plt.savefig(self.saveDir + '/BaseSolution.jpg' ,bbox_inches='tight')

    @staticmethod
    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        return fig.colorbar(mappable, cax=cax)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
