#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  turbine_model.py
#
#  Copyright 2025 Martinez 
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
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RectBivariateSpline

from numpy.polynomial.legendre import leggauss
from numba import njit, prange



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
                    D=1,                           # Turbine diameter [m]
                    th=1,                          # Tower height [D]
                    Ct=None,                       # Thrust coefficient [non-dim]
                    Cp=None,                       # Power coefficient [non-dim]
                    alpha=0,                       # Yaw angle [degrees]
                    tilt=0,                        # Tilt angle [degrees]
                    Uh=1.,                         # Inflow velocity at hub height [m/s]
                    tsr=8.,                        # The tip speed ratio (non-dimensional)
                    rho=1.225,                     # Air density [kg/m^3]
                    n_cp=2,                        # The power of cosine for the yaw Cp
                    location=(1000,1000,90),       # turbine rotor hub location
                    ground=False,                   # Include ground effect
                    name='turbine 1',              # name of the turbine
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
                    # On/off signal as function of time (default to always on) [-] 
                    on_off = lambda t: 1,          # Turbine on or off function (1 is on)
                    alpha_fun=lambda t: 0,             # Yaw angle function [degrees]
                    mean_UV=False,  # Compute power using magnitude of U and V or only U
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
        # Yaw angle in radians (positive means clockwise rotation from above and wake is moved to the left)
        self._alpha = np.deg2rad(alpha)
        # Tilt angle in radians (positive means that bottom half is rotating away from tower and wake is deflected upward)
        self._tilt = np.deg2rad(tilt)
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
        # Turbine name
        self.name = name

        # The list of velocity vs thrust coefficient
        self.u_ls_ct = u_ls_ct
        self.ct_ls = ct_ls
        # The list of velocity vs power coefficient
        self.u_ls_cp = u_ls_cp
        self.cp_ls = cp_ls

        # A function of time to determine if the turbine is on or off
        self.on_off = on_off
        # If the turbine is on (True) or off (False)
        self.state=True  
        # The function will return in radians
        self.alpha_fun = lambda t: np.deg2rad(alpha_fun(t))

        # The velocity used on calculate blade quantities (True means using mag(U,V))
        self.mean_UV=mean_UV

        # The time history used for transient simulations
        # We can append values to these lists
        self.time = []
        self.Cp_time = []
        self.Ct_time = []
        self.pwr_time = []
        self.Uh_time = []
        self.alpha_time = []
        self.location_time = []
        #self.Um_time = []
        #self.Vm_time = []
        
        '''
        Compute needed quantities from inputs
        '''
        # ~ self._compute_induction()

    @property
    def alpha(self):
        """Property getter: Returns the stored alpha value in radians."""
        return self._alpha

    @alpha.setter
    def alpha(self, value_deg):
        """Property setter: Sets the alpha value from degrees to radians."""
        self._alpha = np.deg2rad(value_deg)


    @property
    def tilt(self):
        """Property getter: Returns the stored tilt value in radians."""
        return self._tilt

    @tilt.setter
    def tilt(self, value_deg):
        """Property setter: Sets the tilt value from degrees to radians."""
        self._tilt = np.deg2rad(value_deg)

    def update_alpha(self, t):
        """Property setter: 
        - If t, checks if it should interpolate.
        - If lookup table exists, interpolates based on time.
        - Otherwise, sets alpha directly.
        returns value in degrees
        """
        if isinstance(t, (int, float)):
            self._alpha = self.alpha_fun(t) # returns value in radians
        else:
            print('alpha setter called but without a time value')



    def update_time_vars(self, t=0):
        '''
        Update the time vars by appending the latest time
        '''
        self.time.append(t)
        self.Cp_time.append(self.Cp)
        self.Ct_time.append(self.Ct)
        self.pwr_time.append(self.pwr)
        self.Uh_time.append(self.Uh)
        self.alpha_time.append(self.alpha)
        self.location_time.append(self.location)


    def _compute_induction(self, output=False):
        '''
        Induction factor from actuator disk theory
        The solution to the Ct equation has 2 possible solutions
        Pick the lowest induction value (works well for a<0.5)
        '''

        self.a = (1. - np.sqrt(1. - self.Ct * np.cos(self.alpha)**2)) / 2
        # ~ self.a = (1 - np.sqrt(1 - self.Ct )) / 2
        self.a = min(self.a, .35)  # Set a limit to guarantee numerical stability
        if output: 
            print('Induction a is:', self.a)

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
        
        # A limit based on theory
        # ~ self.Ct = min(24/25, self.Ct)
        # ~ self.Ct = min(24/25*.9, self.Ct)

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

        self.cp_function()
        
        # Compute the power from the power coefficient
        self.pwr = 1/2 * self.Cp * self.rho * np.pi * self.D**2 / 4 * self.Uh**3

        return self.pwr

    def initial_condition(self, Y, Z, U, V=None, sigma=1):
        '''
        Set the initial condition for the wake profile
        This is the first profile
        Y, Z - the mesh in the y-z plane
        U - the velocity at the plane
        V - optional spanwise velocity at plane
        '''

        # The size of the Gaussian function (units of pixels)
        #sig = self.D/2

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
        #r2 = np.sqrt(Y**2 + Zt**2 + xp**2)
        r2 = np.sqrt(Y**2 + Zt**2)

        # The values inside the rotor
        condition = np.where(r2 <= self.D/2)
        
        # The average velocity at the rotor plane used to compute the induced
        #   velocity in the wake
        self.Uh = np.mean(U[condition])

        # Calculate the effective velocity using both velocity components 
        # This should be adjusted 
        if self.mean_UV: self.Uh = np.mean( np.sqrt(U[condition]**2 + V[condition]**2))

        if V is not None: 
            self.Vh = np.mean(V[condition])
            self.thetah = np.mean(np.rad2deg(np.arctan2(V[condition], U[condition])))
            # Consider wind from any direction (approximation will have to be tested)
            #self.Uh = np.sqrt(self.Uh**2 + self.Vh**2)
        
        # Compute the mean velocity
        # Compute the ct and cp
        self.ct_function()
        self.cp_function()
        self.power()
        #print('Turbine ', self.name)
        #print('Velocity of turbine is ', self.Uh)
        #print('Cp of turbine is ', self.cp_function())
        #print('Power of turbine is ', self.power())
        #print('Ct of turbine is ', self.ct_function())

        # The induction takes into account the yaw angle
        self._compute_induction()

        # Initial condition yawed
        uw_initial = - (U * f * self.a)  # This was juse "U" before (should use Uh?)
        #uw_initial = - (self.Uh * f * self.a)  
        ratio = np.sqrt((1-self.a) / (1-f * self.a))# The area ration of wind turbine wakes
        uw = gaussian_filter(uw_initial * (r2 <= ratio * self.D/2), sigma=sigma)
#        print('uw initial', uw_initial)
        #print('mean uw', uw_initial[np.abs(uw_initial)>0].sum())
        #print('mean uw', uw[np.abs(uw)>0].sum())

        #uw = gaussian_filter(uw_initial, sigma=sigma) * (r2 <= ratio * self.D/2)
        self.ic = uw # store the initial condition

        return self.ic

    def add_rotation(self, Y, Z, V, W):
        '''
        Add rotation to the wake
        '''

        # The circulation
        Gamma = self.GammaWakeRotation

        rad2 = Y**2 + Z**2
        mask = (rad2 <= self.D**2/4) 
        #mask *= (Z > -self.th)  # ensure wall is not adjusted

        # Compute the rotational vortex only inside the rotor area
        # The kernel width eps needs to be high (half a diameter), or artificially high velocities get formed near the core
        v, w = (self.vortex(Y, Z, eps=0.4 * self.D, Gamma=Gamma)
                            * (np.sqrt(Y**2 + Z**2) <= (1. * self.D / 2))
                            )
        
        #if self.name=='E06':
        #    y1 = np.linspace(-self.D, self.D, 100)
        #    z1 = 0
        #    for eps in [.05, .1, .2, .3, .4, .5]:
        #        v1, w1 = self.vortex(y1, z1, eps=eps * self.D, Gamma=Gamma)
        #        plt.plot(y1, w1, 'o-', label=f'eps={eps}')
        #    plt.legend()
        #    plt.show()

        # The maximum rotational velocity. This velocity is acutally very small, but because of the lamb-oseen vortex model,
        # the velocities can be artificially large and cause numerical instabilities. 
        # This sets the max velocity at 10% of the rotor wind speed, which is still quite high
        #v_max = self.Uh * .1
        #w_max = self.Uh * .1

        #v = np.clip(v, -v_max, v_max)
        #w = np.clip(w, -w_max, w_max)


        #~ # Add the groun effects
        #~ if self.ground:
            #~ # Add mirror
            #~ vm, wm = self.vortex(self.Y, self.Zt + 2 * self.th,
                            #~ eps=0.2 * self.D, Gamma=-Gamma)
            #~ v += vm
            #~ w += wm

        # Add rotational vortex to velocity
        V += v * mask
        W += w * mask

        #print('Added Rotation')


    def add_curl_quad(self, Y, Z, V, W, N=40):
        '''
        Add vortices due to the curled wake using an elliptical distribution

        Y - the spanwise location with 0 being the rotor center
        Z - the wall-normal location with 0 being the rotor center
        '''

        # Compute total circulation
        self.Gamma = -(
            np.pi * self.D / 4 * 0.5 * self.Ct * self.Uh *
            np.sin(self.alpha) * np.cos(self.alpha)**2
        )

        # Elliptic distribution peak value
        Gamma0 = 4 / np.pi * self.Gamma

        # Effective core size of the vortex
        sigma = 0.2 * self.D

        # Quadrature resolution
        N_quad = N #40  # adjust this if needed for accuracy/speed tradeoff

        # Compute contribution from top curl
        v_top, w_top = delta_vw_fixed_quad(
            N_quad,
            y=Y,
            z=Z,
            R=self.D / 2,
            sigma=sigma,
            Gamma0=Gamma0
        )

        # Add contributions
        V += v_top
        W += w_top

        # Ground mirror image (if applicable)
        if self.ground:
            Z_reflect = Z + 2 * self.th  # reflected height
            v_top_g, w_top_g = delta_vw_fixed_quad(
                N_quad,
                y=Y,
                z=Z_reflect,
                R=self.D / 2,
                sigma=sigma,
                Gamma0=-Gamma0
            )

            v_bot_g, w_bot_g = delta_vw_fixed_quad(
                N_quad,
                y=Y,
                z=-Z_reflect,
                R=self.D / 2,
                sigma=sigma,
                Gamma0=Gamma0
            )

            V += v_top_g + v_bot_g
            W += w_top_g + w_bot_g


    def add_curl_legacy(self, Y, Z, V, W, N=50):
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
        z_vector = np.linspace(0, self.D/2, N)

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

    def add_curl_just_yaw(self, Y, Z, V, W, N=20):
        '''
        Add vortices due to the curled wake using an elliptical distribution
        (with z = R sin(θ) transformation and midpoint integration).
        
        Y, Z - grid coordinates where the velocities are computed
        V, W - velocity fields to which the curl velocities are added
        N    - number of integration points in θ-space
        '''
        
        # Compute total circulation
        self.Gamma = -(
            np.pi * self.D / 4 * 0.5 * self.Ct * self.Uh *
            np.sin(self.alpha) * np.cos(self.alpha)**2
        )

        Gamma0 = 4 / np.pi * self.Gamma
        eps = 0.2 * self.D
        R = self.D / 2

        # Midpoint integration in θ-space
        theta_edges = np.linspace(0, 0.5 * np.pi, N + 1)
        dtheta = theta_edges[1] - theta_edges[0]
        theta = theta_edges[:-1] + 0.5 * dtheta

        z_vector = R * np.sin(theta)
        dz_dtheta = R * np.cos(theta)

        for i in range(N):
            z = z_vector[i]
            dz = dz_dtheta[i] * dtheta  # Jacobian correction

            # Elliptical circulation distribution (avoid division by zero at z=R)
            denom = np.sqrt(1 - (2 * z / self.D)**2)
            denom = max(denom, 1e-12)
            Gamma = (-4 * Gamma0 * z * dz / (self.D**2 * denom))

            # Induced velocity from vortex pair
            yt1, zt1 = 0, z
            yt2, zt2 = 0, -z

            vt1, wt1 = self.vortex(Y - yt1, Z - zt1, Gamma=Gamma, eps=eps)
            vt2, wt2 = self.vortex(Y - yt2, Z - zt2, Gamma=-Gamma, eps=eps)

            V += vt1 + vt2
            W += wt1 + wt2

            # Add mirrored vortices if ground is present
            if self.ground:
                vt1g, wt1g = self.vortex(Y - yt1, Z + zt1 + 2 * self.th, Gamma=-Gamma, eps=eps)
                vt2g, wt2g = self.vortex(Y - yt2, Z + zt2 + 2 * self.th, Gamma=Gamma, eps=eps)

                V += vt1g + vt2g
                W += wt1g + wt2g


    def add_curl(self, Y, Z, V, W, N=20):
        '''
        Add vortices due to the curled wake using an elliptical distribution
        with support for both yaw and tilt.
        '''
#        V = np.zeros_like(Y)
#        W = np.zeros_like(Y)

        R = self.D / 2
        eps = 0.2 * self.D

        # Rotor normal vector (initially pointing along x)
        n = np.array([
            np.cos(self.alpha) * np.cos(self.tilt),  # x-component
            np.sin(self.alpha),                      # y-component (yaw)
            -np.cos(self.alpha) * np.sin(self.tilt)  # z-component (tilt)
        ])

        # Inflow vector (along x)
        u = np.array([1.0, 0.0, 0.0])

        # Total rotation angle from inflow to rotor normal
        theta_total = np.arccos(np.clip(np.dot(n, u), -1.0, 1.0))

        # Rotation axis is perpendicular to inflow and rotor normal
        axis = np.cross(u, n)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            axis = np.array([0.0, 0.0, 1.0])  # default to vertical
        else:
            axis /= axis_norm

        # Circulation magnitude (same formula as before)
        Gamma_total = -(
            np.pi * self.D / 4 * 0.5 * self.Ct * self.Uh *
            np.sin(theta_total) * np.cos(theta_total)**2
        )
        Gamma0 = 4 / np.pi * Gamma_total

        # Midpoint integration in θ-space (to preserve elliptical distribution)
        theta_edges = np.linspace(0, 0.5 * np.pi, N + 1)
        dtheta = theta_edges[1] - theta_edges[0]
        theta = theta_edges[:-1] + 0.5 * dtheta

        r = R * np.sin(theta)             # radial coordinate
        dr = R * np.cos(theta) * dtheta   # arc length correction

        for i in range(N):
            s = r[i]
            ds = dr[i]

            denom = np.sqrt(1 - (2 * s / self.D)**2)
            denom = max(denom, 1e-12)
            Gamma = -4 * Gamma0 * s * ds / (self.D**2 * denom)

            # Get positive and negative vortex positions
            offset = s * axis[1:]  # [y, z] components of the 3D axis

            yt1, zt1 =  offset[0],  offset[1]
            yt2, zt2 = -offset[0], -offset[1]

            vt1, wt1 = self.vortex(Y - yt1, Z - zt1, Gamma=Gamma, eps=eps)
            vt2, wt2 = self.vortex(Y - yt2, Z - zt2, Gamma=-Gamma, eps=eps)

            V += vt1 + vt2
            W += wt1 + wt2

            if self.ground:
                z_offset = 2 * self.th

                vt1g, wt1g = self.vortex(Y - yt1, Z + zt1 + z_offset, Gamma=-Gamma, eps=eps)
                vt2g, wt2g = self.vortex(Y - yt2, Z + zt2 + z_offset, Gamma=Gamma, eps=eps)

                V += vt1g + vt2g
                W += wt1g + wt2g

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



    @staticmethod
    def vortex(x, y, Gamma=1., eps=.2):
        '''
        Compute the vortex velocity
        x - x location
        y - y location
        eps -  the core size of the tip vortex
        '''
#        u =  (Gamma / (2 * np.pi) * y / (x**2 + y**2) *
#                (1 - np.exp(-(x**2 + y**2)/eps**2 )))
#        v = -(Gamma / (2 * np.pi) * x / (x**2 + y**2) *
#                (1 - np.exp(-(x**2 + y**2)/eps**2 )))

        r2 = x*x + y*y
        exp_term = 1 - np.exp(-r2 / (eps*eps))

        # Avoid divide-by-zero: use mask to skip the zero-radius points entirely
        safe = r2 > 1e-12
        factor = np.zeros_like(r2)
        factor[safe] = (Gamma / (2 * np.pi)) * exp_term[safe] / r2[safe]

        u = factor * y
        v = -factor * x

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

#@njit(parallel=True)
def delta_vw_fixed_quad_core(y, z, zp, dzp_dtheta, weights, R, sigma, Gamma0):
    delta_v = np.zeros_like(y)
    delta_w = np.zeros_like(z)

    flat_y = y.ravel()
    flat_z = z.ravel()
    flat_dv = delta_v.ravel()
    flat_dw = delta_w.ravel()

    for i in prange(flat_y.size):
        current_y = flat_y[i]
        current_z = flat_z[i]

        denom = 2 * np.pi * (current_y**2 + (current_z - zp)**2)
        denom = np.where(denom == 0, 1e-12, denom)

        exp_term = 1 - np.exp(-(current_y**2 + (current_z - zp)**2) / sigma**2)
        gamma_term = Gamma0 * zp / R**2

        integrand_v = ((current_z - zp) / denom) * exp_term * gamma_term * dzp_dtheta
        integrand_w = (-current_y / denom) * exp_term * gamma_term * dzp_dtheta

        flat_dv[i] = np.sum(weights * integrand_v)
        flat_dw[i] = np.sum(weights * integrand_w)

    return delta_v, delta_w


def delta_vw_fixed_quad(N, y, z, R, sigma, Gamma0):
    xi, wi = np.polynomial.legendre.leggauss(N)
    theta = 0.5 * np.pi * xi
    weights = 0.5 * np.pi * wi
    zp = R * np.sin(theta)
    dzp_dtheta = R * np.cos(theta)
    return delta_vw_fixed_quad_core(y, z, zp, dzp_dtheta, weights, R, sigma, Gamma0)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
