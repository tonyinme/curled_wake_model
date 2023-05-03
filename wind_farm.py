#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  wind_farm.py
#
#  Copyright 2018 Martinez <lmartin1@LMARTIN1-31527S>
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

import linear_wake_model as cw

import os
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    Solve a linearized wind farm flow
    '''

    # The coordinates for all turbines
    layout_x = [1000, 1882, 2764, 1000, 1882, 2764]
    layout_y = [1000, 1000, 1500, 1500, 2000, 2000]

    # The list of yaw angles
    yaw = [25, 15, 0, 25, 15, 0, ]

    # The length of the domain
    x_min = np.amin(np.array(layout_x))
    x_max = np.amax(np.array(layout_x))

    y_min = np.amin(np.array(layout_y))
    y_max = np.amax(np.array(layout_y))

    # The domain sizes [m]
    Lx = x_max - x_min + 2000
    Ly = y_max - y_min + 400
    Lz = 400

    '''
    Loop of yaw angles
    '''
    # List of turbines
    turbines = [
        # Initialize the linear wake model
        cw.linear_wake_model_class(
                        # Turbine diameter [m]
                        D=126,
                        # Tower height [m]
                        th=90,
                        # Thrust coefficient at Uh (without yaw)
                        Ct=None, # if None, a lookup table will be used
                        # Yaw angle [deg]
                        alpha=yaw[i],
                        # Hub height velocity [m/s]
                        Uh=8.,
                        # Tip speed ratio (specifies the direction with + or -)
                        tsr=9999, # high number means zero rotation
                        #~ tsr=8,
                        # Flag to include the ground effects
                        ground=True,
                        # Location
                        location=(layout_x[i] - x_min + 200 , layout_y[i] - y_min + 200, 90)
                        #~ location=(np.random.rand()*5000, np.random.rand()*5000, 200)
                        )
         for i in range(len(layout_x))
     ]

    # Initiate the wind farm object
    wf = wind_farm_class(
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                Nx=int(Lx/10),
                Ny=int(Ly/10),
                Nz=int(Lz/10),
                Uh=10,
                h=90,
                turbines=turbines,
                saveDir='results',
                )

    # Add the boundary layer
    wf.add_boundary_layer(alpha_shear=0.2)
    # Add the turbulence model
    wf.add_turbulence_model()

    # Call the volver
    wf.solve()

    # Print the power from all turbines
    power = [t.power() for t in turbines]
    print(power)

    # Plot the plane at hub height
    wf.plot_streamwise()

    # Plot different downstream locations
    N = 20
    for i in range(1, 20):
        x = i * Lx / N

        wf.plot_x_plane(x=x, file_name='x'+str(i) + '.png', vmin=.4, vmax=1.1)

    del(wf)


class wind_farm_class:
    '''
    This class runs a wind farm simulation using the curled wake model
    '''

    def __init__(self,
                rho=1.,            # Air density [kg/m^3]
                Lx=5000.,          # Streamwise lenght [m]
                Ly=2000.,          # Spanwise length [m]
                Lz=200.,           # Wall-normal length [m]
                Nx=500,            # Number of grid points in x
                Ny=400,            # Number of grid points in y
                Nz=50,             # Number of grid points in z
                Uh=1.,             # Inflow velocity at hub height [m/s]
                h=90,              # The height of the velocity Uh [m]
                turbines = [],     # A list of turbine objects
                saveDir='./',      # Directory to save data
                meandering=False,  # Wake meandering
                ):
        '''
        Initialize the function
        '''

        # Make all the inputs an element of the object
        self.rho = rho
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Uh = Uh
        self.h = h
        self.turbines = turbines
        self.saveDir = saveDir

        # Reynolds number used for stability
        # Notice that the purpose of this number is to stabilize the solution
        # and does not provide any physical value.
        Re=10**4

        # Molecular viscosity based on Reynolds number
        # This viscosity can be adjusted to account for turbulence effects
        self.nu = self.Uh * self.h / Re

        # Create plotting directory
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)

        '''
        Compute needed quantities from inputs
        '''
        # The coordinates system is defined by having the rotor disk
        # at location (0, 0, tower height)
        # The ground is at z=0
        self.x, self.dx = np.linspace(0, self.Lx, self.Nx, retstep=True)
        self.y, self.dy = np.linspace(0, self.Ly, self.Ny, retstep=True)
        self.z, self.dz = np.linspace(0, self.Lz, self.Nz, retstep=True)

        # The 2-D coordinates in a plane
        self.Y, self.Z, = np.meshgrid(self.y, self.z)

        # Initialize the flow field
        self.U, self.V, self.W = self._create_base_flow()

        # Create the wake deficit array
        self.uw = [self.U * 0]
        self.vw = [self.U * 0]
        self.ww = [self.U * 0]

        # Identify the point in the x location where the wake is active
        self.activate = [np.argmin(np.abs(self.x - t.location[0]))
                            for t in self.turbines]

        # Wake meandering
        self.vm = np.zeros(self.Nx)
        if meandering:
            print('Adding meandering')
            self.vm = self.Uh * .15 * (np.cos(5 * self.x**2/self.Lx**2) + np.sin(7*self.x/self.Lx - 3.2))

    # ~ def _read_base_flow(self):
        # ~ '''
        # ~ This function reads the base flow solution from a numpy file
        # ~ '''
        

    def _create_base_flow(self):
        '''
        Create the base flow solution used to expand

        self.U, self.V, and self.W are the base flow U, V, and W velocities
        '''
        # Define the streamwise component as hub height velocity
        U = self.Uh * np.ones(np.shape(self.Y))

        # Define the spanwise and wall normal components as zero
        # The array is the same size as Y and Z
        V = np.zeros(np.shape(self.Y))
        W = np.zeros(np.shape(self.Y))

        return U, V, W


    def add_boundary_layer(self, alpha_shear=None, z0=None, TI=None):
        '''
        Add a boundary layer to the inflow
        This adds a boundary layer based on a shear exponent.

        alpha_shear - this is the shear exponent
        
        TI - turbulence intensity at given height
        '''
        
        if alpha_shear:
            # Store the shear exponent
            self.alpha_shear = alpha_shear
  
            # Compute the shear profile based on shear exponent
            Ubl = self.Uh * (self.Z / self.h)**self.alpha_shear

            # Only allow velocities higher than half the hub height
            self.U = np.maximum(0.2 * self.U, Ubl)

            print( 'Added Boudary Layer from Power law')

        if z0:

            # Von-karman constant
            kappa = 0.4
        
            # Compute the friction velocity
            us = kappa * self.Uh / np.log(self.h/z0)
                        
            # Boundary layer log law
            Ubl = us / kappa * np.log(self.Z / z0)
            
            # Only allow velocities higher than 20 percent 
            #   the hub height velocity
            self.U = np.maximum(0.2 * self.U, Ubl)

            print( 'Added Boudary Layer from log law')
            
            



        # Compute the log law for a neutral ABL
        if TI:
        
            # Towend-Perry constant
            A1 = 1.25
            
            # This constant is taken as the average from the results presented by 
            #     Stevens et al (range is between 1.5 and 2.5)
            B1 = 2.0
        
            # Von-karman constant
            kappa = 0.4
        
            # Boundary layer height [m]
            delta = 1e3

            # Compute the friction velocity
            us = TI * self.Uh / np.sqrt(B1 - A1 * np.log(self.h/delta))
            
            # Calculate the value of the roughness height in the log law
            z0 = self.h / np.exp(self.Uh/us * kappa)
            
            # Boundary layer log law
            Ubl = 1/kappa * us * np.log(self.Z / z0)
            
            # Only allow velocities higher than 20 percent 
            #   the hub height velocity
            self.U = np.maximum(0.2 * self.U, Ubl)
            
            print('Boundary layer computed based on turbulence intensity')
            print('z0 =', z0)

    def add_veer(self, vtop=1, vbottom=-1):
        '''
        Add veer velocity
        This is done as a linear function
        vtop The spanwise velocity at the top of the domain
        vbottom The spanwise velocity at the bottom of the domain
        
        Implement veer as an angle **************************
        '''

        # The slope based on the distance from the bottom of the tower
        m = (vtop - vbottom ) / (self.z[-1] - self.z[0])

        # The intercept
        b = vtop - m * self.z[-1] 
        
        # Add a spanwise linear velocity profile to mimic wind veer 
        self.V += m * self.Z + b
        
        # Print on the screen added veer
        print('Added Veer')            

    def add_turbulence_model(self, f=1):
        '''
        Modify the viscosity to have turbulence
        This uses a RANS model
        The model is used for an ABL with given turbulence intensity
        f - factor to scale length scale
        '''
        # Velocity gradient in the z direction at hub height
        # This gradient is for the base flow in the atmosphere
        dudz = np.gradient(self.U, self.dz, axis=0)

        # von-Karman constant
        kappa = 0.41

        # Mixing length based on:
        # ALFRED K. BLACKADAR 1962
        # JIELUN SUN 2011
        # ~ lmda = f * 15. # The maximum length [m]
        lmda = f * 27. # The maximum length [m]
        lm = kappa * self.Z / (1 + kappa * self.Z / lmda)

        # Turbulent viscosity
        nu_t = lm**2 * np.abs(dudz)

        # Pick the maximum between the turbulent model and the one required for
        #   numerical stability
        self.nu = np.maximum(nu_t, self.nu)

        print('Added mixing length turbulent viscosity')

        # Print maximum and minimum nu
        print('Minimum turbulent viscosity =', np.amin(self.nu))


    def solve(self, f=4, cf=2):
        '''
        Numerical solution
        f - scaling factor for viscous term
        cf - condition factor for the range of effect of the vortices
        '''

        # Compute the stability requirements
        #~ self.stability()

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
        for i, xi1 in enumerate(self.x[1:]):

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
            duwdy = (np.gradient(self.uw[-1], self.dy, edge_order=2, axis=1))
            duwdz = (np.gradient(self.uw[-1], self.dz, edge_order=2, axis=0))
            #~ duwdy = (np.gradient(self.U + self.uw[-1], self.dy, edge_order=2, axis=1))
            #~ duwdz = (np.gradient(self.U + self.uw[-1], self.dz, edge_order=2, axis=0))

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
                    #dx / (U) *
                    dx / (U + self.uw[-1]) *
                        # V and W advection
                        ( - (V + self.vm[i]) * duwdy - W * duwdz
                        +
                        # Viscous term
                        # ~ self.nu * self.laplacian(self.uw[-1], self.dy, self.dz)
                        f * self.nu * self.laplacian(self.uw[-1], self.dy, self.dz)
                        )
                    )

            # Add the new added wake
            for j, n in enumerate(self.activate):
                if n == i:
                    print('Activating turbine', str(j))
                    # Point to the turbine object
                    t = self.turbines[j]

                    # The wake deficit
                    uwi += t.initial_condition(self.Y - t.location[1], self.Z - t.location[2], U + uwi)

                    # Condition to determine the incluence of the vortices
                    # within a certain distance of the rotor
                    # ~ cond = np.argwhere(np.abs(self.Y - t.location[1]) < t.D * 2)
                    # ~ cond = np.where(np.abs(self.Y - t.location[1]) < t.D * 2)
                    cond = np.asarray((np.abs(self.Y - t.location[1]) < (t.D * cf))).nonzero()

                    # Add the effct of curl
                    if (t.alpha != 0): 
                        
                        # Add the velocities from the curl
                        V[cond], W[cond] = t.add_curl(
                            self.Y[cond] - t.location[1], 
                            self.Z[cond] - t.location[2], 
                            V[cond], W[cond])
                        # ~ t.add_curl(self.Y - t.location[1], self.Z - t.location[2], V, W)

                    # Add the wake rotation
                    t.add_rotation(
                        self.Y - t.location[1], 
                        self.Z - t.location[2], 
                        V, W)
                    # ~ t.add_rotation(self.Y - t.location[1], self.Z - t.location[2], V, W)

            # Adjust the boundary conditions to be zero at the edges
            uwi[ :,  0] *= 0
            uwi[ :, -1] *= 0
            uwi[ 0,  :] *= 0
            uwi[-1,  :] *= 0
            # ~ self.V[ :,  0] *= 0
            # ~ self.V[ :, -1] *= 0
            # ~ self.V[ 0,  :] *= 0
            # ~ self.V[-1,  :] *= 0

            #~ print(np.shape(uwi))
            # Add the new time
            self.uw.append(uwi)
            self.vw.append(V)
            self.ww.append(W)

            # Store the previous xi
            xi = xi1

        print("Solver finished")

        # Compute the time of the solver by subtracting and and start time
        end = time.time()
        print("Solver running time=", end - start, 's')

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

    def plot_background(self,  name='backgroundProfile.png'):
        '''
        Plot the inflow profile
        '''
        plt.clf()
        
        plt.plot(self.U[:,0], self.z, '-o', label='U')
        plt.plot(self.V[:,0], self.z, '-d', label='V')

        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Distance [m]')
        plt.legend()

        plt.savefig(self.saveDir + '/' + name,
                bbox_inches='tight', dpi=250)

    def plot_streamwise(self, name='horizontalProfile.png', background=False,
         vmin=None, vmax=None, xlim=None, ylim=None, title=None, yshift=None):
        '''
        Plot horizontal profiles
        '''

        # The field perturbation
        uw = np.asarray(self.uw)

        plt.clf()

        # The 2-D coordinates in a plane
        X, Y = np.meshgrid(self.x , self.y )

        # Index that shows minimum distance to tower height position
        i = np.abs(self.z - self.h).argmin()

        # Slice the data at a plane at hub height
        if background:
            data = np.asarray(uw[:, i, :]).T
            print(np.shape(data))
            print(np.shape(self.U))

            for j in range(self.Nx): data[:, j] += self.U[i, :]
        else:
            data = self.Uh + np.asarray(uw[:, i, :]).T
        data /= self.Uh

        # Shift the domain for the plotting
        if yshift: Y += yshift

        for t in self.turbines:
            x1 = t.location[0]
            x2 = t.location[0]
            y1 = t.location[1] - t.D / 2 
            y2 = t.location[1] + t.D / 2
            
            # Shift the domain for the plotting
            if yshift: 
                y1 += yshift
                y2 += yshift
            
            # The mid point which to rotate about
            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2
            xr1 = (x1-xm) * np.cos(t.alpha) - (y1-ym) * np.sin(t.alpha)
            yr1 = (y1-ym) * np.cos(t.alpha) + (x1-xm) * np.sin(t.alpha)
            xr2 = (x2-xm) * np.cos(t.alpha) - (y2-ym) * np.sin(t.alpha)
            yr2 = (y2-ym) * np.cos(t.alpha) + (x2-xm) * np.sin(t.alpha)
            xr1 += xm
            xr2 += xm
            yr1 += ym
            yr2 += ym
            # ~ plt.plot([x1, x2], [y1, y2], '-k', lw=2)
            plt.plot([xr1, xr2], [yr1, yr2], '-k', lw=2)

        # The point at which to sample
        # Same size x and y
        plt.gca().set_aspect('equal', adjustable='box')
        img = plt.pcolormesh(X, Y, data,
                cmap='viridis', #shading='gouraud', 
                vmin=vmin, vmax=vmax)

        # Set the figure labels
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$y$ [m]')

        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)

        if title: plt.title(title)

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

        plt.savefig(self.saveDir + '/' + name,
                bbox_inches='tight', dpi=250)
                
    def plot_x_plane(self, x=300, file_name='downstream.pdf',
            vmin=None, vmax=None, streamplot=False,
            cmap='viridis', savedata=False, contourf=False):
        '''
        Plot x profiles

        x - the location downstream
        '''

        # The field perturbation
        uw = np.asarray(self.uw)

        plt.clf()

        # The 2-D coordinates in a plane
        X, Y = np.meshgrid(self.y , self.z )

        # Index that shows minimum distance to tower height position
        i = np.abs(self.x - x).argmin()

        # Slice the data at a plane at hub height
        data = self.U + np.asarray(uw[i, :, :])#.T
        # ~ data = -np.asarray(uw[i, :, :])#.T
        data /= self.Uh

        # The point at which to sample
        # Same size x and y
        plt.gca().set_aspect('equal', adjustable='box')

        if contourf:
            img = plt.contourf(X, Y, data, 56,
                cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            
        else:
            img = plt.pcolormesh(X, Y, data,
                cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

        plt.xlim([np.amin(self.y), np.amax(self.y)])
        plt.ylim([np.amin(self.z), np.amax(self.z)])

        # Set the figure labels
        plt.xlabel(r'$y$ [m]')
        plt.ylabel(r'$z$ [m]')

        # Set the colormap the same size as the y axis
        x1, x2 = plt.xlim()
        y1, y2 = plt.ylim()

        # Scale for the density based on axes
        scale = 1#(self.Ly / self.h) / np.abs(x1 - x2)

        if streamplot:
        # Plot streamlines
            plt.streamplot(X, Y,
                            self.V,
                            self.W,
                            color='k', density=1.5 * scale)

        if not contourf:
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

        plt.savefig(self.saveDir + '/' + file_name,
                bbox_inches='tight', dpi=150)

		# Save the data that was plotted
        if savedata:
            np.savez(self.saveDir + '/' + file_name[:-4], x=X, y=Y, u=data*self.Uh, v=self.V, w=self.W)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
