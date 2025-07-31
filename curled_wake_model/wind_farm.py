#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  wind_farm.py
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
wind_farm.py

This module defines the `wind_farm_class`, a numerical simulation tool for modeling 
wind farm wakes using a curled wake approach. It supports different turbulence models, 
boundary layer profiles, wake veer effects, and multiple time-stepping solvers including:

- Steady-state solver
- Quasi-steady time-marching solver
- Time-dependent dynamic solver

It also includes convenience methods for:

- Adding atmospheric boundary layer inflow
- Applying wake veer
- Modifying turbulence using a RANS-like mixing-length model
- Computing and saving turbine-level power and error metrics
- Attaching plotting methods and running from script via CLI

Example:
--------
To simulate a wind farm and visualize results:
    $ python wind_farm.py

Dependencies:
-------------
- numpy
- scipy
- pandas
- matplotlib (for plotting, attached separately)
- user-defined modules: `turbine_model`, `wind_farm_solvers`, `wind_farm_plot`
"""

#import turbine_model as tm
from . import turbine_model as tm

from .wind_farm_solvers.solver_steady import solve
from .wind_farm_solvers.solver_time import solve_in_time
from .wind_farm_solvers.solver_quasisteady import solve_steady_state_in_time

# Auto-import and attach all plot functions from another file
import types
from . import wind_farm_plot as methods_plot

import os
import shutil

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, circmean
from scipy.ndimage import gaussian_filter
import pandas as pd

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
        tm.turbine_model_class(
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
    """
    wind_farm_class

    A computational model for simulating flow through wind farms using a curled wake model.
    The class supports multi-turbine simulation on a structured 3D Cartesian grid, and can 
    evolve the flow using steady-state or time-marching schemes.

    Key Features
    ------------
    - Curled wake representation with turbine-induced momentum deficit
    - Flexible turbine layouts and yaw control
    - Inflow boundary layer options (power-law, log-law, TI-based)
    - Mixing-length turbulence model
    - Time-dependent solvers: steady, dynamic (RK4), quasi-steady
    - Built-in turbine performance tracking and error computation
    - Plotting functions attached dynamically via `wind_farm_plot`

    Parameters
    ----------
    rho : float
        Air density in kg/m³ (default: 1.0)
    Lx, Ly, Lz : float
        Domain dimensions in x, y, z [m]
    Nx, Ny, Nz : int
        Number of grid points in x, y, z
    CFL : float
        Courant-Friedrichs-Lewy number for time integration
    Re : float
        Artificial Reynolds number for viscosity scaling
    Uh : float
        Hub-height inflow velocity [m/s]
    h : float
        Hub height [m]
    turbines : list
        List of turbine_model_class instances
    saveDir : str
        Directory path to save output files and figures
    label : str
        Case label for organizational purposes
    meandering : bool
        Enable simple wake meandering (default: False)

    Notes
    -----
    - Call `add_boundary_layer()` to shape the inflow before solving.
    - Use `solve()`, `solve_in_time()`, or `solve_steady_state_in_time()` to run a simulation.
    - Plotting and power output utilities are included.
    """


    def __init__(self,
                rho=1.,            # Air density [kg/m^3]
                Lx=5000.,          # Streamwise lenght [m]
                Ly=2000.,          # Spanwise length [m]
                Lz=200.,           # Wall-normal length [m]
                Nx=500,            # Number of grid points in x
                Ny=400,            # Number of grid points in y
                Nz=50,             # Number of grid points in z
                CFL=1,             # The CFL number to use in time solver [-]
                Re=10.**4,         # Numerical stability Reynolds number
                Uh=1.,             # Inflow velocity at hub height [m/s]
                h=90,              # The height of the velocity Uh [m]
                turbines = [],     # A list of turbine objects
                saveDir='./',      # Directory to save data
                label='case',      # A label to describe this wind farm
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
        self.CFL=CFL
        self.Uh = Uh
        self.h = h
        self.turbines = turbines
        self.saveDir = saveDir
        self.label = label
        self.meandering = meandering

        # Reynolds number used for stability
        # Notice that the purpose of this number is to stabilize the solution
        # and does not provide any physical value.
        self.Re=Re

        # Create plotting directory
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
            
        # Create the domain size
        self._init_domain()

        # Initialize the flow field variables
        self.U, self.V, self.W = self._create_base_flow()

        # Initial condition 
        self._initial_condition()

        self._wspd=None
        self._wdir=None

        self.time_video = [0]


    def _initial_condition(self):
        """
        Set the initial flow and wake field conditions.

        Initializes:
        - Background viscosity field `nu` using a nominal Reynolds number
        - Flow fields (U, V, W) to match the base inflow
        - Wake deficit fields (uw, vw, ww) to zeros
        - Activation indices for each turbine based on x-location

        This method is called during initialization and may be manually rerun if needed.
        """
        # Molecular viscosity based on Reynolds number
        # This viscosity can be adjusted to account for turbulence effects
        self.nu = self.U * self.h / self.Re
        self.nu_min = self.U * self.h / self.Re

        # Initialize flow fields to zero            
        self.U.fill(self.Uh)
        self.V.fill(0)
        self.W.fill(0)
        
        # Create the wake deficit array
        self.uw = np.zeros_like(self.U, dtype=np.float64)
        self.vw = np.zeros_like(self.U, dtype=np.float64)
        self.ww = np.zeros_like(self.U, dtype=np.float64)

        # Identify the point in the x location where the wake is active
        self.activate = [np.argmin(np.abs(self.x - t.location[0]))
                            for t in self.turbines]
        
    def _init_domain(self):
        """
        Initialize the computational domain and coordinate system.

        This sets up the 3D structured mesh over which the wake model is computed. 
        The grid spans the domain defined by `Lx`, `Ly`, and `Lz` using the specified 
        resolution `Nx`, `Ny`, and `Nz`.

        Coordinate arrays created:
        - self.x, self.y, self.z : 1D arrays of grid points in each direction
        - self.X, self.Y, self.Z : 3D meshgrids used in simulation and turbine calculations
        - self.dx, self.dy, self.dz : Grid spacing in x, y, z directions
        """

        # The coordinates system is defined by having the rotor disk
        # at location (0, 0, tower height)
        # The ground is at z=0
        self.x, self.dx = np.linspace(0, self.Lx, self.Nx, retstep=True)
        self.y, self.dy = np.linspace(0, self.Ly, self.Ny, retstep=True)
        self.z, self.dz = np.linspace(0, self.Lz, self.Nz, retstep=True)
        
        # The 3-D coordinates in the domain
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij') 

    def _create_base_flow(self):
        """
        Create the base flow field (U, V, W) prior to applying turbines or turbulence.

        The base flow assumes uniform streamwise velocity at hub height (`Uh`) and zero
        lateral and vertical components. Returns arrays matching the shape of the grid.

        Returns
        -------
        U : ndarray
            Streamwise velocity field initialized to Uh
        V : ndarray
            Spanwise velocity field (zeros)
        W : ndarray
            Vertical velocity field (zeros)
        """
        # Define the streamwise component as hub height velocity
        U = self.Uh * np.ones(np.shape(self.Y))

        # Define the spanwise and wall normal components as zero
        # The array is the same size as Y and Z
        V = np.zeros(np.shape(self.Y), dtype=np.float64)
        W = np.zeros(np.shape(self.Y), dtype=np.float64)

        return U, V, W

    @property
    def wspd(self):
        if self._wspd is None:
            u_total = self.U + np.nan_to_num(self.uw, nan=0.0)
            v_total = self.V + np.nan_to_num(self.vw, nan=0.0)
            w_total = self.W + np.nan_to_num(self.ww, nan=0.0)
            self._wspd = np.sqrt(u_total**2 + v_total**2 + w_total**2)
        return self._wspd

    @property
    def wdir(self):
        if self._wdir is None:
            u_total = self.U + np.nan_to_num(self.uw, nan=0.0)
            v_total = self.V + np.nan_to_num(self.vw, nan=0.0)
            # Angle of the vector, measured from x+ (right) counter-clockwise
            angle = np.arctan2(v_total, u_total)  # standard unit circle angle
            # Convert to "from" direction (i.e., 180° opposite), then to degrees
            self._wdir = (np.degrees(angle)) % 360
        return self._wdir

    def add_boundary_layer(self, alpha_shear=None, z0=None, TI=None):
        """
        Apply a boundary layer inflow profile to the streamwise velocity field.

        Supports multiple methods:
        - Power-law profile with exponent `alpha_shear`
        - Log-law profile with surface roughness `z0`
        - TI-based log-law using turbulence intensity `TI` and neutral ABL assumptions

        Parameters
        ----------
        alpha_shear : float, optional
            Shear exponent for power-law profile, e.g. 0.14 to 0.2 for typical ABL
        z0 : float, optional
            Surface roughness length [m] used in log-law profile
        TI : float, optional
            Turbulence intensity at hub height for TI-based velocity profile

        Notes
        -----
        The method modifies `self.U` directly and prints which method was applied.
        """ 

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

    def add_veer(self, 
                        mode='linear', 
                        vtop=1, vbottom=-1, 
                        theta=0, th=0, D=0, Uh=0,
                        ):
        """
        Impose veer (spanwise velocity variation with height) in the inflow.

        Parameters
        ----------
        mode : str
            'linear' for velocity gradient from `vtop` to `vbottom`,
            or 'theta' for veer angle [deg] centered at hub height.
        vtop : float
            Spanwise velocity at top of the domain [m/s] (used in linear mode)
        vbottom : float
            Spanwise velocity at ground [m/s] (used in linear mode)
        theta : float
            Veer angle [deg] between top and bottom of rotor (used in theta mode)
        th : float
            Hub height [m] (used in theta mode)
        D : float
            Rotor diameter [m] (used in theta mode)
        Uh : float
            Hub-height streamwise velocity [m/s] (used in theta mode)

        Notes
        -----
        Modifies the spanwise velocity field `self.V` directly.
        """
        # Apply the veer based on angle
        if mode=='theta':
            # ~ v = np.tan(np.deg2rad(theta)) * Uh / 2
            # ~ v_int = interp1d([th-D/2, th+D/2], [v, -v], fill_value="extrapolate")
            # ~ self.V += v_int(self.Z)
            
            # Formula for linear angle
            angle = np.deg2rad(theta) / 2
            ang_int = interp1d([th-D/2, th+D/2], [angle, -angle], fill_value="extrapolate")
            self.V += np.tan(ang_int(self.Z)) * self.U
            print('Added veer based on theta')
            
        elif mode=='linear':
            # The slope based on the distance from the bottom of the tower
            m = (vtop - vbottom ) / (self.z[-1] - self.z[0])
    
            # The intercept
            b = vtop - m * self.z[-1] 
            
            # Add a spanwise linear velocity profile to mimic wind veer 
            self.V += m * self.Z + b

            # Print on the screen added veer
            print('Added veer based on vtop and vbottom')  

    def add_turbulence_model(self, f=1, C=4):
        """
        Apply a turbulence model to modify the kinematic viscosity `nu` using a mixing length approach.

        The model is based on atmospheric boundary layer (ABL) mixing length theory using
        the Blackadar formulation. A maximum length scale of ~27 m (or scaled by `f`) is used.

        Parameters
        ----------
        f : float, optional
            Scaling factor applied to the mixing length (default is 1)
        C : float, optional
            Scaling factor applied to the turbulent viscosity

        Notes
        -----
        - Uses the velocity gradient ∂U/∂z from the base flow to compute turbulent viscosity.
        - Updates `self.nu` to the maximum of base (numerical) viscosity and turbulent model.
        - Prints diagnostics about viscosity range and model application.
        """

        # Velocity gradient in the z direction at hub height
        # This gradient is for the base flow in the atmosphere
        # ~ dudz = np.gradient(self.U, self.dz, axis=0)
        dudz = np.gradient(self.U, self.dz, axis=2)

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
        self.nu = C * np.maximum(nu_t, self.nu_min)

        print('Added mixing length turbulent viscosity')

        # Print maximum and minimum nu
        print('Minimum turbulent viscosity =', np.amin(self.nu))


    def save_turbine_power(self, fname='turbine_power.csv'):
        '''
        Save the turbine power to a file
        '''
        # Get time vector (assumes all turbines share the same time base)
        times = self.turbines[0].time

        # Build a dictionary: keys are turbine names, values are power time series in kW
        power_data = {
            turb.name: np.array(turb.pwr_time) * 1e-3 for turb in self.turbines
        }

        # Create DataFrame with time as index
        df = pd.DataFrame(power_data, index=times)
        df.index.name = 'Time'

        # Save to CSV
        df.to_csv(os.path.join(self.saveDir, fname))

    def calculate_turbine_error(self, scada):
        '''
        Compute the turbine error based on scada
        '''
        for turb in self.turbines:
    
            # Interpolate the SCADA
            scada_int = interp1d(scada.time_seconds, scada.power[turb.name], kind='linear', fill_value="extrapolate")

            # Calculate error vs SCADA
            power_vals = np.array(turb.pwr_time) * 1e-3
            scada_vals = scada_int(turb.time)
            turb.err_time = (power_vals - scada_vals) / scada_vals * 100
            turb.scada_power = scada_vals  # save scada power in object
            if np.allclose(power_vals, 0):
                turb.err = 0
                turb.pearson = 1
            else:
                turb.err = np.mean(power_vals - scada_vals) / np.mean(scada_vals) * 100
                turb.pearson, _ = pearsonr(power_vals, scada_vals)

# Attach the methods to the class
wind_farm_class.solve = solve
wind_farm_class.solve_in_time = solve_in_time
wind_farm_class.solve_steady_state_in_time = solve_steady_state_in_time

# Import the plot methods from other file
for name in dir(methods_plot):
    func = getattr(methods_plot, name)
    if isinstance(func, types.FunctionType):
        setattr(wind_farm_class, name, func)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
