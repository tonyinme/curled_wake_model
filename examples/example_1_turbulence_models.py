#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  wind_farm.py
#
#  Copyright 2025 Martinez <lmartin1@LMARTIN1-31527S>
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

import curled_wake_model as cwm

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change current working directory to that location
os.chdir(script_dir)

def main(args):

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
        cwm.turbine_model_class(
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
    wf = cwm.wind_farm_class(
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                Nx=int(Lx/8),
                Ny=int(Ly/10),
                Nz=int(Lz/10),
                Uh=10,
                h=90,
                turbines=turbines,
                saveDir='example_1_results',
                )

    # Add the boundary layer
    wf.add_boundary_layer(alpha_shear=0.2)
    # Add the turbulence model
    wf.add_turbulence_model()

    # Loop through the different turbulence model options
    for turb in ['Scott', 'standard', 'kl']:

        # Call the volver
        wf.solve(nut_model=turb)

        wf.plot_streamwise(name=f'plane_h_{turb}_plane.png')

        # Plot the power from all turbines
        power = [t.power() for t in turbines]
        plt.plot(power, 'o-', label=turb)

    plt.xlabel('Turbine')
    plt.ylabel('Power [MW]')
    plt.legend()
    plt.savefig(os.path.join(wf.saveDir,'power.png'))




if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
