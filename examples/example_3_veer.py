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

    D = 126  # diameter [m]
    Lx = 12*D  # length of domain [m]
    Ly=6*D  # width of domain [m]
    Lz = 300 # height of domain [m]

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
                        Cp=None,
                        # Yaw angle [deg]
                        alpha=0,
                        tilt=0, # [deg]
                        # Hub height velocity [m/s]
                        Uh=8.,
                        # Tip speed ratio (specifies the direction with + or -)
                        tsr=8,
                        # Flag to include the ground effects
                        ground=True,
                        # Location
                        location=(2*D, Ly/2, 90)
                        )
     ]

    # Initiate the wind farm object
    wf = cwm.wind_farm_class(
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                Nx=int(Lx/5),
                Ny=int(Ly/8),
                Nz=int(Lz/8),
                Uh=10,
                h=90,
                turbines=turbines,
                saveDir='example_3_results',
            )

    # Add the boundary layer
    wf.add_boundary_layer(TI=0.02)
    # Add the turbulence model
    wf.add_turbulence_model()
    # Add veer
    wf.add_veer(mode='theta', theta=20, th=90, D=126)

    # Call the volver
    wf.solve(nut_model='standard')

    wf.plot_streamwise(name=f'plane_h_plane.png')

    # Plot different downstream locations
    for i in range(1, 10, 2):
        x = 2*D + i * D
        wf.plot_x_plane(x=x, file_name='x'+str(i) + 'D.png', vmin=.5, vmax=1., rotors=True)




if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
