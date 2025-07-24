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

import os
import shutil

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle

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
matplotlib.rcParams['axes.labelsize'] = 24 
matplotlib.rcParams['axes.titlesize'] = 24 
matplotlib.rcParams['legend.fontsize'] = 14

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'figure.autolayout': True})

# For font types (Journal does not accept type 3 fonts)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Attempt to find ffmpeg in system PATH
ffmpeg_path = shutil.which("ffmpeg") or os.path.join(
    os.environ["USERPROFILE"],
    r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
)

if ffmpeg_path is not None:
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
else:
    raise FileNotFoundError("ffmpeg executable not found in system PATH.")


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

def plot_streamwise_new(self, name='horizontalProfile.png', field='Uh+uw', z=None,
                        coord_transform=None, background=False, vmin=None, vmax=None,
                        xlim=None, ylim=None, title=None, turbine_names=None,
                        turb_loc=True, video=False, turbine_error=False, cmap='viridis',
                        all_times=False, compare_data=None):
    """
    Plot horizontal profiles of a specified velocity-related field at hub height.
    Parameters:
        field: str, one of ['Uh+uw', 'vmag', 'U', 'uw', etc.]
        coord_transform: callable, (X, Y) -> (Xp, Yp), applied to meshgrid of self.x, self.y
    """

    if z is None:
        z = self.h
    i = np.abs(self.z - z).argmin()  # hub height index

    if all_times and hasattr(self, 'u_time'):
        for idx, frame in enumerate(self.u_time):
            data = (frame[:, :, i] + self.Uh) if not background else frame[:, :, i] 
            #data = data / self.Uh
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            if coord_transform:
                Xp, Yp = coord_transform(self.time_video[idx], X, Y)
            else:
                Xp, Yp = X, Y

            if compare_data is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
                ax1.set_aspect('equal', adjustable='box')
                ax2.set_aspect('equal', adjustable='box')
                ax3.set_aspect('equal', adjustable='box')
                if not vmin: vmin=np.amin(data)
                if not vmax: vmax=np.amax(data)
            else:
                fig, ax1 = plt.subplots()
                ax1.set_aspect('equal', adjustable='box')

            img1 = ax1.pcolormesh(Xp, Yp, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
            ax1.set_title(f"Model at t={self.time_video[idx]:.1f} s")

            if compare_data is not None:
                try:
                    compare = compare_data(self.time_video[idx], Xp, Yp)
                except Exception as e:
                    print(f"Error in compare_data at time {self.time_video[idx]}: {e}")
                    compare = np.zeros_like(data)
                img2 = ax2.pcolormesh(Xp, Yp, compare, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
                ax2.set_title(f"Data at t={self.time_video[idx]:.1f} s")

                error_data = data-compare

                valid_mask = ~np.isnan(compare)
                err_metric = np.nanmean(np.abs(error_data[valid_mask])) # MAE
                val_max = np.nanmax(np.abs(error_data))  # max value of the error for the colorbar
                img3 = ax3.pcolormesh(Xp, Yp, error_data, cmap='bwr', vmin=-val_max, vmax=val_max, shading='nearest')

                ax3.set_title(f"MAE at t={self.time_video[idx]:.1f} s is {err_metric:.2f} m/s")

                for a in (ax1, ax2, ax3):
                    a.set_xlabel(r'$x$ [m]')
                    a.set_ylabel(r'$y$ [m]')
                    if xlim: a.set_xlim(xlim)
                    if ylim: a.set_ylim(ylim)
                    if turb_loc:
                        for t in self.turbines:
                            ################### FIX THIS, the time is off, it should interpolate the turbine data
                            idt = np.abs(np.array(t.time) - self.time_video[idx]).argmin()
                            x1 = x2 = t.location_time[idt][0] #t.location[0]
                            #y1, y2 = t.location[1] - t.D / 2, t.location[1] + t.D / 2
                            y1, y2 = t.location_time[idt][1] - t.D / 2, t.location_time[idt][1] + t.D / 2
                            xm, ym = x1, (y1 + y2) / 2
                            alpha = t.alpha_fun(self.time_video[idx]) if hasattr(t, 'alpha_fun') else t.alpha
                            dx, dy = x1 - xm, y1 - ym
                            xr1 = xm + dx * np.cos(alpha) - dy * np.sin(alpha)
                            yr1 = ym + dy * np.cos(alpha) + dx * np.sin(alpha)
                            dx, dy = x2 - xm, y2 - ym
                            xr2 = xm + dx * np.cos(alpha) - dy * np.sin(alpha)
                            yr2 = ym + dy * np.cos(alpha) + dx * np.sin(alpha)
                            if coord_transform:
                                xr, yr = coord_transform(self.time_video[idx], np.array([[xr1, xr2]]), np.array([[yr1, yr2]]))
                                xr1, xr2 = xr[0, 0], xr[0, 1]
                                yr1, yr2 = yr[0, 0], yr[0, 1]
                            a.plot([xr1, xr2], [yr1, yr2], '-k', lw=2)
                fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
                fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
                fig.colorbar(img3, ax=ax3, fraction=0.046, pad=0.04)
            else:
                fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)

            fig.savefig(self.saveDir + f'/{name[:-4]}_t{idx:04d}.png', bbox_inches='tight', dpi=250)
            plt.close(fig)
        return

    # Fallback to single frame mode (not all_times)
    if field == 'Uh+uw':
        data = np.asarray(self.uw[:, :, i] + self.Uh)
    elif field == 'wspd':
        data = self.wspd[:, :, i]
    elif field == 'wdir':
        data = self.wdir[:, :, i]
    elif hasattr(self, field):
        data = np.asarray(getattr(self, field)[:, :, i])
    else:
        raise ValueError(f"Unknown field: {field}")

    if field == 'Uh+uw':
        data /= self.Uh

    X, Y = np.meshgrid(self.x, self.y, indexing='ij')
    if coord_transform:
        Xp, Yp = coord_transform(0, X, Y)
    else:
        Xp, Yp = X, Y

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    img = ax.pcolormesh(Xp, Yp, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')

    if turb_loc:
        for t in self.turbines:
            x1 = x2 = t.location[0]
            y1, y2 = t.location[1] - t.D / 2, t.location[1] + t.D / 2
            xm, ym = x1, (y1 + y2) / 2
            dx, dy = x1 - xm, y1 - ym
            xr1 = xm + dx * np.cos(t.alpha) - dy * np.sin(t.alpha)
            yr1 = ym + dy * np.cos(t.alpha) + dx * np.sin(t.alpha)
            dx, dy = x2 - xm, y2 - ym
            xr2 = xm + dx * np.cos(t.alpha) - dy * np.sin(t.alpha)
            yr2 = ym + dy * np.cos(t.alpha) + dx * np.sin(t.alpha)
            if coord_transform:
                xr, yr = coord_transform(0, np.array([[xr1, xr2]]), np.array([[yr1, yr2]]))
                xr1, xr2 = xr[0, 0], xr[0, 1]
                yr1, yr2 = yr[0, 0], yr[0, 1]
            ax.plot([xr1, xr2], [yr1, yr2], '-k', lw=2)
            if turbine_names:
                label_x, label_y = t.location[0] + 300, t.location[1] + 100
                if coord_transform:
                    label_x, label_y = coord_transform(0, np.array([[label_x]]), np.array([[label_y]]))
                    label_x, label_y = label_x[0, 0], label_y[0, 0]
                ax.text(label_x, label_y, t.name)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if title: ax.set_title(title)

    if turbine_error:
        x = np.array([t.location[0] for t in self.turbines])
        y = np.array([t.location[1] for t in self.turbines])
        if coord_transform:
            x, y = coord_transform(0, x[None, :], y[None, :])
            x, y = x[0], y[0]
        errors = [t.err for t in self.turbines]
        sc = ax.scatter(x, y, c=errors, cmap='RdBu_r', edgecolors='k', s=100, vmin=-20, vmax=20)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()

    if turbine_error:
        cax2 = divider.append_axes("right", size="5%", pad=0.4)
        cbar2 = fig.colorbar(sc, cax=cax2)
        cbar2.set_label("% Error", fontsize=12)
        cax.set_position([cax.get_position().x0 - 0.05, cax.get_position().y0,
                        cax.get_position().width, cax.get_position().height])

    fig.savefig(self.saveDir + '/' + name, bbox_inches='tight', dpi=250)

    if video and hasattr(self, 'u_time'):
        if field not in ('Uh+uw', 'wspd'):
            raise NotImplementedError("Video is only implemented for field='Uh+uw' or 'wspd'")

        def compute_transformed_data():
            out = []
            for frame in self.u_time:
                d = (frame[:, :, i] + self.Uh) / self.Uh if not background else frame[:, :, i] / self.Uh
                if coord_transform:
                    X, Y = np.meshgrid(self.x, self.y, indexing='ij')
                    Xp, Yp = coord_transform(self.time_video[frame], X, Y)
                else:
                    Xp, Yp = X, Y
                out.append((Xp, Yp, d))
            return out

        transformed_data = compute_transformed_data()

        def update(frame):
            Xp, Yp, d = transformed_data[frame]
            img.set_array(d.flatten())
            img.set_clim(vmin, vmax)
            ax.set_title(f"Simulation Time: {self.time_video[frame]:.1f} s")

            if turb_loc:
                for line, t in zip(ax.get_lines(), self.turbines):
                    x1 = x2 = t.location_time[frame][0]
                    y1, y2 = t.location_time[frame][1] - t.D / 2, t.location_time[frame][1] + t.D / 2
                    xm, ym = x1, (y1 + y2) / 2
                    alpha = t.alpha_fun(self.time_video[frame])
                    dx, dy = x1 - xm, y1 - ym
                    xr1 = xm + dx * np.cos(alpha) - dy * np.sin(alpha)
                    yr1 = ym + dy * np.cos(alpha) + dx * np.sin(alpha)
                    dx, dy = x2 - xm, y2 - ym
                    xr2 = xm + dx * np.cos(alpha) - dy * np.sin(alpha)
                    yr2 = ym + dy * np.cos(alpha) + dx * np.sin(alpha)
                    if coord_transform:
                        xr, yr = coord_transform(self.time_video[frame], np.array([[xr1, xr2]]), np.array([[yr1, yr2]]))
                        xr1, xr2 = xr[0, 0], xr[0, 1]
                        yr1, yr2 = yr[0, 0], yr[0, 1]
                    line.set_data([xr1, xr2], [yr1, yr2])

            if turbine_error:
                errors = [np.interp(self.time_video[frame], t.time, t.err_time) for t in self.turbines]
                sc.set_array(np.array(errors))

            return img,

        ani = FuncAnimation(fig, update, frames=len(self.u_time), blit=True, cache_frame_data=False)
        fps = 2.5 * 60 / self.dt
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='OpenAI'), bitrate=5000)
        ani.save(self.saveDir + '/' + name[:-4] + "_time_history.mp4", writer=writer)

    plt.close(fig)


def plot_streamwise(self, name='horizontalProfile.png', background=False,
        vmin=None, vmax=None, xlim=None, ylim=None, title=None, yshift=None,
        turbine_names=None, turb_loc=True,
        video=False, turbine_error=False):
    '''
    Plot horizontal profiles
    '''

    # The field perturbation
    # ~ uw = np.asarray(self.uw)
    uw = self.uw

    fig, ax = plt.subplots()

    # The 2-D coordinates in a plane
    # ~ X, Y = np.meshgrid(self.x , self.y )

    # Index that shows minimum distance to tower height position
    i = np.abs(self.z - self.h).argmin()

    # Slice the data at a plane at hub height
    if background:
        #data = np.asarray(uw[:, :, i]).T
#            data = np.asarray((uw + self.U)[:, :, i]).T
        data = np.asarray((uw + self.Uh)[:, :, i]).T
        #print(np.shape(data))
        #print(np.shape(self.U))

        # ~ for j in range(self.Nx): data[:, j] += self.U[i, :]
    else:
        data = self.Uh + np.asarray(uw[:, :, i]).T
    data /= self.Uh

    # Shift the domain for the plotting
    # ~ if yshift: Y += yshift

    for t in self.turbines:
        x1 = t.location[0]
        x2 = t.location[0]
        y1 = t.location[1] - t.D / 2 
        y2 = t.location[1] + t.D / 2
        
        # ~ # Shift the domain for the plotting
        # ~ if yshift: 
            # ~ y1 += yshift
            # ~ y2 += yshift
        
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
        
        if turb_loc: 
            plt.plot([xr1, xr2], [yr1, yr2], '-k', lw=2)
        
        if turbine_names: 
            plt.text(t.location[0] + 300, t.location[1]+100, t.name)  # Adjust offset if needed


    # The point at which to sample
    # Same size x and y
    ax.set_aspect('equal', adjustable='box')
    img = ax.pcolormesh(self.x, self.y, data, #X, Y, data,
            cmap='viridis', shading='nearest', #shading='gouraud', 
            vmin=vmin, vmax=vmax)

    # Set the figure labels
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    if title: ax.set_title(title)

    if turbine_error:
        # Extracting data
        x = np.array([t.location[0] for t in self.turbines])
        y = np.array([t.location[1] for t in self.turbines])
        errors = np.array([t.err for t in self.turbines])

        # Scatter plot with colormap
        sc = plt.scatter(x, y, c=errors, cmap='RdBu_r', edgecolors='k', s=100, vmin=-20, vmax=20)


    # Set the colormap the same size as the y axis
    x1, x2 = ax.set_xlim()
    y1, y2 = ax.set_ylim()

    # Set the colorbar
    # ~ ax = img.axes
    # ~ fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    #~ cax.yaxis.get_major_locator().set_params(nbins=3)

    # Update the number of ticks
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # If turbine errors are plotted, add second colorbar
    if turbine_error:
        cax2 = divider.append_axes("right", size="5%", pad=0.4)
        cbar2 = fig.colorbar(sc, cax=cax2)
        cbar2.set_label("Turbine Error")
        # Adjust spacing between colorbars
        cax.set_position([cax.get_position().x0 - 0.05, cax.get_position().y0, cax.get_position().width, cax.get_position().height])
        cbar2.set_label("% Error", fontsize=12)  # Label for turbine error


    fig.savefig(self.saveDir + '/' + name,
            bbox_inches='tight', dpi=250)
            
    if video:

        if background:
            transposed_data = [frame_data[:, :, i].T / self.Uh for frame_data in self.u_time]
        else: 
            transposed_data = [1 + frame_data[:, :, i].T / self.Uh for frame_data in self.u_time]

        # Let's save the video now
        # Function to update the data for each frame
        def update(frame):
#                new_data = 1 + self.u_time[frame][:, :, i].T / self.Uh
            # ~ new_data = data
#                img.set_array(new_data.ravel())  # pcolormesh needs flattened data
            img.set_array(transposed_data[frame].flatten())
            ax.set_title(f"Simulation Time: {self.time_video[frame]:.1f} s")

            lines = ax.get_lines()
            for i, t in enumerate(self.turbines):
                idt = np.abs(np.array(t.time) - self.time_video[frame]).argmin()
                t.location = t.location_time[idt]
                x1 = t.location[0]
                x2 = t.location[0]
                y1 = t.location[1] - t.D / 2 
                y2 = t.location[1] + t.D / 2
                
                alpha = t.alpha_fun(self.time_video[frame])
                # The mid point which to rotate about
                xm = (x1 + x2) / 2
                ym = (y1 + y2) / 2
                xr1 = (x1-xm) * np.cos(alpha) - (y1-ym) * np.sin(alpha)
                yr1 = (y1-ym) * np.cos(alpha) + (x1-xm) * np.sin(alpha)
                xr2 = (x2-xm) * np.cos(alpha) - (y2-ym) * np.sin(alpha)
                yr2 = (y2-ym) * np.cos(alpha) + (x2-xm) * np.sin(alpha)
                xr1 += xm
                xr2 += xm
                yr1 += ym
                yr2 += ym
                
                if turb_loc: 
                    lines[i].set_data([xr1, xr2], [yr1, yr2])

            if turbine_error:
                errors = np.array([np.interp(self.time_video[frame], t.time, t.err_time) for t in self.turbines])
                sc.set_array(errors)  # update color values
            
            return img,
            
        ani = FuncAnimation(fig, update, frames=len(self.u_time), blit=True, cache_frame_data=False)
        # Save the animation as a video
        fps = 2.5*60 / self.dt # (seconds of simulation per frame)
        #fps = 1 / self.dt
        bitrate=5000 # 1200
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Tony'), bitrate=bitrate)
        #writer = PillowWriter(fps=20)  # Use PillowWriter instead of FFMpegWriter
        ani.save(self.saveDir + '/' + name[:-4] + "_time_history.mp4", writer=writer)
        #ani.save(self.saveDir + '/' + "time_history.gif", writer=writer)

    plt.close(fig)

def plot_error_turbines(self, file_name='error_turbines.png', vmin=-20, vmax=20):
    '''
    Plot a contour of the errors
    '''

    if not self.turbines:
        return

    fig, ax = plt.subplots()

    # Extracting data
    x = np.array([t.location[0] for t in self.turbines])
    y = np.array([t.location[1] for t in self.turbines])
    errors = np.array([t.err for t in self.turbines])

    for t in self.turbines:
        #plt.text(t.location[0] + 300, t.location[1]+100, t.name)  # Adjust offset if needed
        plt.text(t.location[0] + 200, t.location[1] + 250, f"{t.name}\n{int(t.err)}%", ha='left', va='top')


    # Scatter plot with colormap
    sc = plt.scatter(x, y, c=errors, cmap='RdBu_r', edgecolors='k', s=100, vmin=vmin, vmax=vmax)

    ax.set_aspect('equal', adjustable='box')

    # The limit of the power label 
    ax.set_xlim([x.min()-500, x.max()+1300]) 
    ax.set_ylim([y.min()-500, y.max()+500]) 

    # Set the figure labels
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')

    #fig.colorbar(sc, label="Error")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sc, cax=cax)
    #~ cax.yaxis.get_major_locator().set_params(nbins=3)

    # Update the number of ticks
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
    cbar.locator = tick_locator
    cbar.update_ticks()

    fig.savefig(os.path.join(self.saveDir, file_name), bbox_inches='tight', dpi=250)
    plt.close(fig)

def plot_qoi_turbines(self, file_name='qoi_turbines.png', qoi='pwr_ti', vmin=None, vmax=None, fmt="{:.2f}", ):
    '''
    Plot a contour of the errors
    '''

    if not self.turbines:
        return

    fig, ax = plt.subplots()

    # Extracting data
    x = np.array([t.location[0] for t in self.turbines])
    y = np.array([t.location[1] for t in self.turbines])

    if qoi == 'pwr_ti':
        for t in self.turbines: t.qoi = np.std(t.scada_power)/np.mean(t.scada_power)
        errors = np.array([t.qoi for t in self.turbines])

    for t in self.turbines:
        #plt.text(t.location[0] + 300, t.location[1]+100, t.name)  # Adjust offset if needed
        plt.text(t.location[0] + 200, t.location[1] + 250, f"{t.name}\n{fmt.format(t.qoi)}", ha='left', va='top')


    # Scatter plot with colormap
    sc = plt.scatter(x, y, c=errors, cmap='RdBu_r', edgecolors='k', s=100, vmin=vmin, vmax=vmax)

    ax.set_aspect('equal', adjustable='box')

    # The limit of the power label 
    ax.set_xlim([x.min()-500, x.max()+1300]) 
    ax.set_ylim([y.min()-500, y.max()+500]) 

    # Set the figure labels
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')

    #fig.colorbar(sc, label="Error")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sc, cax=cax)
    #~ cax.yaxis.get_major_locator().set_params(nbins=3)

    # Update the number of ticks
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
    cbar.locator = tick_locator
    cbar.update_ticks()

    fig.savefig(os.path.join(self.saveDir, file_name), bbox_inches='tight', dpi=250)
    plt.close(fig)

def plot_pearson_turbines(self, file_name='pearson_turbines.png', vmin=0, vmax=1):
    '''
    Plot a contour of the pearson coefficient
    '''
    if not self.turbines:
        return

    fig, ax = plt.subplots()

    # Extracting data
    x = np.array([t.location[0] for t in self.turbines])
    y = np.array([t.location[1] for t in self.turbines])
    pearson = np.array([t.pearson for t in self.turbines])

    for t in self.turbines:
        #plt.text(t.location[0] + 300, t.location[1]+100, t.name)  # Adjust offset if needed
        plt.text(t.location[0] + 200, t.location[1] + 250, f"{t.name}\n$\\rho={"{:.2f}".format(t.pearson)}$", ha='left', va='top')

    # Scatter plot with colormap
    sc = plt.scatter(x, y, c=pearson, cmap='RdBu_r', edgecolors='k', s=100, vmin=vmin, vmax=vmax)

    ax.set_aspect('equal', adjustable='box')

    # The limit of the power label 
    ax.set_xlim([x.min()-500, x.max()+1300]) 
    ax.set_ylim([y.min()-500, y.max()+500]) 

    # Set the figure labels
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sc, cax=cax)

    # Update the number of ticks
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
    cbar.locator = tick_locator
    cbar.update_ticks()

    fig.savefig(os.path.join(self.saveDir, file_name), bbox_inches='tight', dpi=250)
    plt.close(fig)
                
def plot_x_plane(self, x=300, file_name='downstream.pdf',
        vmin=None, vmax=None, streamplot=False,
        cmap='viridis', savedata=False, contourf=False,
        video=False, rotors=False):
    '''
    Plot x profiles

    x - the location downstream
    '''

    # The field perturbation
    #uw = np.asarray(self.uw)
    uw = self.uw

    fig, ax = plt.subplots(figsize=(20, 6))

    # Index that shows minimum distance to tower height position
    i = np.abs(self.x - x).argmin()

    # Slice the data at a plane at hub height
    #data = ((self.U + uw)[i, :, :]).T
    data = np.asarray((uw + self.Uh)[i, :, :]).T
    #data = np.asarray(self.U[i, :, :]).T
    # ~ data = -np.asarray(uw[i, :, :])#.T
    data /= self.Uh

    # The point at which to sample
    # Same size x and y
    ax.set_aspect('equal', adjustable='box')

    if contourf:
        img = plt.contourf(self.y, self.z, data, 56,
            cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        
    else:
        img = plt.pcolormesh(self.y, self.z, data,
            cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

    ax.set_xlim([np.amin(self.y), np.amax(self.y)])
    ax.set_ylim([np.amin(self.z), np.amax(self.z)])

    # Set the figure labels
    ax.set_xlabel(r'$y$ [m]')
    ax.set_ylabel(r'$z$ [m]')

    # Set the colormap the same size as the y axis
    x1, x2 = plt.xlim()
    y1, y2 = plt.ylim()

    # Scale for the density based on axes
    scale = 1#(self.Ly / self.h) / np.abs(x1 - x2)

    if streamplot:
    # Plot streamlines
        #plt.streamplot(self.Y[i,:,:], self.Z[i,:,:],
        #                self.V[i,:,:],
        #                self.W[i,:,:],
        #                color='k', density=1.5 * scale)
        Y_grid, Z_grid = np.meshgrid(self.y, self.z,)# indexing='ij')
        ax.streamplot(Y_grid, Z_grid,
            #(self.V[i,:,:]+self.vw[i,:,:]).T, (self.W[i,:,:]+self.ww[i,:,:]).T,
            self.vw[i,:,:].T, self.ww[i,:,:].T,
            color='k', density=(5, 2)
            )

    if rotors:
        for turb in self.turbines:
            # Add rotor diameter
            circle = Circle((turb.location[1], turb.location[2]), radius=turb.D/2, edgecolor='k', facecolor='none', linewidth=2)
            ax.add_patch(circle)


    if not contourf:
        # Set the colorbar
        #ax = img.axes
        #fig = ax.figure
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

    if video:

        transposed_data = [frame_data[i, :, :].T / self.Uh for frame_data in self.u_time]
        #transposed_data = [1 + frame_data[i, :, :].T / self.Uh for frame_data in self.u_time]

        # Let's save the video now
        # Function to update the data for each frame
        def update(frame):
#                img.set_array(new_data.ravel())  # pcolormesh needs flattened data
            img.set_array(transposed_data[frame].flatten())
#                ax.set_title(f"Simulation Time: {self.time_video[frame]:.1f} s")
            
            return img,
            
        ani = FuncAnimation(fig, update, frames=len(self.u_time), blit=True, cache_frame_data=False)
        # Save the animation as a video
        fps = 2.5*60 / self.dt # (seconds of simulation per frame)
        #fps = 1 / self.dt
        bitrate=5000 # 1200
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Tony'), bitrate=bitrate)
        #writer = PillowWriter(fps=20)  # Use PillowWriter instead of FFMpegWriter
        ani.save(self.saveDir + '/' + file_name[:-4] + "_time_history.mp4", writer=writer)

    plt.close(fig)

