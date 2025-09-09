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

import numpy as np
from numba import njit



@njit
def finite_diff(arr, dy, dz):
    """
    Compute finite differences in a grid with 'ij' indexing:
    - First dimension (axis 0): y-direction
    - Second dimension (axis 1): z-direction

    Parameters:
        arr: 2D numpy array of shape (ny, nz)
        dy: Grid spacing in the y-direction (axis 0)
        dz: Grid spacing in the z-direction (axis 1)
    
    Returns:
        duwdy: Approximation of partial derivative w.r.t. y (axis 0)
        duwdz: Approximation of partial derivative w.r.t. z (axis 1)
    """
    ny, nz = arr.shape  # Number of points in y and z directions
    duwdy = np.zeros_like(arr, dtype=np.float32)  # Partial derivative w.r.t. y
    duwdz = np.zeros_like(arr, dtype=np.float32)  # Partial derivative w.r.t. z

    # Central difference (interior points)
    for i in range(1, ny - 1):  # Iterate over y (axis 0)
        for j in range(1, nz - 1):  # Iterate over z (axis 1)
            # Derivative w.r.t. y (axis 0)
            duwdy[i, j] = (arr[i + 1, j] - arr[i - 1, j]) / (2 * dy)
            # Derivative w.r.t. z (axis 1)
            duwdz[i, j] = (arr[i, j + 1] - arr[i, j - 1]) / (2 * dz)

    # Handle edges (copy values from adjacent points)
    duwdy[0, :] =  0. # duwdy[1, :]      # Edge at y = 0
    duwdy[-1, :] = 0. # duwdy[-2, :]    # Edge at y = ny - 1
    duwdz[:, 0] = 0. # duwdz[:, 1]      # Edge at z = 0
    duwdz[:, -1] = 0. # duwdz[:, -2]    # Edge at z = nz - 1

    return duwdy, duwdz


@njit
def finite_diff_3d(arr, dx, dy, dz):
    """
    Compute finite differences in a 3D grid with 'ijk' indexing:
    - First dimension (axis 0): x-direction
    - Second dimension (axis 1): y-direction
    - Third dimension (axis 2): z-direction

    Parameters:
        arr: 3D numpy array of shape (nx, ny, nz)
        dx: Grid spacing in the x-direction (axis 0)
        dy: Grid spacing in the y-direction (axis 1)
        dz: Grid spacing in the z-direction (axis 2)
    
    Returns:
        duwdx: Approximation of partial derivative w.r.t. x (axis 0)
        duwdy: Approximation of partial derivative w.r.t. y (axis 1)
        duwdz: Approximation of partial derivative w.r.t. z (axis 2)
    """
    nx, ny, nz = arr.shape  # Number of points in x, y, and z directions
    duwdx = np.zeros_like(arr, dtype=np.float32)  # Partial derivative w.r.t. x
    duwdy = np.zeros_like(arr, dtype=np.float32)  # Partial derivative w.r.t. y
    duwdz = np.zeros_like(arr, dtype=np.float32)  # Partial derivative w.r.t. z

    # Central difference (interior points)
#    for i in range(1, nx - 1):  # Iterate over x (axis 0)
    for i in range(1, nx):  # Iterate over x (axis 0)
        for j in range(1, ny - 1):  # Iterate over y (axis 1)
            for k in range(1, nz - 1):  # Iterate over z (axis 2)

                # These are numerically unstable
                #if i>2:
                #    duwdx[i, j, k] = (11 * arr[i,j,k] - 18 * arr[i-1,j,k] + 9 * arr[i-2,j,k] - 2 * arr[i-3,j,k]) / (6 * dx)
                    #duwdx[i, j, k] = (3 * arr[i, j, k] - 4 * arr[i - 1, j, k] + arr[i - 2, j, k]) / (2 * dx)

                # Derivative w.r.t. x (axis 0) backward difference
                duwdx[i, j, k] = (arr[i, j, k] - arr[i - 1, j, k]) / (dx)
                # Derivative w.r.t. y (axis 1)
                duwdy[i, j, k] = (arr[i, j + 1, k] - arr[i, j - 1, k]) / (2 * dy)
                # Derivative w.r.t. z (axis 2)
                duwdz[i, j, k] = (arr[i, j, k + 1] - arr[i, j, k - 1]) / (2 * dz)

    return duwdx, duwdy, duwdz



# This is xy indexing (new code is ij)
# ~ @njit
# ~ def finite_diff(arr, dy, dz):
    # ~ ny, nz = arr.shape
    # ~ duwdy = np.zeros_like(arr)
    # ~ duwdz = np.zeros_like(arr)

    # ~ # Central difference (interior points)
    # ~ for i in range(1, ny - 1):
        # ~ for j in range(1, nz - 1):
            # ~ duwdy[i, j] = (arr[i, j + 1] - arr[i, j - 1]) / (2 * dy)
            # ~ duwdz[i, j] = (arr[i + 1, j] - arr[i - 1, j]) / (2 * dz)

    # ~ # Handle edges (copy values from adjacent points)
    # ~ duwdy[:, 0] = duwdy[:, 1]
    # ~ duwdy[:, -1] = duwdy[:, -2]
    # ~ duwdz[0, :] = duwdz[1, :]
    # ~ duwdz[-1, :] = duwdz[-2, :]

    # ~ return duwdy, duwdz

@njit
def laplacian3d(u, dx, dy, dz):
    nx, ny, nz = u.shape
    #d2udx2 = np.zeros_like(u)
    d2udy2 = np.zeros_like(u, dtype=float)
    d2udz2 = np.zeros_like(u, dtype=float)

    # Compute second derivatives in y-direction (axis=1)
    for i in range(nx):
        for j in range(2, ny-2):
            for k in range(2, nz-2):
                #d2udx2[i, j, k] = (u[i - 2, j, k] - 2 * u[i, j, k] + u[i + 2, j, k]) / (4 * dx * dx)
                d2udy2[i, j, k] = (u[i, j - 2, k] - 2 * u[i, j, k] + u[i, j + 2, k]) / (4 * dy * dy)
                d2udz2[i, j, k] = (u[i, j, k - 2] - 2 * u[i, j, k] + u[i, j, k + 2]) / (4 * dz * dz)

    # Second-order accurate one-sided differences for boundaries
    #d2udx2[0, :, :] = (u[2, :, :]/2 - u[0, :, :]/2 - u[1, :, :] + u[0, :, :]) / (dx * dx)
    #d2udx2[1, :, :] = (u[3, :, :]/2 - u[1, :, :]/2 - u[1, :, :] + u[0, :, :]) / (2 * dx * dx)  
    #d2udx2[-2, :, :] = (u[-1, :, :] - u[-2, :, :] - u[-2, :, :]/2 + u[-4, :, :]/2) / (2 * dx * dx)
    #d2udx2[-1, :, :] = (u[-1, :, :] - u[-2, :, :] - u[-1, :, :]/2 + u[-3, :, :]/2) / (dx * dx)

    d2udy2[:, 0, :] = (u[:, 2, :]/2 - u[:, 0, :]/2 - u[:, 1, :] + u[:, 0, :]) / (dy * dy)
    d2udy2[:, 1, :] = (u[:, 3, :]/2 - u[:, 1, :]/2 - u[:, 1, :] + u[:, 0, :]) / (2 * dy * dy)  
    d2udy2[:, -2, :] = (u[:, -1, :] - u[:, -2, :] - u[:, -2, :]/2 + u[:, -4, :]/2) / (2 * dy * dy)
    d2udy2[:, -1, :] = (u[:, -1, :] - u[:, -2, :] - u[:, -1, :]/2 + u[:, -3, :]/2) / (dy * dy)

    d2udz2[:, :, 0] = (u[:, :, 2]/2 - u[:, :, 0]/2 - u[:, :, 1] + u[:, :, 0]) / (dz * dz)
    d2udz2[:, :, 1] = (u[:, :, 3]/2 - u[:, :, 1]/2 - u[:, :, 1] + u[:, :, 0]) / (2 * dz * dz) 
    d2udz2[:, :, -2] = (u[:, :,  -1] - u[:, :, -2] - u[:, :, -2]/2 + u[:, :, -4]/2) / (2 * dz * dz) 
    d2udz2[:, :, -1] = (u[:, :,  -1] - u[:, :, -2] - u[:, :, -1]/2 + u[:, :, -3]/2) / (dz * dz)

    return d2udy2 + d2udz2 #+ d2udx2

@njit
def laplacian(u, dy, dz):
    ny, nz = u.shape
#    d2udy2 = np.zeros_like(u)
#    d2udz2 = np.zeros_like(u)
    lap = np.zeros_like(u, dtype=np.float32)

    # Compute second derivatives in y-direction (axis=1)
    for j in range(2, ny-2):
        for k in range(2, nz-2):
#            d2udy2[j, k] = (u[j - 2, k] - 2 * u[j, k] + u[j + 2, k]) / (4 * dy * dy)
#            d2udz2[j, k] = (u[j, k - 2] - 2 * u[j, k] + u[j, k + 2]) / (4 * dz * dz)
            lap[j,k] = (u[j - 2, k] - 2 * u[j, k] + u[j + 2, k]) / (4 * dy * dy) + (u[j, k - 2] - 2 * u[j, k] + u[j, k + 2]) / (4 * dz * dz)

    for k in range(nz):
        lap[0, k] = lap[0, k] + (u[2, k]/2 - u[0, k]/2 - u[1, k] + u[0, k]) / (dy * dy)
        lap[1, k] = lap[1, k] + (u[3, k]/2 - u[1, k]/2 - u[1, k] + u[0, k]) / (2 * dy * dy)  
        lap[-2, k] = lap[-2, k] + (u[-1, k] - u[-2, k] - u[-2, k]/2 + u[-4, k]/2) / (2 * dy * dy)
        lap[-1, k] = lap[-1, k] + (u[-1, k] - u[-2, k] - u[-1, k]/2 + u[-3, k]/2) / (dy * dy)

    for j in range(ny):
        lap[j, 0] = lap[j, 0] + (u[j, 2]/2 - u[j, 0]/2 - u[j, 1] + u[j, 0]) / (dz * dz)
        lap[j, 1] = lap[j, 1] + (u[j, 3]/2 - u[j, 1]/2 - u[j, 1] + u[j, 0]) / (2 * dz * dz) 
        lap[j, -2] = lap[j, -2] + (u[j,  -1] - u[j, -2] - u[j, -2]/2 + u[j, -4]/2) / (2 * dz * dz) 
        lap[j, -1] = lap[j, -1] + (u[j,  -1] - u[j, -2] - u[j, -1]/2 + u[j, -3]/2) / (dz * dz)


    return lap #d2udy2 + d2udz2

def laplacian_(u, dy, dz):
    '''
    Compute the laplacian in 2D
    '''
    d2udy2 = np.gradient(np.gradient(u, dy, axis=0), dy, axis=0)
    d2udz2 = np.gradient(np.gradient(u, dz, axis=1), dz, axis=1)

    return d2udy2 + d2udz2

@njit
def compute_rhs_dynamic(u_current, U, V, W, dx, dy, dz, f, nu):
    # Compute spatial derivatives using the numba-compiled functions
    duwdx, duwdy, duwdz = finite_diff_3d(u_current, dx, dy, dz)
    # Compute the right-hand side of the governing equation
    rhs = -(U + u_current) * duwdx - V * duwdy - W * duwdz + f * nu * laplacian3d(u_current, dx, dy, dz)
    return rhs

@njit
def compute_rhs_steady(u_current, U, V, W, dy, dz, nu):
    duwdy, duwdz = finite_diff(u_current, dy, dz)
    inv_U = 1.0 / (U + u_current)  # Precompute inverse
    rhs = inv_U * (-V * duwdy - W * duwdz + nu * laplacian(u_current, dy, dz))
    return rhs

@njit
def runge_kutta_step(u_current, dx, U, V, W, dy, dz, nu):
    k1 = compute_rhs_steady(u_current, U, V, W, dy, dz, nu)
    
    tmp = u_current + 0.5 * dx * k1  # In-place variable reuse
    k2 = compute_rhs_steady(tmp, U, V, W, dy, dz, nu)

    tmp = u_current + 0.5 * dx * k2
    k3 = compute_rhs_steady(tmp, U, V, W, dy, dz, nu)

    tmp = u_current + dx * k3
    k4 = compute_rhs_steady(tmp, U, V, W, dy, dz, nu)

    return u_current + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) 













######################
# Numba dk test
######################

@njit
def compute_grad_y(field, dy):
    ny, nz = field.shape
    grad = np.zeros_like(field)
    for j in range(1, ny - 1):
        for k in range(nz):
            grad[j, k] = (field[j + 1, k] - field[j - 1, k]) / (2 * dy)
    grad[0, :] = 0.0
    grad[-1, :] = 0.0
    return grad

@njit
def compute_grad_z(field, dz):
    ny, nz = field.shape
    grad = np.zeros_like(field)
    for j in range(ny):
        for k in range(1, nz - 1):
            grad[j, k] = (field[j, k + 1] - field[j, k - 1]) / (2 * dz)
        grad[j, 0] = 0.0
        grad[j, 1] = 0.0
        grad[j, 2] = 0.0
    return grad

@njit
def compute_dkdx_numba(dk, du, U, V, W, nu_T, lmix, dy, dz, C_k1=1.0, C_k2=1.0):
    ny, nz = dk.shape

    grad_dk_dy = compute_grad_y(dk, dy)
    grad_dk_dz = compute_grad_z(dk, dz)
    grad_du_dy = compute_grad_y(du, dy)
    grad_du_dz = compute_grad_z(du, dz)
    grad_U_dy = compute_grad_y(U, dy)
    grad_U_dz = compute_grad_z(U, dz)

    term1 = grad_du_dz * grad_U_dz
    term2 = compute_grad_z(nu_T * grad_dk_dz, dz)
    for j in range(ny):
        for k in range(3):  # clear first few z-values
            term1[j, k] = 0.0
            term2[j, k] = 0.0

    laplacian_dk_dy = compute_grad_y(nu_T * grad_dk_dy, dy)
    dk_pos = np.maximum(dk, 0.0)
    dk_pow = dk_pos ** 1.5

    dkdx = (-V * grad_dk_dy - W * grad_dk_dz
            + nu_T * (grad_du_dy * grad_U_dy + term1)
            + C_k1 * (laplacian_dk_dy + term2)
            - C_k2 * (dk_pow / lmix)) / U
    return dkdx

@njit
def rk4_step_kwake_numba(dk, du, U, V, W, nu_T, lmix, dy, dz, dx):
    k1 = compute_dkdx_numba(dk, du, U, V, W, nu_T, lmix, dy, dz)
    k2 = compute_dkdx_numba(dk + 0.5 * dx * k1, du, U, V, W, nu_T, lmix, dy, dz)
    k3 = compute_dkdx_numba(dk + 0.5 * dx * k2, du, U, V, W, nu_T, lmix, dy, dz)
    k4 = compute_dkdx_numba(dk + dx * k3, du, U, V, W, nu_T, lmix, dy, dz)
    return dk + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

