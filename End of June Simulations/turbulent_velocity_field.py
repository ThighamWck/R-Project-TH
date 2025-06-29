"""
#################### 
# Turbulent velocity field - for Fourth Year Project
# Author - Thomas Higham and Tobias Grafke. Adapted from incompressible Navier Stokes code by Tobias.
# Date - 25/03/25
# University of Warwick
#####################
# """

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator

def validate_resolution(N_x, N_y, Reynolds):
    """
    Validate if grid resolution is sufficient for given Reynolds number
    """
    N_min = int(np.ceil(Reynolds**(3/4)))
    if N_x < N_min or N_y < N_min:
        print(f"Warning: Grid resolution {N_x}x{N_y} may be insufficient for Re={Reynolds}")
        print(f"Recommended minimum resolution: {N_min}x{N_min}")
        return False
    return True

def turbulent_velocity_field(eps_1, eps_2, Reynolds, L, nu):
    """
    Generate a turbulent velocity field scaled for a specific Reynolds number
    
    Parameters:
    eps_1, eps_2 - domain parameters that define rectangular epsilon neighbourhood of Omega
    Reynolds - Reynolds number of the flow
    L - characteristic length scale
    nu - kinematic viscosity
    """
    # Validate resolution
    N_x, N_y = 1000, 1000
    # if not validate_resolution(N_x, N_y, Reynolds):
    #     N_x = N_y = int(np.ceil(Reynolds**(3/4)))
    #     print(f"Adjusting resolution to {N_x}x{N_y}")
    
    # ===========================
    # SET UP FOURIER SPACE
    # ===========================
    # Store characterisitc length L for Reynolds number calculation
    L_char = L
    
    ### This is new code trying to fit turbulence to domain better

    if eps_1 < 0:
        L_left = eps_1
    else:
        L_left = 0
    if eps_2 > 0:
        L_right = 1+ eps_2
    else:
        L_right = 1

    # Domain size for spectral method
    L_domain = L_right - L_left

    # Define grid and spacing in real space
    x, y = np.linspace(L_left, L_right, N_x, endpoint=False), np.linspace(0, 1, N_y, endpoint=False)
    dx, dy = L_domain / N_x, L_domain / N_y

    # Define grid and spacing in Fourier space
    k_xv = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)
    k_yv = 2 * np.pi * np.fft.fftfreq(N_y, d=dy)
    kx, ky = np.meshgrid(k_xv, k_yv, indexing='ij')
    k2 = kx**2 + ky**2  # Squared wavenumber magnitude

    # ===========================
    # GENERATE HERMITIAN SYMMETRIC MODES
    # ===========================
    def generate_hermitian_random_field(shape, active_modes, amplitude, scaling):
        """
        Generate a random field with proper Hermitian symmetry for real IFFT.
        Only generates the independent modes and sets their conjugates.
        """
        N_x, N_y = shape
        field = np.zeros((N_x, N_y), dtype=complex)
        
        for i in range(N_x):
            for j in range(N_y):
                if not active_modes[i, j]:
                    continue
                
                # Find conjugate indices
                i_conj = (-i) % N_x
                j_conj = (-j) % N_y
                
                # Handle special cases that must be real
                if (i == 0 and j == 0) or \
                   (N_x % 2 == 0 and i == N_x//2 and j == 0) or \
                   (N_y % 2 == 0 and i == 0 and j == N_y//2) or \
                   (N_x % 2 == 0 and N_y % 2 == 0 and i == N_x//2 and j == N_y//2):
                    # These modes must be real
                    field[i, j] = amplitude * scaling[i, j] * np.random.randn()
                else:
                    # Check if we've already set this mode via its conjugate
                    if field[i, j] != 0:
                        continue
                    
                    # Generate complex random coefficient
                    coeff = amplitude * scaling[i, j] * (np.random.randn() + 1j * np.random.randn())
                    field[i, j] = coeff
                    
                    # Set conjugate symmetric coefficient
                    field[i_conj, j_conj] = np.conj(coeff)
        
        return field

    # ===========================
    # PROJECT TO DIVERGENCE-FREE SPACE
    # ===========================
    # Projection operator to ensure the velocity field is divergence-free
    def P(ux, uy):
        """Projection operator to ensure the velocity field is divergence-free"""
        # Avoid division by zero at k=0
        k2_safe = k2.copy()
        k2_safe[0, 0] = 1.0
    
        # Compute divergence in Fourier space
        uxprojected = ux * (1 - kx**2/k2_safe) - uy * kx*ky/k2_safe
        uyprojected = uy * (1 - ky**2/k2_safe) - ux * kx*ky/k2_safe
    
        # Set DC component to zero (no mean flow)
        uxprojected[0, 0] = 0
        uyprojected[0, 0] = 0

        return uxprojected, uyprojected
    # ===========================
    # INITIALIZE VELOCITY FIELD
    # ===========================
    # Define wavenumber and Nyquist limit
    k = np.sqrt(k2)
    k_nyquist = np.pi * N_x / L_domain
    
    # Select modes based on physical scales
    kfmin = 2 * np.pi / L_domain  # Largest scale ~ domain size
    kfmax = min(k_nyquist / 2, 20 * kfmin)  # Limit by resolution and reasonable range
    
    # Use logarithmic spacing for mode selection
    log_k = np.log10(k + 1e-10)  # Avoid log(0) errors
    log_kfmin = np.log10(kfmin)
    log_kfmax = np.log10(kfmax)
    
    # Select modes in logarithmic bands
    initModes = (log_k >= log_kfmin) & (log_k <= log_kfmax)
    
    n_active = np.sum(initModes)
    # print(f"Active modes: {n_active}")
    # print(f"Wavenumber range: {kfmin:.2f} to {kfmax:.2f}")
    # print(f"Number of modes in each dimension: {int(np.sqrt(n_active))}")
    
    # if n_active < 10:
    #     raise ValueError(f"Too few modes ({n_active}). Try increasing resolution.")
    
    # Create random Fourier modes with larger initial amplitude
    ux = np.zeros((N_x, N_y), dtype=complex)
    uy = np.zeros((N_x, N_y), dtype=complex)
    
    # Create random Fourier modes with proper Hermitian symmetry
    ux = np.zeros((N_x, N_y), dtype=complex)
    uy = np.zeros((N_x, N_y), dtype=complex)

    # Generate random coefficients with proper normalization
    amplitude = 1e2  # Reduced from 1e4 for better numerical stability

    #Apply energy spectrum scaling (Kolmogorov-like for 2D)
    scaling = np.ones_like(k)
    mask = (k > 0) & initModes
    scaling[mask] = k[mask]**(-4/3)  # For E(k) ~ k^(-5/3): amplitude ~ k^(-4/3)

    # Generate fields and check energy before projection
    ux = generate_hermitian_random_field((N_x, N_y), initModes, amplitude, scaling)
    uy = generate_hermitian_random_field((N_x, N_y), initModes, amplitude, scaling)

    # Apply projection
    ux, uy = P(ux, uy)
    
    # Apply scaling to Fourier coefficients
    ux *= scaling
    uy *= scaling


    # Transform back to physical space and ensure real 
    ux_physical = np.real(ifft2(ux))
    uy_physical = np.real(ifft2(uy))

    # ===========================
    # NORMALIZE TO TARGET RMS
    # ===========================
    #rms_target = np.sqrt(Reynolds * nu / L_char)  # Corrected RMS velocity target
    rms_target = (Reynolds * nu) / L_char

    current_rms = np.sqrt(np.mean(ux_physical**2 + uy_physical**2))
    
    # if current_rms < 1e-10:
    #     raise ValueError(f"RMS velocity too small ({current_rms}). Try increasing initial amplitude.")
    # Scale velocity field to match required Reynolds number
    scaling_factor = rms_target / current_rms
    ux_physical *= scaling_factor
    uy_physical *= scaling_factor   
    # Verify final Reynolds number
    final_rms = np.sqrt(np.mean(ux_physical**2 + uy_physical**2))
    actual_reynolds = (final_rms * L_char) / nu
    # print(f"Target Reynolds number: {Reynolds}")
    # print(f"Actual Reynolds number: {actual_reynolds:.4f}")

    # # ===========================
    # # VISUALIZE RESULTS
    # # ===========================
    # plt.figure(figsize=(12, 10))

    # # Velocity magnitude
    # velocity_mag = np.sqrt(ux_physical**2 + uy_physical**2)
    # plt.subplot(2, 2, 1)
    # plt.pcolormesh(x, y, velocity_mag, cmap='viridis', shading='auto')
    # plt.colorbar(label='Velocity Magnitude')
    # plt.title('Velocity Magnitude')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.savefig("turbulence.png", dpi=300, bbox_inches="tight")


    # ===========================
    # INTERPOLATOR SETUP
    # ===========================
    x, y = np.linspace(L_left, L_right, N_x, endpoint=True), np.linspace(0, 1, N_y, endpoint=True)

    # Create interpolators for each component of the vector field
    interp_ux = RegularGridInterpolator((x, y), ux_physical, method='cubic')
    interp_uy = RegularGridInterpolator((x, y), uy_physical, method='cubic')
    




    return interp_ux, interp_uy
