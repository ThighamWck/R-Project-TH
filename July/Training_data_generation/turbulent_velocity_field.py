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

    # Apply projection to make divergence-free
    ux_projected, uy_projected = P(ux, uy)

        # ===========================
    # RESCALE TO THEORETICAL SPECTRUM
    # ===========================
    def rescale_to_theoretical_spectrum(N_x, N_y, ux, uy, kx, ky):
        """
        Rescale the Fourier coefficients so that the energy spectrum follows k^(-5/3) scaling.
        Uses a more robust approach based on wavenumber magnitude.
        """
        k_mag = np.sqrt(kx**2 + ky**2)
        
        # Avoid division by zero at k=0
        k_safe = k_mag.copy()
        k_safe[k_safe == 0] = 1.0
        
        # Create theoretical scaling: amplitude should scale as k^(-4/3) for E(k) ~ k^(-5/3)
        # But we need to be more conservative to avoid overshoot
        theoretical_scaling = np.ones_like(k_mag)
        
        # Only scale the active modes (avoid DC and very high frequencies)
        kfmin = 2 * np.pi / L_domain
        kfmax = min(np.pi * N_x / L_domain / 2, 20 * kfmin)
        
        active_mask = (k_mag >= kfmin) & (k_mag <= kfmax) & (k_mag > 0)
        
        # Apply theoretical scaling only to active modes
        theoretical_scaling[active_mask] = (k_mag[active_mask] / kfmin)**(-4/3)
        
        # Get current scaling from the field
        current_energy = 0.5 * (np.abs(ux)**2 + np.abs(uy)**2)
        current_amplitude = np.sqrt(current_energy)
        
        # Avoid division by zero
        current_amplitude[current_amplitude == 0] = 1.0
        
        # Calculate scaling factor more conservatively
        # Use a smooth transition and limit the maximum scaling factor
        scale_factor = np.ones_like(k_mag)
        
        # Only apply scaling to active modes
        if np.any(active_mask):
            # Normalize theoretical scaling to match current field's overall amplitude
            theory_norm = np.mean(theoretical_scaling[active_mask])
            current_norm = np.mean(current_amplitude[active_mask])
            
            if theory_norm > 0 and current_norm > 0:
                normalization = current_norm / theory_norm
                theoretical_scaling *= normalization
                
                # Apply scaling with a maximum limit to prevent overshoot
                raw_scaling = theoretical_scaling / current_amplitude
                max_scale = 5.0  # Limit maximum scaling factor
                scale_factor = np.clip(raw_scaling, 1.0/max_scale, max_scale)
                
                # Only apply to active modes
                scale_factor[~active_mask] = 1.0
        
        # Apply scaling
        ux_scaled = ux * scale_factor
        uy_scaled = uy * scale_factor
        
        return ux_scaled, uy_scaled
    # Apply rescaling to match theoretical spectrum
    ux_scaled, uy_scaled = rescale_to_theoretical_spectrum(N_x, N_y, ux_projected.copy(), uy_projected.copy(), kx, ky)

    # Optionally re-project to ensure divergence-free after scaling
    ux_final, uy_final = P(ux_scaled, uy_scaled)
    

    # Transform back to physical space and ensure real 
    ux_physical = np.real(ifft2(ux_final))
    uy_physical = np.real(ifft2(uy_final))

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
    x, y = np.linspace(L_left - 1/L_domain, L_right + 1/L_domain, N_x, endpoint=True), np.linspace(0, 1, N_y, endpoint=True)

    # Create interpolators for each component of the vector field
    interp_ux = RegularGridInterpolator((x, y), ux_physical, method='cubic')
    interp_uy = RegularGridInterpolator((x, y), uy_physical, method='cubic')
    




    return interp_ux, interp_uy
