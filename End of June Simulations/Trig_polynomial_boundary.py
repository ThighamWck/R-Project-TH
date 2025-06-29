
"""
#################### 
# Generate domain boundaries - for Fourth Year Project
# Author - Thomas Higham
# Date - 31/03/25
# University of Warwick
#####################
# """


from autograd import numpy as anp

def generate_trig_functions(Print, num_terms_range=(1, 10)):
    """ Code to generate a scaled random trig polynomial made up of sin functions, parametrised on y in [0,1]. phi(0) = phi(1)=0 and psi(0) = psi(1) = 1.
    para:
         num_terms_range - fixed range for order of polynomials. Ensures the polynomials are not too crazy.
    para:
        Print - Boolean. Can be used in testing to see what the trigonometric polynomial looks like.
     """

    def generate_single_function(N):
        # Use trackable autograd arrays
        coeffs = anp.random.uniform(-1, 1, N)
        return coeffs, lambda y: anp.sum(coeffs[:, None] * anp.sin(anp.arange(1, N+1)[:, None] * anp.pi * y), axis=0)

    N1, N2 = anp.random.randint(*num_terms_range, size=2)
    #print("N1 is", N1, "N2 is", N2)
    coeffs_phi, phi = generate_single_function(N1)
    coeffs_psi, psi = generate_single_function(N2)

    # Keep all operations in autograd numpy
    y_vals = anp.linspace(0, 1, 200)
    max_phi = 2*anp.max(anp.abs(phi(y_vals))) + anp.mean(anp.abs(phi(y_vals)))
    max_psi = 2*anp.max(anp.abs(psi(y_vals))) + anp.mean(anp.abs(psi(y_vals)))

    # Calculate areas analytically using integral of sin(nx):
    # int_0^1 sin(nπy)dy = -[cos(nπy)/(nπ)]_0^1 = -(cos(nπ) - 1)/(nπ)
    area_phi = -anp.sum(coeffs_phi * (anp.cos(anp.arange(1,N1+1)*anp.pi) - 1) / (anp.arange(1,N1+1)*anp.pi)) / max_phi
    area_psi = -anp.sum(coeffs_psi * (anp.cos(anp.arange(1,N2+1)*anp.pi) - 1) / (anp.arange(1,N2+1)*anp.pi)) / max_psi

    # Total area is unit square (1) plus area from phi boundary plus area from psi boundary
    Omega = 1 + area_phi + area_psi
    if Print == True:
        def build_symbolic(coeffs, max_val):
            return ' + '.join(f"{c/max_val:.3f} * anp.sin({i+1} * anp.pi * y)" for i, c in enumerate(coeffs))

        phi_symbolic = build_symbolic(coeffs_phi, max_phi)
        psi_symbolic = build_symbolic(coeffs_psi, max_psi)

        print(f"φ(y) = {phi_symbolic}")
        print(f"ψ(y) = ({psi_symbolic}) + anp.ones_like(y)")

    return (lambda y: phi(y) / max_phi, 
            lambda y: psi(y) / max_psi + anp.ones_like(y), Omega)
