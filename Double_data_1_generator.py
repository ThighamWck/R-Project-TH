import numpy as np
import pandas as pd
import autograd.numpy as anp
from Finite_difference_forward_comittor import solve_forward_committor_2D
from Finite_difference_backwards_committor import solve_backwards_committor_2D
from Trig_polynomial_boundary import generate_trig_functions
from turbulent_velocity_field import turbulent_velocity_field

def generate_training_1(num_solutions):
    """
    This code generates training data for my committor function problem, on square domain [0,1]^2, to be used in a FNO.
    """
    #Setting up data list
    data_list_32 = []
    data_list_63 = []

    #random seed to make training data reproducible
    np.random.seed(220325)  #training data
    #np.random.seed(2203252)  #test data

    for solution_id in range(num_solutions):
        #progress tracking
        print(f"Generating solution {solution_id+1}/{num_solutions}")
        
        ##################################################
        #Identity coordinate transform. No special boundaries.
        phi = lambda y: 0.0  # Left boundary
        psi = lambda y: 1.0  # Right boundary

        eps_1 = 0
        eps_2 = 0        
        ##################################################
        
                
        #generate random vector field b(x,y) and define btilde(f(x,y),y) = b(f^{-1}(x,y),y) so in correct coords
        b_x , b_y = turbulent_velocity_field(eps_1, eps_2, Reynolds = 10, L = 0.2, nu = 1)
        def btilde_x(x,y):
            # Reshape meshgrid points into format needed by RegularGridInterpretor
            points = anp.vstack(( x.flatten(), y.flatten())).T
            return b_x(points).reshape(x.shape)
        def btilde_y(x,y):
             # Reshape meshgrid points into format needed by RegularGridInterpretor
            points = anp.vstack(( x.flatten(), y.flatten())).T
            return b_y(points).reshape(x.shape)
        
        # *10 toDefine the 311x311 grid - easy to identify with a coarser 32 x 32 grid when training.
        # *5 to define 311 x311 for 63x63 grid.
        #n_fine = (grid_size-1)*5 + 1

        #Solving for forward committor
        qplus, X1, Y1 = solve_forward_committor_2D(phi, psi, btilde_x, btilde_y, N= (309)) #Nx and Ny count num of interior points
        #Solving for backward committor
        qminus, X2, Y2 = solve_backwards_committor_2D(phi, psi, btilde_x, btilde_y, N = (309))
        #Committor function
        rho = qplus * qminus
      # Create a meshgrid of coarse indices 
        coarse_indices_32 = np.linspace(0, 310, 32, dtype=int)
        coarse_indices_32_i, coarse_indices_32_j = np.meshgrid(coarse_indices_32, coarse_indices_32)

        # Extract the coarse grid from X and Y
        coarse_grid_32_x = X1[coarse_indices_32_i, coarse_indices_32_j]
        coarse_grid_32_y = Y1[coarse_indices_32_i, coarse_indices_32_j]

        #Storing data in dataframe. We only want grid points at resolution for training..
        for i in range(32):
            for j in range(32):
                x_ij = coarse_grid_32_x[i,j]
                y_ij = coarse_grid_32_y[i,j]
                
                # need to evaluate at finv(x,y) as identity transform.
                b1, b2 = btilde_x(x_ij, y_ij) , btilde_y(x_ij, y_ij)
               

                # Get the u value at the corresponding indices in the original array
                rho_val = rho[coarse_indices_32_i[i,j], coarse_indices_32_j[i,j]]
                #qplus_val = qplus[coarse_indices_i[i,j], coarse_indices_j[i,j]]
                data_list_32.append([solution_id, x_ij, y_ij, b1, b2, rho_val])

        coarse_indices_63 = np.linspace(0, 310, 63, dtype=int)
        coarse_indices_63_i, coarse_indices_63_j = np.meshgrid(coarse_indices_63, coarse_indices_63)

        # Extract the coarse grid from X and Y
        coarse_grid_63_x = X2[coarse_indices_63_i, coarse_indices_63_j]
        coarse_grid_63_y = Y2[coarse_indices_63_i, coarse_indices_63_j]



        for i in range(63):
            for j in range(63):
                x_ij = coarse_grid_63_x[i,j]
                y_ij = coarse_grid_63_y[i,j]
                
                # need to evaluate at finv(x,y) as identity transform.
                b1, b2 = btilde_x(x_ij, y_ij) , btilde_y(x_ij, y_ij)
               

                # Get the u value at the corresponding indices in the original array
                rho_val = rho[coarse_indices_63_i[i,j], coarse_indices_63_j[i,j]]
                #qplus_val = qplus[coarse_indices_i[i,j], coarse_indices_j[i,j]]
                data_list_63.append([solution_id, x_ij, y_ij, b1, b2, rho_val])

        # for i in range(n_fine):
        #     for j in range(n_fine):
        #         x_ij = X1[i,j]
        #         y_ij = Y1[i,j]
                
        #         # need to evaluate at finv(x,y) as identity transform.
        #         b1, b2 = btilde_x(x_ij, y_ij) , btilde_y(x_ij, y_ij)
               

        #         # Get the u value at the corresponding indices in the original array
        #         rho_val = rho[i,j]
        #         #qplus_val = qplus[coarse_indices_i[i,j], coarse_indices_j[i,j]]
        #         data_list.append([solution_id, x_ij, y_ij, b1, b2, rho_val])
    # Create DataFrame
    df_1 = pd.DataFrame(data_list_32, columns=["solution_id", "x", "y", "b1", "b2", "rho"])
    df_2 = pd.DataFrame(data_list_63, columns=["solution_id", "x", "y", "b1", "b2", "rho"])

    # Save to CSV
    df_1.to_csv("train_32_square_1000.csv", index=False)
    df_2.to_csv("train_63_square_1000.csv", index=False)

    return