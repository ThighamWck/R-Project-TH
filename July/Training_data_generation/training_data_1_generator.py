
"""
#################### 
# Square domain training data generator - for Fourth Year Project
# Author - Thomas Higham
# Date - 31/03/25
# Updated - 09/06/25 for extended work on project
# University of Warwick
#####################
Initialisations is in main function at bottom
# """

import numpy as np
import pandas as pd
import autograd.numpy as anp
from Finite_difference_forward_comittor import solve_forward_committor_2D
from Finite_difference_backwards_committor import solve_backwards_committor_2D
from turbulent_velocity_field import turbulent_velocity_field
import json
import os

def generate_training_1():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config_training.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    training_cfg = config["training"]
    output_cfg = config["output"]
    grid_size = training_cfg["grid_size"]
    num_solutions = training_cfg["data_size"]
    save_path = output_cfg["data_save_path"]
    """
    This code generates training data for my committor function problem, on square domain [0,1]^2, to be used in a FNO.
    """
    #Setting up data list
    data_list = []
    #random seed to make training data reproducible
    np.random.seed(220325)  #training data
    #np.random.seed(2203252)  #test data
    
    np.random.seed(100625) #training data 2

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
        b_x , b_y = turbulent_velocity_field(eps_1, eps_2, Reynolds = 10000, L = 0.002, nu = 0.00001)
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
        n_fine = (grid_size-1)*8 + 1

        #Solving for forward committor
        qplus, X1, Y1 = solve_forward_committor_2D(phi, psi, btilde_x, btilde_y, N= (n_fine-2)) #Nx and Ny count num of interior points
        #Solving for backward committor
        qminus, _, _ = solve_backwards_committor_2D(phi, psi, btilde_x, btilde_y, N = (n_fine-2))
        #Committor function
        rho = qplus * qminus
      # Create a meshgrid of coarse indices 
        coarse_indices = np.linspace(0, n_fine - 1, grid_size, dtype=int)
        coarse_indices_i, coarse_indices_j = np.meshgrid(coarse_indices, coarse_indices)

        # Extract the coarse grid from X and Y
        coarse_grid_x = X1[coarse_indices_i, coarse_indices_j]
        coarse_grid_y = Y1[coarse_indices_i, coarse_indices_j]

        #Storing data in dataframe. We only want grid points at resolution for training..
        for i in range(grid_size):
            for j in range(grid_size):
                x_ij = coarse_grid_x[i,j]
                y_ij = coarse_grid_y[i,j]
                
                # need to evaluate at finv(x,y) as identity transform.
                b1, b2 = btilde_x(x_ij, y_ij) , btilde_y(x_ij, y_ij)
               

                # Get the u value at the corresponding indices in the original array
                rho_val = rho[coarse_indices_i[i,j], coarse_indices_j[i,j]]
                #qplus_val = qplus[coarse_indices_i[i,j], coarse_indices_j[i,j]]
                data_list.append([solution_id, x_ij, y_ij, b1, b2, rho_val])

    # # Create DataFrame
    # df = pd.DataFrame(data_list, columns=["solution_id", "x", "y", "b1", "b2", "rho"])

    # # Save to CSV
    # df.to_csv("5000_resolution_249_train_120625_square.csv", index=False)

    array_data = np.array(data_list)
    columns = ["solution_id", "x_ij", "y_ij", "b1", "b2", "rho_val"]

    np.savez_compressed(save_path, 
                    data=array_data, columns=columns)

    return

if __name__ == "__main__":
    generate_training_1()