import numpy as np

def read_file(path):
    """
    Reads the file and returns the numpy arrays for creating the delay embedding.
    embed_array contains the first 3 PID columns of MI_timesteps.txt and full_embed_array contains all of them.
    """
    f = open(path)

    embed_mat = []
    full_mat = []
    for i, line in enumerate(f):
        if i>=1000:
            words = line.split(" ")
            if str.isnumeric(line[0]): 
                embed_mat.append(words[1:4])
                full_mat.append(words[1:])

    embed_array = np.array(embed_mat, dtype=float)
    full_embed_array = np.array(full_mat, dtype=float)

    return embed_array, full_embed_array

def compute_best_rbf_params(L_range, eps_range, cum_arc_len_array, F, verbose=False):
    """
    Computes the best set of rbf parameters and coefficient matrix
    as long as there is a substantial reduction in mean squared error 
    between the predicted and actual utilization value

    input: L_range: range of L values to try
           eps_range: range of epsilon values to try
           cum_arc_len_array: array comprising of cumulative arc length values at every time step of the PCA curve
           F: F_utilization array

    output: Number of basis functions, epsilon, xl centers of basis functions, C Coefficient matrix, phi_X rbf functions  
    """
    #L = 200
    X = np.array(cum_arc_len_array)
    min_MSE = 10e6
    best_L = L_range[0]
    best_eps = eps_range[0]
    for L in L_range:
        best_current_mse = 10e6
        for eps in eps_range:

            #X = np.array(cum_arc_len_array)
            xl = np.linspace(np.min(X), np.max(X), L)

            X_ = np.expand_dims(X, axis=1).repeat(len(xl), axis=1)
            xl = np.expand_dims(xl, axis=0).repeat(len(X), axis=0)

            #epsilon = 0.005*np.max(X - xl)

            phi_X = np.exp(-((X_ - xl)/eps)**2)

            C = np.linalg.lstsq(phi_X, F, rcond=None)[0]

            current_MSE = np.sum((phi_X@C - F)**2)/len(F)
            
            if current_MSE < best_current_mse:
                best_current_mse = current_MSE
            
            if abs(current_MSE - min_MSE)<0.01: return L, eps, xl, C, phi_X
            elif current_MSE < min_MSE:
                min_MSE = current_MSE
                best_L = L
                best_eps = eps
                best_xl = xl
                best_C = C
                best_phi_X = phi_X
                  
        if verbose:
            print(f"Current config L:{L}, Best Current MSE: {best_current_mse:.6f}, Best MSE : {min_MSE:.6f}, best_L: {best_L}, best_eps:{best_eps}")

    return best_L, best_eps, best_xl, best_C, best_phi_X

            

            