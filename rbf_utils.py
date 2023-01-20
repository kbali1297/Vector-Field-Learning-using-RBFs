import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rbf_model2(t,z, eps, xl, C):
    """
    Model function for solve_ivp.
    Inputs : time t, input field z, epsilon eps, centers xl and Coefficient matrix C
    Returns : Vector field at z : phi(z)@C = F_rbf(z) 
    """
    z = np.tile(np.expand_dims(z, axis=0), [len(xl),1]) 
    phi_z = np.exp(-np.sum((z-xl)**2, axis=1)/(eps**2)).reshape(1,-1)
    
    return (phi_z@C).reshape(-1)

def find_best_params_rbf(L_range, eps_range, X0, X1, F_rbf, dt, verbose=False):
    """
    Here we compute phi for multiple values of L and epsilon in the range specified in the input
    and integrate it to the time step = 0.01.

    Input : L_range : array of L values
            eps_range : array of epsilon values
            X0 : Input vector field to phi
            X1 : Final vector field 
            F_rbf : change in vector field with time 
            dt : Final integration time
    Returns : epsilon, L, C and centers x_l for the best approximated function phi to the vector field
    """
    best_L = L_range[0]
    best_eps = eps_range[0]
    MSE_min = 1e6
    for L in L_range:
        local_best_mse = 1e6
        for eps in eps_range:
            xl = X0[np.random.choice(len(X0[:,0]), size=L, replace=False), :]
            X = X0

            # Reshaping X and x_l matrices as NXLd
            # N : Number of data points 
            # L : Number of centers
            # d : Input data point dimensions (here d=2)

            X = np.tile(X, [1,L])

            x_l2 = xl.reshape(1, -1)
            x_l2 = np.tile(x_l2, [len(X), 1])
            
            diff = (X-x_l2)**2
            #print(f"diff: {diff}")

            # Here we Compute the sum across axis 1 to reduce the shape of
            # diff from NXLd to NXL by summing over d columns and 
            # concatenating them into a new array of shape NXL
            diff_temp = diff[:, 0] + diff[:, 1]
            diff_new = np.expand_dims(diff_temp, axis=1) 
            for i in range(2, len(diff[0,:]), 2):
                diff_temp = diff[:, i] + diff[:, i+1]
                diff_new = np.concatenate([diff_new, np.expand_dims(diff_temp, axis=1)], axis=1)
            
            # Input to the phi function the new array of the shape NXL
            phi_X = np.exp( -diff_new/eps**2)


            # Computing Least Squares Solution Coefficient matrix C
            # C = np.linalg.pinv(phi_X.T@phi_X)@phi_X.T@F_rbf
            C = np.linalg.lstsq(phi_X, F_rbf, rcond=None)[0]

            sol_list2 = []
            time = np.linspace(0, dt, 100)
            for i in range(len(X0)):
                sol = solve_ivp(rbf_model2, t_span=[0,dt], y0=X0[i,:].T,args=(eps, xl, C, ) ,t_eval=time)
                sol_list2.append(sol.y[:,-1])

            sol_array2 = np.array(sol_list2)
            MSE = np.sum((sol_array2 - X1)**2)/len(X1)
            
            if MSE<MSE_min:
                MSE_min = MSE
                best_eps = eps
                best_L = L
                best_C = C
                centers = xl
            
            if MSE<local_best_mse:
                local_best_mse = MSE
        if verbose:
            print(f"Present Config L={L} MSE: {local_best_mse:.6f}, Min MSE:{MSE_min:.6f}, Min eps:{best_eps:.3f}, Min L:{best_L}")
    
    return best_eps, best_L, best_C, centers

def plot_rbf_streamplot(xl, eps, C):
    """
    Plots streamlines for the vector field represented by a radial basis function 
    with centers xl and epsilon eps for all points in the domain
    Inputs : Centers xl, Epsilon eps
    Returns : Streamline plot given by radial basis function and C matrix
    """
    x,y = np.meshgrid(np.linspace(-4.5, 4.5, 100), np.linspace(-4.5, 4.5, 100))
    x_ = np.expand_dims(x, axis=2)
    y_ = np.expand_dims(y, axis=2)
    X_vec = np.concatenate([x_, y_],axis=2) 
    num_points_x = X_vec.shape[0]
    num_points_y = X_vec.shape[1]
    X_vec = X_vec.reshape(num_points_x*num_points_y,-1)
    L = len(xl)
    Z = np.tile(X_vec, [L,1])
    X_l_ = np.repeat(xl, num_points_x*num_points_y, axis=0)
    Phi_z =  np.reshape(np.exp(-np.sum((Z-X_l_)**2, axis=1)/(eps**2)), (num_points_x * num_points_y, -1), order='F')  
    # print(Phi_z.shape)
    # print(Z)
    # print(X_l_)
    non_linear_field = Phi_z@C
    non_linear_field = np.reshape(non_linear_field, (num_points_x, num_points_y, 2), order='C')
    #non_linear_field.reshape(X_vec.shape[0], X_vec.shape[1], 2)
    fig1, ax1 = plt.subplots()
    ax1.streamplot(x, y, non_linear_field[:,:,0], non_linear_field[:,:,1])
    plt.show()