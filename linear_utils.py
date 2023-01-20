import numpy as np
import matplotlib.pyplot as plt

def linear_model(t, z, A):
    """
    Model function for linear solve_ivp.
    Inputs : time t, input field z, estimation matrix A
    Returns : Vector field at z : A@z = F_linear(z) 
    """
    return A@z


def plot_linear_streamplot(A):
    """
    Plots streamlines for the vector field represented by estimation matrix A 
    for all points in the domain
    Inputs : Estimation Matrix A
    Returns : Streamline plot
    """
    x,y = np.meshgrid(np.linspace(-4.5, 4.5, 100), np.linspace(-4.5, 4.5, 100))
    x_ = np.expand_dims(x, axis=2)
    y_ = np.expand_dims(y, axis=2)
    X_vec = np.concatenate([x_, y_],axis=2) 
    linear_field = X_vec@A.T
    fig, ax = plt.subplots()
    ax.streamplot(x,y,linear_field[:,:,0], linear_field[:,:,1], density=1)
    plt.show()

