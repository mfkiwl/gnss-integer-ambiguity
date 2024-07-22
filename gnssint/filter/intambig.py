import numpy as np
from scipy.linalg import ldl

def floatrem(floatarray):
    """
    Returns decimal and integer parts of float.
    """

    fltarray = np.array(floatarray)
    intarray = fltarray.astype(int)
    decarray = fltarray - intarray

    return decarray, intarray

def covfloat_as_ldl(cov_float):
    """
    Decomposes the covariance of a float solution as LDL' with
    L lower-triangular and D a (2D-block) diagonal.

    Parameters:
    -----------
    cov_float: array (n_state, n_state)
    Covariance matrix of the float solution.

    Returns:
    --------
    L, D
    """

    L, D, perm = ldl(A=cov_float, lower=True, overwrite_a=False)

    return L, D, perm

def covfloat_to_covdecorr(cov_float, Z):
    """
    Returns covariance of a corrected (ie decorrelated) float solution
    based on a deccorelation matrix Z and (float solution) covariance.

    Parameters:
    -----------
    cov_float: array (n_state, n_state)
    Covariance matrix of the float solution.

    Z: array (n_state, n_state)
    Deccorelation matrix for the float solution.

    Returns:
    --------
    cov_float: array (n_state, n_state)
    Covariance matrix of the float solution.
    """

    return np.matmul(Z.T, np.matmul(cov_float, Z))

# def ldl_covfloat()
