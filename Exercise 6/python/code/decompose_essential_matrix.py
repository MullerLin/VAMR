import numpy as np


def decomposeEssentialMatrix(E):
    """ Given an essential matrix, compute the camera motion, i.e.,  R and T such
     that E ~ T_x R
     
     Input:
       - E(3,3) : Essential matrix

     Output:
       - R(3,3,2) : the two possible rotations
       - u3(3,1)   : a vector with the translation information
    """
    R = np.zeros([3, 3, 2])
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    U, sigma, VT = np.linalg.svd(E, full_matrices=True)

    if (np.linalg.det(U @ W @ VT) > 0):
        R[:, :, 0] = U @ W @ VT
    else:
        R[:, :, 0] = - U @ W @ VT

    if (np.linalg.det(U @ W.T @ VT) > 0):
        R[:, :, 1] = U @ W.T @ VT
    else:
        R[:, :, 1] = - U @ W.T @ VT

    u3 = U[:, 2]

    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    return R, u3
