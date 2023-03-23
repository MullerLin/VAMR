import numpy as np

def fundamentalEightPoint(p1, p2):
    """ The 8-point algorithm for the estimation of the fundamental matrix F

     The eight-point algorithm for the fundamental matrix with a posteriori
     enforcement of the singularity constraint (det(F)=0).
     Does not include data normalization.

     Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """
    num_P = p1.shape[1]
    A = np.zeros((num_P, 9))
    for i in range(num_P):
        A[i, :] = np.kron(p1[:, i], p2[:, i]).T
    U, sigma, VT = np.linalg.svd(A, full_matrices=True)
    F = VT.T[:, -1].reshape(3, 3)
    U_F, sigma_F, VT_F = np.linalg.svd(F, full_matrices=True)
    sigma_F[2] = 0
    F_mod = U_F @ np.diag(sigma_F) @ VT_F

    return F_mod
