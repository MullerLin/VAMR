import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    # YOUR CODE GOES HERE! 
    # THE COMMENTS ARE MEANT TO HELP YOU
    
    # Convert 2D to normalized coordinates
    pts_2d_aug = np.c_[p, np.ones((p.shape[0], 1))]
    p_cam = (np.mat(K).I * pts_2d_aug.T).T[:, 0:2]

    # Build measurement matrix Q
    num_corners = p.shape[0]
    p_W_mat_h = np.hstack((P, np.mat(np.ones((num_corners, 1)))))

    Q = np.mat(np.zeros((num_corners * 2, 12)))
    for i in range(num_corners):
        Q[2 * i, 0:4] = p_W_mat_h[i]
        Q[2 * i, 8:12] = p_W_mat_h[i] * (-1) * p_cam[i, 0]
        Q[2 * i + 1, 4:8] = p_W_mat_h[i]
        Q[2 * i + 1, 8:12] = p_W_mat_h[i] * (-1) * p_cam[i, 1]

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1

    # Extract [R | t] with the correct scale

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix

    # Build M_tilde with the corrected rotation and scale and return it

    U, sigma, VT = np.linalg.svd(Q, full_matrices=True)
    M = VT.T[:, -1].reshape(3, 4)
    if (np.linalg.det(M[:, :3]) < 0):
        M *= -1
    R = M[0:3, 0:3]
    U_R, Sigma_R, VT_R = np.linalg.svd(R)
    R_ = U_R * VT_R
    alpha = np.linalg.norm(R_) / np.linalg.norm(R)
    M_ = np.c_[R_, alpha * M[:, 3]]

    return M_
    
     
