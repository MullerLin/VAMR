import numpy as np


def poseVectorToTransformationMatrix(pose_vec):
    """Converts a 6x1 pose vector into a 4x4 transformation matrix"""
    omega = pose_vec[:3]
    t = pose_vec[3:]

    theta = np.sqrt((omega**2).sum())
    k = omega / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T
