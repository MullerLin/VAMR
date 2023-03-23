import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.transform import Rotation

from estimate_pose_dlt import estimatePoseDLT
from reproject_points import reprojectPoints
from draw_camera import drawCamera
from plot_trajectory_3D import plotTrajectory3D

def main():
    # Load 
    #    - an undistorted image
    #    - the camera matrix
    #    - detected corners
    image_idx = 1
    undist_img_path = "../data/images_undistorted/img_%04d.jpg" % image_idx
    undist_img = cv2.imread(undist_img_path, cv2.IMREAD_GRAYSCALE)

    K = np.loadtxt("../data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("../data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0]

    # YOUR CODE GOES HERE!

    # Load the 2D projected points that have been detected on the
    # undistorted image into an array
    pts_2d = np.reshape(
            np.loadtxt("../data/detected_corners.txt")[image_idx-1,:],
            (-1, 2) )
    # Now that we have the 2D <-> 3D correspondances let's find the camera pose
    # with respect to the world using the DLT algorithm

    M_ = estimatePoseDLT(pts_2d, p_W_corners, K)

    # Plot the original 2D points and the reprojected points on the image
    p_reproj = reprojectPoints(p_W_corners, M_, K)


    plt.figure()
    plt.imshow(undist_img, cmap = "gray")
    # plot the detected and reprojected points

    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], marker='o')
    plt.scatter(np.array(p_reproj[:, 0]), np.array(p_reproj[:, 1]), marker='+')

    # Make a 3D plot containing the corner positions and a visualization
    # of the camera axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_W_corners[:,0], p_W_corners[:,1], p_W_corners[:,2])

    # use ax.scatter(X, Y, Z) for plotting

    # Position of the camera given in the world frame from the DLT algorithm
    M_tilde_dst = np.array(M_)
    R_C_W = M_tilde_dst[:3,:3];
    t_C_W = M_tilde_dst[:3,3];
    rotMat = R_C_W.T;
    pos = -R_C_W.T @ t_C_W;

    # uncomment this line to draw a camera
    drawCamera(ax, pos, rotMat, length_scale = 0.1, head_size = 10)
    plt.show()


def main_video():
    K = np.loadtxt("../data/K.txt")
    p_W_corners = np.loadtxt("../data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0]

    all_pts_2d = np.loadtxt("../data/detected_corners.txt")
    num_images = all_pts_2d.shape[0]
    translations = np.zeros((num_images, 3))
    quaternions = np.zeros((num_images, 4))
    
    # YOUR CODE GOES HERE

    for idx in range(num_images):
        pts_2d = np.reshape(all_pts_2d[idx, :], (-1, 2))
        M_ = estimatePoseDLT(pts_2d, p_W_corners, K)
        M_tilde_dst = np.array(M_)
        R_C_W = M_tilde_dst[:3, :3];
        t_C_W = M_tilde_dst[:3, 3];
        quaternions[idx, :] = Rotation.from_matrix(R_C_W.T).as_quat()
        translations[idx, :] = -R_C_W.T @ t_C_W;

    fps = 30
    filename = "../motion.avi"
    plotTrajectory3D(fps, filename, translations, quaternions, p_W_corners)


if __name__=="__main__":
    # main()
    main_video()
