import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from parabolaRansac import parabolaRansac
from ransacLocalization import ransacLocalization
from code_previous_exercises.describeKeypoints import describeKeypoints
from code_previous_exercises.harris import harris
from code_previous_exercises.matchDescriptors import matchDescriptors
from code_previous_exercises.selectKeypoints import selectKeypoints
from code_previous_exercises.plotMatches import plotMatches
from code_previous_exercises.draw_camera import drawCamera


# Data for parts 3 and 4
K = np.loadtxt('../../data/K.txt')
keypoints = np.loadtxt('../../data/keypoints.txt').T
p_W_landmarks = np.loadtxt('../../data/p_W_landmarks.txt')

# Data for part 4
database_image = cv2.imread('../../data/000000.png', cv2.IMREAD_GRAYSCALE)

# Parts 2 and 3 - Localization with RANSAC + DLT/P3P
query_image = cv2.imread('../../data/000001.png', cv2.IMREAD_GRAYSCALE)

# Parameters from exercise 3.
harris_patch_size = 9
harris_kappa = 0.08
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 5

# Other parameters
num_keypoints = 1000

# Detect and match keypoints.
database_keypoints = keypoints
query_harris = harris(query_image, harris_patch_size, harris_kappa)
query_keypoints = selectKeypoints(query_harris, num_keypoints, nonmaximum_supression_radius)
query_descriptors = describeKeypoints(query_image, query_keypoints, descriptor_radius)
database_descriptors = describeKeypoints(database_image, database_keypoints, descriptor_radius)
all_matches = matchDescriptors(query_descriptors, database_descriptors, match_lambda)

# Drop unmatched keypoints and get 3d landmarks for the matched ones.
matched_query_keypoints = query_keypoints[:, all_matches >= 0]
corresponding_matches = all_matches[all_matches >= 0]
corresponding_landmarks = p_W_landmarks[corresponding_matches, :]

# perform RANSAC to find best Pose and inliers
out = ransacLocalization(matched_query_keypoints, corresponding_landmarks, K)
R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = out

print('Found transformation T_C_W = ')
print(np.concatenate([R_C_W, t_C_W[:, None]], axis=-1))
print('Estimated inlier ratio is')
print(inlier_mask.mean())

matched_query_keypoints = query_keypoints[:, all_matches >= 0]
corresponding_matches = all_matches[all_matches >= 0]

plt.clf()
plt.close()
plt.plot(np.arange(len(num_iteration_history)))
plt.plot(num_iteration_history)
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Estimated Max Number of Iterations")
plt.show()

plt.clf()
plt.close()
fig, axs = plt.subplots(3, 1)
axs[0].imshow(query_image, cmap='gray')
axs[0].plot(query_keypoints[1, :], query_keypoints[0, :], 'rx', linewidth=2)
plotMatches(all_matches, query_keypoints, keypoints, axs[0])
axs[0].set_title('All keypoints and matches')
axs[0].axis('off')

axs[1].imshow(query_image, cmap='gray')
axs[1].plot(matched_query_keypoints[1, np.logical_not(inlier_mask)], matched_query_keypoints[0, np.logical_not(inlier_mask)],
            'rx', linewidth=2)
axs[1].plot(matched_query_keypoints[1, inlier_mask], matched_query_keypoints[0, inlier_mask], 'rx', linewidth=2)
plotMatches(corresponding_matches[inlier_mask], matched_query_keypoints[:, inlier_mask], keypoints, axs[1])
axs[1].set_title('Inlier and outlier matches')
axs[1].axis('off')

axs[2].plot(max_num_inliers_history)
axs[2].set_title('Maximum inlier count over RANSAC iterations.')

plt.tight_layout()
plt.show()


plt.clf()
plt.close()
fig = plt.figure()

for i in range(9):
    fig.clear()
    query_image = cv2.imread('../../data/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)

    # Detect and match keypoints
    database_keypoints = keypoints
    query_harris = harris(query_image, harris_patch_size, harris_kappa)
    query_keypoints = selectKeypoints(query_harris, num_keypoints, nonmaximum_supression_radius)
    query_descriptors = describeKeypoints(query_image, query_keypoints, descriptor_radius)
    database_descriptors = describeKeypoints(database_image, database_keypoints, descriptor_radius)
    all_matches = matchDescriptors(query_descriptors, database_descriptors, match_lambda)

    # Drop unmatched keypoints and get 3d landmarks for the matched ones.
    matched_query_keypoints = query_keypoints[:, all_matches >= 0]
    corresponding_matches = all_matches[all_matches >= 0]
    corresponding_landmarks = p_W_landmarks[corresponding_matches, :]

    # perform RANSAC to find best Pose and inliers
    out = ransacLocalization(matched_query_keypoints, corresponding_landmarks, K)
    R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = out

    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(query_image, cmap='gray')
    ax.plot(matched_query_keypoints[1, np.logical_not(inlier_mask)],
                matched_query_keypoints[0, np.logical_not(inlier_mask)],
                'rx', linewidth=2)
    if inlier_mask.sum() > 0:
        ax.plot(matched_query_keypoints[1, inlier_mask], matched_query_keypoints[0, inlier_mask], 'rx', linewidth=2)
        plotMatches(corresponding_matches[inlier_mask], matched_query_keypoints[:, inlier_mask], keypoints, ax)
    ax.set_title('Inlier and outlier matches')
    ax.axis('off')

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.set_xlim3d(-15, 10)
    ax.set_ylim3d(-10, 5)
    ax.set_zlim3d(-1, 40)
    ax.scatter(p_W_landmarks[:, 0], p_W_landmarks[:, 1], p_W_landmarks[:, 2], s=0.5)
    if R_C_W is not None:
        drawCamera(ax, -np.matmul(R_C_W.T, t_C_W), R_C_W.T, length_scale=10, head_size=10, set_ax_limits=True)
        print('Frame {} localized with {} inliers!'.format(i, inlier_mask.sum()))
    else:
        print('Frame {} failed to localize!'.format(i))

    plt.pause(1)

