import numpy as np


def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    padded = np.lib.pad(img, (r, r), 'constant', constant_values=(0, 0))
    _, num_keypoints = keypoints.shape
    descriptors = []
    for i in range(num_keypoints):
        temp_pos = keypoints[:, i] + r
        temp_descriptor = padded[temp_pos[0] - r:temp_pos[0] + r + 1, temp_pos[1] - r:temp_pos[1] + r + 1]
        descriptors.append(temp_descriptor)

    descriptors = np.array(descriptors)
    descriptors = descriptors.reshape([num_keypoints, (2 * r + 1) ** 2])
    descriptors = descriptors.T

    return descriptors



