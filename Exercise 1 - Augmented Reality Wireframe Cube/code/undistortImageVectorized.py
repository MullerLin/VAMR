import numpy as np

from distortPoints import distortPoints


def undistortImageVectorized(img, K, D):
    height, width = img.shape[-2:]
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    px_locs = np.stack([X, Y], axis=-1).reshape([height*width, 2])

    dist_px_locs = distortPoints(px_locs, D, K)
    intensity_vals = img[np.round(dist_px_locs[:, 1].astype(np.int)),
                         np.round(dist_px_locs[:, 0]).astype(np.int)]
    undimg = intensity_vals.reshape(img.shape).astype(np.uint8)

    return undimg
