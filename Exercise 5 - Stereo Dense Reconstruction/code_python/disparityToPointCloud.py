import numpy as np


def disparityToPointCloud(disp_img, K, baseline, left_img):
    """
    points should be Nx3 and intensities N, where N is the amount of pixels which have a valid disparity.
    I.e., only return points and intensities for pixels of left_img which have a valid disparity estimate!
    The i-th intensity should correspond to the i-th point.
    """

    n_rows, n_cols = disp_img.shape
    X, Y = np.meshgrid(np.arange(1, n_cols + 1), np.arange(1, n_rows + 1))
    X = X.reshape(n_cols * n_rows)
    Y = Y.reshape(n_cols * n_rows)

    l_pix = np.stack([X, Y, np.ones_like(X)], axis=1).astype(np.float)
    r_pix = l_pix.copy()
    r_pix[:, 0] -= disp_img.reshape(n_rows * n_cols)

    disp_flag = disp_img.reshape(n_rows * n_cols) > 0
    l_pix = l_pix[disp_flag, :]
    r_pix = r_pix[disp_flag, :]

    intensities = left_img[disp_flag.reshape([n_rows, n_cols])]

    K_inv = np.linalg.inv(K)

    A_left = np.matmul(K_inv, l_pix[:, :, None]).squeeze(-1)
    A_right = np.matmul(K_inv, r_pix[:, :, None]).squeeze(-1)

    b = np.asarray([baseline, 0, 0])

    A = np.stack([A_left, -A_right], axis=-1)
    A_pinv = np.linalg.pinv(A)
    lambd = np.matmul(A_pinv, b[:, None])

    points = A_left * lambd[:, 0, :]

    return points, intensities
